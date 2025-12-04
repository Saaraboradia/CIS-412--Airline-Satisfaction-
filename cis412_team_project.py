# app.py
"""
Interactive Streamlit app for CIS412 — What-If Dashboard
- Sensitivity / one-click removed
- Live confusion matrix graphics (seaborn heatmaps) instead of tables
"""

import os
import warnings
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_theme()
st.set_page_config(layout="wide", page_title="CIS412 - Interactive What-If Dashboard")

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# Helper functions
# -------------------------
def load_csv(uploaded) -> Optional[pd.DataFrame]:
    """Load dataframe from uploaded file or common repo paths."""
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV read successfully.")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            return None
    for p in ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                st.sidebar.info(f"Loaded CSV from {p}")
                return df
            except Exception as e:
                st.sidebar.error(f"Failed to read {p}: {e}")
                return None
    st.sidebar.warning("No CSV found. Upload train.csv via the uploader or add train.csv to the repo.")
    return None

def detect_and_convert_target(df: pd.DataFrame, target_hint: Optional[str]=None) -> Optional[pd.Series]:
    """Detect a satisfaction-like target and convert to integer Series."""
    df = df.copy()
    # explicit hint
    if target_hint and target_hint in df.columns:
        col = df[target_hint]
    else:
        # prefer explicit name
        candidates = [c for c in df.columns if 'satisfaction' in c.lower()]
        if len(candidates) == 1:
            col = df[candidates[0]]
        elif 'satisfaction' in df.columns:
            col = df['satisfaction']
        else:
            # try any binary-looking column
            possible = []
            for c in df.columns:
                vals = df[c].dropna().unique()
                if set(vals) <= {0,1} or set(map(str,vals)) <= {'0','1'}:
                    possible.append(c)
            if possible:
                col = df[possible[0]]
            else:
                # give up - return None
                return None

    # now robustly convert col to binary ints
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        # coerce numeric (fill NaNs with mode or 0)
        if col.isnull().any():
            if not col.mode().empty:
                col = col.fillna(col.mode().iloc[0])
            else:
                col = col.fillna(0)
        try:
            return col.astype(int).copy()
        except Exception:
            return pd.to_numeric(col, errors='coerce').fillna(0).astype(int).copy()
    else:
        # text labels: map contains('satisf') -> 1, contains('dissat'|'neutral'|'not') ->0; else factorize fallback
        s = col.astype(str).str.strip().str.lower()
        mapped = pd.Series([None]*len(s), index=s.index)
        mapped[s.str.contains('satisf', na=False)] = 1
        mapped[s.str.contains('dissat', na=False) | s.str.contains('not satisfied', na=False) | s.str.contains('unsatisf', na=False) | s.str.contains('disloyal', na=False) | s.str.contains('neutral', na=False)] = 0
        if mapped.notna().sum() >= max(2, int(0.05 * len(mapped))):  # if mapping worked for some reasonable fraction
            # fill remaining with mode (if any)
            if mapped.mode().shape[0] > 0:
                mapped = mapped.fillna(mapped.mode().iloc[0])
            else:
                mapped = mapped.fillna(0)
            return mapped.astype(int).copy()
        # fallback: factorize into integer codes
        codes, uniques = pd.factorize(col)
        st.warning(f"Target column was non-numeric and mapping heuristics incomplete. Factorized into integer codes; uniques: {list(uniques)}")
        return pd.Series(codes, index=col.index).astype(int).copy()

def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric NaN by median, categorical NaN by mode; clamp negative delays."""
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include=['object','category']).columns:
        if df[c].isnull().any() and df[c].mode().shape[0]:
            df[c].fillna(df[c].mode().iloc[0], inplace=True)
    for d in ['Departure Delay in Minutes','Arrival Delay in Minutes']:
        if d in df.columns:
            df[d] = df[d].apply(lambda x: x if pd.isna(x) else max(x,0))
    return df

def plot_prob_bar(prob_lr: float, prob_rf: float):
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.barh(['Logistic','RandomForest'], [prob_lr, prob_rf], color=['#2b8cbe','#fdae6b'])
    ax.set_xlim(0,1)
    for i,p in enumerate([prob_lr, prob_rf]):
        ax.text(p + 0.02, i, f"{p:.3f}", va='center')
    ax.set_xlabel("Predicted probability (class=1)")
    st.pyplot(fig)

def plot_rf_importances(rf_model, feature_names):
    imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(imp)*0.12)))
    imp.plot(kind='barh', ax=ax)
    ax.set_title("RandomForest feature importances (fallback)")
    st.pyplot(fig)

def plot_confusion_heatmap(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    st.pyplot(fig)

# -------------------------
# UI: sidebar controls
# -------------------------
st.sidebar.title("CIS412 — Controls")
uploaded = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"], key="uploader_main")
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True, key="scale_num")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05, key="test_size")
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1, key="rand_seed"))
show_debug = st.sidebar.checkbox("Show prediction debug info", value=False, key="show_debug")
use_shap = st.sidebar.checkbox("Enable SHAP (may be slow)", value=False, key="use_shap")

# Load & preprocess data
df_raw = load_csv(uploaded)
if df_raw is None:
    st.title("CIS412 — Interactive What-If Dashboard")
    st.write("Upload `train.csv` using the uploader in the sidebar or add `train.csv` to the repo root.")
    st.stop()

st.header("Raw Data Preview")
st.write(f"Shape: {df_raw.shape}")
st.dataframe(df_raw.head())

df_clean = basic_impute(df_raw)

# detect & convert target
y_series = detect_and_convert_target(df_clean)
if y_series is None:
    st.error("Could not automatically detect a satisfaction-like target column. Ensure a 'satisfaction' or 'satisfaction_satisfied' column exists.")
    st.stop()

# append converted target to processed df
df_processed = df_clean.copy()
df_processed['_target_converted'] = y_series.values
y_full = df_processed['_target_converted']

st.write("Detected/converted target distribution:")
st.write(pd.Series(y_full).value_counts().to_frame('count'))

# one-hot encode remaining categorical columns (excluding our _target_converted)
obj_cols = [c for c in df_processed.select_dtypes(include=['object','category']).columns if c != '_target_converted']
if obj_cols:
    df_processed = pd.get_dummies(df_processed, columns=obj_cols, drop_first=True)

# features list (exclude the target helper column)
all_features = [c for c in df_processed.columns if c != '_target_converted']

# heuristic default features (service-like)
rating_keywords = ['inflight','wifi','checkin','food','clean','baggage','seat','boarding','comfort','entertainment','service','leg room','ease','gate']
service_candidates = [c for c in df_clean.columns if any(k in c.lower() for k in rating_keywords) and pd.api.types.is_numeric_dtype(df_clean[c])]
service_candidates = [c for c in service_candidates if c in all_features]
default_features = service_candidates + [c for c in all_features if c not in service_candidates][:20]

st.sidebar.header("Model features")
chosen_features = st.sidebar.multiselect("Features to include for modeling", options=all_features, default=default_features[:min(20,len(default_features))], key="chosen_feats")
if not chosen_features:
    st.error("Select at least one feature in the sidebar to proceed.")
    st.stop()

# Prepare modeling data (X_full, y_full)
X_full = df_processed[chosen_features].copy()
y_full = y_full.astype(int).copy()

# train/test split & cleaning
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full if y_full.nunique()>1 else None)

def clean_impute_df(df_to_clean: pd.DataFrame, ref: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    dfc = df_to_clean.copy()
    dfc.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in dfc.columns:
        if dfc[c].isnull().any():
            med = (ref[c].median() if (ref is not None and c in ref.columns) else dfc[c].median())
            dfc[c].fillna(med, inplace=True)
    return dfc

X_train = clean_impute_df(X_train)
X_test  = clean_impute_df(X_test, ref=X_train)

# ensure numeric dtypes and drop non-convertible
to_drop = []
for col in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        try:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col]  = pd.to_numeric(X_test[col], errors='coerce')
            if X_train[col].isnull().any():
                X_train[col].fillna(X_train[col].median(), inplace=True)
            if X_test[col].isnull().any():
                X_test[col].fillna(X_train[col].median(), inplace=True)
        except Exception:
            to_drop.append(col)
if to_drop:
    st.warning(f"Dropping non-numeric columns from features: {to_drop}")
    X_train.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

# scaling
scaler = None
if scale_numeric and numeric_cols:
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

# show class counts
st.write("Target class distribution (train):")
st.write(pd.Series(y_train).value_counts().to_frame('count'))

# Fit models robustly
n_classes = y_train.nunique()
if n_classes <= 1:
    st.warning("y_train contains only one class; training a DummyClassifier that predicts the majority class.")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    lr_model = dummy
    rf_model = dummy
    st.session_state['model_is_dummy'] = True
else:
    try:
        if n_classes > 2:
            lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
        else:
            lr_model = LogisticRegression(max_iter=2000)
        lr_model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"LogisticRegression failed to fit: {e}. Falling back to DummyClassifier.")
        lr_model = DummyClassifier(strategy="most_frequent")
        lr_model.fit(X_train, y_train)

    try:
        rf_model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"RandomForest failed to fit: {e}. Falling back to DummyClassifier.")
        rf_model = DummyClassifier(strategy="most_frequent")
        rf_model.fit(X_train, y_train)
    st.session_state['model_is_dummy'] = False

# Save models & metadata
st.session_state['lr_model'] = lr_model
st.session_state['rf_model'] = rf_model
st.session_state['scaler'] = scaler
st.session_state['model_feature_order'] = X_train.columns.tolist()
st.session_state['X_train'] = X_train
st.session_state['X_test'] = X_test
st.session_state['y_train'] = y_train
st.session_state['y_test'] = y_test

st.success("Models ready. Use sliders below to perform what-if predictions.")

# quick metrics
def quick_stats(m, Xtr, Xte, ytr, yte):
    tr = m.score(Xtr, ytr)
    te = m.score(Xte, yte)
    try:
        te_auc = roc_auc_score(yte, m.predict_proba(Xte)[:,1])
    except Exception:
        te_auc = np.nan
    return tr, te, te_auc

lr_tr, lr_te, lr_auc = quick_stats(lr_model, X_train, X_test, y_train, y_test)
rf_tr, rf_te, rf_auc = quick_stats(rf_model, X_train, X_test, y_train, y_test)

st.subheader("Performance (quick)")
st.write(pd.DataFrame({
    'Model':['Logistic','RandomForest'],
    'Train acc':[lr_tr, rf_tr],
    'Test acc':[lr_te, rf_te],
    'Test ROC AUC':[lr_auc, rf_auc]
}).set_index('Model'))

# -------------------------
# Interactive sliders (expose service columns)
# -------------------------
st.header("Interactive sliders — tweak service features")

slider_candidates = [c for c in service_candidates if c in chosen_features]
if not slider_candidates:
    slider_candidates = [c for c in chosen_features if pd.api.types.is_numeric_dtype(X_full[c])][:8]

slider_values: Dict[str, Any] = {}
cols_left, cols_right = st.columns(2)
for i, feat in enumerate(slider_candidates):
    col = cols_left if i % 2 == 0 else cols_right
    if feat in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[feat]):
        lo = int(df_raw[feat].min()); hi = int(df_raw[feat].max()); default = int(df_raw[feat].median())
    else:
        ser = X_full[feat]
        lo = int(ser.min()) if pd.api.types.is_numeric_dtype(ser) else 0
        hi = int(ser.max()) if pd.api.types.is_numeric_dtype(ser) else 5
        default = int(ser.median()) if pd.api.types.is_numeric_dtype(ser) else 0
    slider_values[feat] = col.slider(feat, min_value=lo, max_value=hi, value=default, step=1, key=f"slider_{feat}")

with st.expander("Edit additional numeric features (optional)"):
    other_numeric = [c for c in chosen_features if c not in slider_candidates and pd.api.types.is_numeric_dtype(X_full[c])]
    other_inputs = {}
    cols2 = st.columns(2)
    for i, feat in enumerate(other_numeric[:10]):
        col = cols2[i % 2]
        ser = X_full[feat]
        lo, hi = float(ser.min()), float(ser.max())
        default = float(ser.median())
        other_inputs[feat] = col.number_input(feat, value=default, min_value=lo, max_value=hi, key=f"num_{feat}")

# -------------------------
# Build model input from sliders robustly and predict
# -------------------------
st.header("Live prediction")
threshold = st.slider("Decision threshold (to produce class=1)", 0.01, 0.99, 0.5, 0.01, key="threshold")

model_feats: List[str] = st.session_state['model_feature_order']
base_input = {f: float(pd.to_numeric(X_full[f], errors='coerce').median()) if f in X_full.columns else 0.0 for f in model_feats}

for sname, sval in slider_values.items():
    if sname in base_input:
        base_input[sname] = float(sval)
    else:
        lowered = sname.lower()
        matches = [m for m in model_feats if lowered in m.lower() or m.lower() in lowered]
        for mm in matches:
            base_input[mm] = float(sval)

for name, val in (other_inputs.items() if 'other_inputs' in locals() else {}):
    if name in base_input:
        base_input[name] = float(val)

input_df = pd.DataFrame([base_input], columns=model_feats)

scaler_saved = st.session_state.get('scaler', None)
numeric_cols_model = [c for c in model_feats if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]
if scaler_saved is not None and len(numeric_cols_model) > 0:
    try:
        input_df[numeric_cols_model] = scaler_saved.transform(input_df[numeric_cols_model])
    except Exception:
        try:
            tmp = StandardScaler()
            tmp.fit(pd.concat([st.session_state['X_train'][numeric_cols_model], st.session_state['X_test'][numeric_cols_model]], axis=0))
            input_df[numeric_cols_model] = tmp.transform(input_df[numeric_cols_model])
            st.session_state['scaler'] = tmp
        except Exception:
            pass

if show_debug:
    st.subheader("Debug: input passed to model (post-scaling)")
    st.write("Model features (order):")
    st.write(model_feats)
    st.dataframe(input_df.T)

try:
    prob_lr = float(st.session_state['lr_model'].predict_proba(input_df)[:,1][0])
    prob_rf = float(st.session_state['rf_model'].predict_proba(input_df)[:,1][0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

cls_lr = int(prob_lr >= threshold)
cls_rf = int(prob_rf >= threshold)

st.subheader("Predicted probabilities (reflect slider inputs)")
plot_prob_bar(prob_lr, prob_rf)
st.write(f"Logistic predicted class: {cls_lr} | RandomForest predicted class: {cls_rf}")
prob_lr = lr.predict_proba(input_scaled)[0, 1]
prob_rf = rf.predict_proba(input_scaled)[0, 1]
# ---- Prediction Result Section ----
st.subheader("Final Prediction Result")

# Choose a model to report final class (use Random Forest by default)
threshold = 0.5  # or let this come from a slider earlier in your app
pred_class = 1 if prob_rf >= threshold else 0

if pred_class == 1:
    st.success(f"🎉 Prediction: **SATISFIED** (probability = {prob_rf:.3f})")
else:
    st.error(f"⚠️ Prediction: **NOT SATISFIED** (probability = {prob_rf:.3f})")

# -------------------------
# Graphical confusion matrices (replacing textual tables)
# -------------------------
st.header("Confusion matrices (test set) - live graphics")

# Logistic confusion matrix graphic
st.subheader("Logistic Regression - Confusion Matrix (test)")
y_pred_lr_test = st.session_state['lr_model'].predict(st.session_state['X_test'])
plot_confusion_heatmap(st.session_state['y_test'], y_pred_lr_test, title="Logistic Regression - Confusion Matrix (test)")

# Random Forest confusion matrix graphic
st.subheader("Random Forest - Confusion Matrix (test)")
y_pred_rf_test = st.session_state['rf_model'].predict(st.session_state['X_test'])
plot_confusion_heatmap(st.session_state['y_test'], y_pred_rf_test, title="Random Forest - Confusion Matrix (test)")

# Also show classification report as optional expandable text (kept but not primary)
with st.expander("Show classification reports (text)"):
    st.write("Logistic regression (test):")
    st.text(classification_report(st.session_state['y_test'], y_pred_lr_test, zero_division=0))
    st.write("Random Forest (test):")
    st.text(classification_report(st.session_state['y_test'], y_pred_rf_test, zero_division=0))

# -------------------------
# Feature importance / SHAP
# -------------------------
st.header("Feature importance & explanation")
if use_shap and HAS_SHAP:
    try:
        st.write("Computing SHAP values (TreeExplainer on RF). This may take a moment.")
        explainer = shap.TreeExplainer(st.session_state['rf_model'])
        sample = st.session_state['X_train'].sample(n=min(200, st.session_state['X_train'].shape[0]), random_state=1)
        shap_vals = explainer.shap_values(sample)
        arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        mean_abs = np.abs(arr).mean(axis=0)
        s = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=True).tail(30)
        fig, ax = plt.subplots(figsize=(8, max(3, len(s)*0.25)))
        s.plot(kind='barh', ax=ax)
        ax.set_title("Top mean |SHAP value| (RF)")
        st.pyplot(fig)
        # local explanation for current input
        local = explainer.shap_values(input_df)
        local_arr = local[1] if isinstance(local, list) else local
        df_local = pd.DataFrame(local_arr.reshape(1,-1), columns=input_df.columns).T
        df_local.columns = ['SHAP value']
        st.subheader("Local SHAP values (current input)")
        st.dataframe(df_local.sort_values('SHAP value', ascending=False))
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}. Showing RF importances instead.")
        plot_rf_importances(st.session_state['rf_model'], st.session_state['X_train'].columns)
else:
    st.info("SHAP not enabled or not installed. Displaying RandomForest feature importances.")
    plot_rf_importances(st.session_state['rf_model'], st.session_state['X_train'].columns)

st.success("Interactive dashboard ready — move sliders and view live confusion matrix graphics.")

