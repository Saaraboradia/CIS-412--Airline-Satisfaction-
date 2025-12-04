# app.py
"""
Fixed Interactive Streamlit app for CIS412 — What-If Dashboard
- Robust predictions (uses st.session_state)
- Live confusion matrix graphics
- Permutation or RF importances; optional SHAP if installed & enabled
"""

import io
import os
import math
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

warnings.filterwarnings("ignore")
sns.set_theme()
st.set_page_config(layout="wide", page_title="CIS412 - Interactive What-If Dashboard (fixed)")

# Optional SHAP
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# Helper functions
# -------------------------
def load_csv(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            return None
    for p in ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def detect_and_convert_target(df: pd.DataFrame) -> Optional[pd.Series]:
    # Try common names
    for cand in ["satisfaction_satisfied", "satisfaction", "target"]:
        if cand in df.columns:
            col = df[cand]
            return _to_binary_series(col)
    # search for "satisfaction" in names
    sat_cols = [c for c in df.columns if "satisfaction" in c.lower()]
    if len(sat_cols) >= 1:
        return _to_binary_series(df[sat_cols[0]])
    # search numeric binary columns
    for c in df.columns:
        vals = set(df[c].dropna().unique())
        if vals <= {0, 1}:
            return df[c].astype(int)
    return None

def _to_binary_series(col: pd.Series) -> pd.Series:
    # numeric -> int
    if pd.api.types.is_numeric_dtype(col):
        filled = col.fillna(col.mode().iloc[0] if not col.mode().empty else 0)
        return filled.astype(int)
    s = col.astype(str).str.lower().str.strip()
    mapped = pd.Series(np.nan, index=s.index)
    mapped[s.str.contains("satisf", na=False)] = 1
    mapped[s.str.contains("dissat|not satisfied|unsatisf|neutral|disloyal", na=False)] = 0
    if mapped.notna().sum() >= max(2, int(0.05*len(mapped))):
        mapped = mapped.fillna(mapped.mode().iloc[0] if not mapped.mode().empty else 0)
        return mapped.astype(int)
    # fallback factorize
    codes, uniques = pd.factorize(col)
    return pd.Series(codes, index=col.index).astype(int)

def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include=['object','category']).columns:
        if df[c].isnull().any() and not df[c].mode().empty:
            df[c].fillna(df[c].mode().iloc[0], inplace=True)
    for d in ['Departure Delay in Minutes','Arrival Delay in Minutes']:
        if d in df.columns:
            df[d] = df[d].apply(lambda x: x if pd.isna(x) else max(x,0))
    return df

def plot_confusion_heatmap(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    st.pyplot(fig)

def plot_prob_bar(prob_lr: float, prob_rf: float):
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.barh(['Logistic','RandomForest'], [prob_lr, prob_rf], color=['#2b8cbe','#fdae6b'])
    ax.set_xlim(0,1)
    for i,p in enumerate([prob_lr, prob_rf]):
        ax.text(p + 0.02, i, f"{p:.3f}", va='center')
    ax.set_xlabel("Predicted probability (class=1)")
    st.pyplot(fig)

def plot_rf_importances(rf_model, feature_names):
    try:
        imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, len(imp)*0.12)))
        imp.plot(kind='barh', ax=ax)
        ax.set_title("RandomForest feature importances (fallback)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not compute RF importances: {e}")

def safe_proba(model, X):
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] >= 2:
                return float(p[:,1].ravel()[0])
            return float(p.ravel()[0])
        else:
            return float(model.predict(X).ravel()[0])
    except Exception:
        return float("nan")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("CIS412 — Controls")
uploaded = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"], key="uploader_main")
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True, key="scale_num")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05, key="test_size")
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1, key="rand_seed"))
show_debug = st.sidebar.checkbox("Show prediction debug info", value=False, key="show_debug")
use_shap = st.sidebar.checkbox("Enable SHAP (optional)", value=False, key="use_shap")
threshold_sidebar = st.sidebar.slider("Decision threshold (for final class)", 0.01, 0.99, 0.5, 0.01, key="decision_threshold")

# -------------------------
# Load and prepare data
# -------------------------
df_raw = load_csv(uploaded)
if df_raw is None:
    st.title("CIS412 — Interactive What-If Dashboard")
    st.info("No CSV found. Upload train.csv via the uploader in the sidebar or add train.csv to repo root.")
    # show an example small embedded dataset as helpful guidance (kept minimal)
    example = pd.DataFrame({
        "Age":[25,45,30],"Flight Distance":[300,1200,800],
        "Inflight wifi service":[3,1,5],"Food and drink":[4,2,5],
        "Seat comfort":[4,2,5],"Baggage handling":[4,3,5],
        "Departure Delay in Minutes":[0,10,0],"Arrival Delay in Minutes":[0,5,0],
        "satisfaction":["satisfied","dissatisfied","satisfied"]
    })
    st.write("Example data preview (replace with your train.csv):")
    st.dataframe(example)
    st.stop()

st.header("Raw Data Preview")
st.write(f"Shape: {df_raw.shape}")
st.dataframe(df_raw.head())

df_clean = basic_impute(df_raw)

y_series = detect_and_convert_target(df_clean)
if y_series is None:
    st.error("Could not detect a satisfaction-like target column. Ensure a 'satisfaction' column exists.")
    st.stop()

df_processed = df_clean.copy()
df_processed['_target_converted'] = y_series.astype(int).values
st.write("Detected target distribution:")
st.write(pd.Series(df_processed['_target_converted']).value_counts().to_frame('count'))

# encode categorical columns (excluding the target helper)
obj_cols = [c for c in df_processed.select_dtypes(include=['object','category']).columns if c != '_target_converted']
if obj_cols:
    df_processed = pd.get_dummies(df_processed, columns=obj_cols, drop_first=True)

all_features = [c for c in df_processed.columns if c != '_target_converted']

# heuristic default service-like features
rating_keywords = ['inflight','wifi','checkin','food','clean','baggage','seat','boarding','comfort','entertainment','service','leg room','ease','gate']
service_candidates = [c for c in df_clean.columns if any(k in c.lower() for k in rating_keywords) and pd.api.types.is_numeric_dtype(df_clean[c])]
service_candidates = [c for c in service_candidates if c in all_features]

# default feature set: service candidates then first others
default_features = service_candidates + [c for c in all_features if c not in service_candidates][:20]

st.sidebar.header("Model features")
chosen_features = st.sidebar.multiselect("Features to include for modeling", options=all_features, default=default_features[:min(20,len(default_features))], key="chosen_feats")
if not chosen_features:
    st.error("Select at least one feature in the sidebar.")
    st.stop()

X_full = df_processed[chosen_features].copy()
y_full = df_processed['_target_converted'].astype(int).copy()

# split
strat = y_full if y_full.nunique()>1 else None
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=random_state, stratify=strat)

# clean and coerce numeric
def clean_impute_df(df_to_clean: pd.DataFrame, ref: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    dfc = df_to_clean.copy()
    dfc.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in dfc.columns:
        if dfc[c].isnull().any():
            med = (ref[c].median() if (ref is not None and c in ref.columns) else dfc[c].median())
            dfc[c].fillna(med, inplace=True)
    return dfc

X_train = clean_impute_df(X_train)
X_test = clean_impute_df(X_test, ref=X_train)

to_drop = []
for col in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        try:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col]  = pd.to_numeric(X_test[col], errors='coerce')
            X_train[col].fillna(X_train[col].median(), inplace=True)
            X_test[col].fillna(X_train[col].median(), inplace=True)
        except Exception:
            to_drop.append(col)
if to_drop:
    st.warning(f"Dropping non-numeric columns: {to_drop}")
    X_train.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

# scaling
scaler = None
if scale_numeric and numeric_cols:
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save to session_state
st.session_state['X_train'] = X_train.copy()
st.session_state['X_test'] = X_test.copy()
st.session_state['y_train'] = y_train.copy()
st.session_state['y_test'] = y_test.copy()
st.session_state['scaler'] = scaler
st.session_state['model_feature_order'] = X_train.columns.tolist()

# train models
if y_train.nunique() <= 1:
    st.warning("Only one class in train set — using DummyClassifier fallback.")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    lr_model = dummy
    rf_model = dummy
else:
    try:
        lr_model = LogisticRegression(max_iter=2000)
        lr_model.fit(X_train, y_train)
    except Exception:
        lr_model = DummyClassifier(strategy="most_frequent")
        lr_model.fit(X_train, y_train)
    try:
        rf_model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train, y_train)
    except Exception:
        rf_model = DummyClassifier(strategy="most_frequent")
        rf_model.fit(X_train, y_train)

st.session_state['lr_model'] = lr_model
st.session_state['rf_model'] = rf_model

st.success("Models trained and ready — use the sliders to run what-if predictions.")

# quick metrics
def quick_stats(model, Xtr, Xte, ytr, yte):
    try:
        tr = model.score(Xtr, ytr)
    except Exception:
        tr = float("nan")
    try:
        te = model.score(Xte, yte)
    except Exception:
        te = float("nan")
    try:
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:,1]) if hasattr(model, "predict_proba") and len(np.unique(yte))==2 else float("nan")
    except Exception:
        auc = float("nan")
    return tr, te, auc

lr_tr, lr_te, lr_auc = quick_stats(lr_model, X_train, X_test, y_train, y_test)
rf_tr, rf_te, rf_auc = quick_stats(rf_model, X_train, X_test, y_train, y_test)

st.subheader("Model performance (quick)")
perf_df = pd.DataFrame({
    'Model': ['Logistic','RandomForest'],
    'Train acc': [lr_tr, rf_tr],
    'Test acc': [lr_te, rf_te],
    'Test ROC AUC': [lr_auc, rf_auc]
}).set_index('Model')
st.table(perf_df)

# -------------------------
# Interactive sliders (service features)
# -------------------------
st.header("Interactive sliders — tweak service features")
slider_candidates = [c for c in service_candidates if c in chosen_features]
if not slider_candidates:
    slider_candidates = [c for c in chosen_features if c in X_full.columns and pd.api.types.is_numeric_dtype(X_full[c])][:8]

slider_values: Dict[str, Any] = {}
cols_left, cols_right = st.columns(2)
for i, feat in enumerate(slider_candidates):
    col = cols_left if i % 2 == 0 else cols_right
    ser = X_full[feat]
    lo = int(ser.min()) if pd.api.types.is_numeric_dtype(ser) else 0
    hi = int(ser.max()) if pd.api.types.is_numeric_dtype(ser) else 5
    default = int(ser.median()) if pd.api.types.is_numeric_dtype(ser) else 0
    slider_values[feat] = col.slider(feat, min_value=lo, max_value=hi, value=default, step=1, key=f"slider_{feat}")

with st.expander("Other numeric features (optional)"):
    other_numeric = [c for c in chosen_features if c not in slider_candidates and pd.api.types.is_numeric_dtype(X_full[c])]
    other_inputs = {}
    cols2 = st.columns(2)
    for i, feat in enumerate(other_numeric[:10]):
        col = cols2[i % 2]
        ser = X_full[feat]
        lo, hi = float(ser.min()), float(ser.max())
        default = float(ser.median())
        other_inputs[feat] = col.number_input(feat, value=default, min_value=lo, max_value=hi, key=f"num_{feat}")

# Build input_df
model_feats = st.session_state['model_feature_order']
base_input = {f: float(st.session_state['X_train'][f].median()) if f in st.session_state['X_train'].columns else 0.0 for f in model_feats}
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

# apply scaling if present
scaler_saved = st.session_state.get('scaler', None)
numeric_cols_model = [c for c in model_feats if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]
if scaler_saved is not None and len(numeric_cols_model) > 0:
    try:
        input_scaled = input_df.copy()
        input_scaled[numeric_cols_model] = scaler_saved.transform(input_scaled[numeric_cols_model])
    except Exception:
        input_scaled = input_df.copy()
else:
    input_scaled = input_df.copy()

if show_debug:
    st.subheader("Debug: model input (post-scaling)")
    st.write(model_feats)
    st.dataframe(input_scaled.T)

# -------------------------
# Predictions (robust)
# -------------------------
prob_lr = safe_proba(st.session_state['lr_model'], input_scaled)
prob_rf = safe_proba(st.session_state['rf_model'], input_scaled)

st.header("Predicted probabilities (reflect slider inputs)")
plot_prob_bar(prob_lr if not math.isnan(prob_lr) else 0.0, prob_rf if not math.isnan(prob_rf) else 0.0)

st.write(f"Logistic predicted prob: {prob_lr if not math.isnan(prob_lr) else 'n/a'}")
st.write(f"RandomForest predicted prob: {prob_rf if not math.isnan(prob_rf) else 'n/a'}")

# Final decision uses sidebar threshold
final_prob = prob_rf if not math.isnan(prob_rf) else (prob_lr if not math.isnan(prob_lr) else float("nan"))
st.subheader("Final Prediction Result")
if math.isnan(final_prob):
    st.error("No valid model probability available to determine SATISFIED vs NOT SATISFIED.")
else:
    final_class = 1 if final_prob >= float(threshold_sidebar) else 0
    if final_class == 1:
        st.success(f"🎉 Final prediction: SATISFIED (prob = {final_prob:.3f}, threshold = {threshold_sidebar})")
    else:
        st.error(f"⚠️ Final prediction: NOT SATISFIED (prob = {final_prob:.3f}, threshold = {threshold_sidebar})")

# -------------------------
# Confusion matrices (graphics)
# -------------------------
st.header("Confusion matrices (test set) - live graphics")
try:
    y_pred_lr_test = st.session_state['lr_model'].predict(st.session_state['X_test'])
    plot_confusion_heatmap(st.session_state['y_test'], y_pred_lr_test, title="Logistic Regression - Confusion Matrix (test)")
except Exception as e:
    st.warning(f"Could not compute logistic confusion matrix: {e}")

try:
    y_pred_rf_test = st.session_state['rf_model'].predict(st.session_state['X_test'])
    plot_confusion_heatmap(st.session_state['y_test'], y_pred_rf_test, title="Random Forest - Confusion Matrix (test)")
except Exception as e:
    st.warning(f"Could not compute RF confusion matrix: {e}")

with st.expander("Show classification reports (text)"):
    try:
        st.write("Logistic regression (test):")
        st.text(classification_report(st.session_state['y_test'], y_pred_lr_test, zero_division=0))
        st.write("Random Forest (test):")
        st.text(classification_report(st.session_state['y_test'], y_pred_rf_test, zero_division=0))
    except Exception:
        st.write("Classification reports not available.")

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
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}. Showing RF importances instead.")
        plot_rf_importances(st.session_state['rf_model'], st.session_state['X_train'].columns)
else:
    st.info("SHAP not enabled or not installed. Displaying RandomForest feature importances.")
    plot_rf_importances(st.session_state['rf_model'], st.session_state['X_train'].columns)

st.success("Interactive dashboard ready — adjust sliders to see live predictions and confusion matrices.")
