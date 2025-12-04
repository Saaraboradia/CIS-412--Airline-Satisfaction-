# app.py
"""
Interactive Streamlit app for CIS412 — What-If Dashboard
Single-file app: upload CSV or use repo train.csv -> trains models -> interactive sliders -> live predictions
Features:
 - robust slider -> model input alignment (handles one-hot-ish names)
 - scaler saved & applied to inputs
 - sensitivity ("what-if") tests
 - SHAP explainer if shap is installed; fallback to RF feature importances
 - debug toggle to show exact input sent to model
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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_theme()
st.set_page_config(layout="wide", page_title="CIS412 - What-If Dashboard")

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# Helper functions
# -------------------------
def load_csv_from_uploaded_or_repo(uploaded) -> Optional[pd.DataFrame]:
    """Load dataframe from uploaded file or fallback repo paths."""
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV read successfully.")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            return None
    # fallback file paths
    fallback_paths = ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]
    for p in fallback_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                st.sidebar.info(f"Loaded CSV from {p}")
                return df
            except Exception as e:
                st.sidebar.error(f"Failed reading {p}: {e}")
                return None
    st.sidebar.warning("No CSV found. Upload train.csv via the uploader or add train.csv to the repo.")
    return None

def detect_target(df: pd.DataFrame) -> Optional[str]:
    """Try to detect the satisfaction target column."""
    if 'satisfaction_satisfied' in df.columns:
        return 'satisfaction_satisfied'
    cands = [c for c in df.columns if 'satisfaction' in c.lower()]
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        vals = set(df[c].dropna().unique())
        if vals <= {0,1}:
            return c
    # fallback: if there's a column named 'satisfaction' exactly
    if 'satisfaction' in df.columns:
        return 'satisfaction'
    return None

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # numeric impute median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    # categorical impute mode
    for c in df.select_dtypes(include=['object','category']).columns:
        if df[c].isnull().any() and df[c].mode().shape[0]:
            df[c].fillna(df[c].mode()[0], inplace=True)
    # clamp negative delays if present
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

# -------------------------
# Main UI & flow
# -------------------------
st.title("CIS412 — Interactive What-If Dashboard")
st.write("Upload `train.csv` or place `train.csv` in the repo root. Use the sidebar to control options.")

# single uploader with explicit key (avoid duplicate IDs)
uploaded_file = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"], key="uploader_main")

# other sidebar options
st.sidebar.markdown("---")
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
show_debug = st.sidebar.checkbox("Show prediction debug info", value=False)
use_shap = st.sidebar.checkbox("Compute SHAP (may be slow)", value=False)

# Load data
df_raw = load_csv_from_uploaded_or_repo(uploaded_file)
if df_raw is None:
    st.stop()

st.subheader("Raw data preview")
st.write(f"Shape: {df_raw.shape}")
st.dataframe(df_raw.head())

# basic preprocessing
df_clean = basic_preprocess(df_raw)

# detect target
target_col = detect_target(df_clean)
if target_col is None:
    st.error("Could not detect a satisfaction-like target column. Please ensure a column named 'satisfaction' or 'satisfaction_satisfied' (binary) exists.")
    st.stop()
st.write("Detected target:", target_col)

# one-hot encode categorical columns (except the target)
obj_cols = [c for c in df_clean.select_dtypes(include=['object','category']).columns if c != target_col]
if obj_cols:
    df_processed = pd.get_dummies(df_clean, columns=obj_cols, drop_first=True)
else:
    df_processed = df_clean.copy()

# feature selection defaults
all_features = [c for c in df_processed.columns if c != target_col]
# heuristic: pick rating-like columns first
rating_keywords = ['inflight','wifi','checkin','food','clean','baggage','seat','boarding','comfort','entertainment','service','leg room','ease','gate']
service_candidates = [c for c in df_clean.columns if any(k in c.lower() for k in rating_keywords) and pd.api.types.is_numeric_dtype(df_clean[c])]
service_candidates = [c for c in service_candidates if c in all_features]  # ensure present after encoding
default_features = service_candidates + [c for c in all_features if c not in service_candidates][:20]

st.sidebar.header("Model features")
chosen_features = st.sidebar.multiselect("Features to include (choose at least 1)", options=all_features, default=default_features[:min(20,len(default_features))])

if not chosen_features:
    st.error("Please select at least one feature in the sidebar.")
    st.stop()

# Prepare X,y
X_full = df_processed[chosen_features].copy()
y_full = df_processed[target_col].astype(int).copy()

# split & scale (auto-train on load)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full if len(y_full.unique())>1 else None)

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
scaler = None
if scale_numeric and numeric_cols:
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train models
lr = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Save into session state for later use
st.session_state['lr_model'] = lr
st.session_state['rf_model'] = rf
st.session_state['scaler'] = scaler
st.session_state['model_feature_order'] = X_train.columns.tolist()
st.session_state['X_train'] = X_train
st.session_state['X_test'] = X_test
st.session_state['y_train'] = y_train
st.session_state['y_test'] = y_test

st.success("Models trained (auto). Use sliders below to run what-if predictions.")

# quick metrics
st.subheader("Quick performance metrics (on test set)")
def quick_stats(m, Xtr, Xte, ytr, yte):
    tr = m.score(Xtr, ytr)
    te = m.score(Xte, yte)
    try:
        te_auc = roc_auc_score(yte, m.predict_proba(Xte)[:,1])
    except Exception:
        te_auc = np.nan
    return tr, te, te_auc

lr_tr, lr_te, lr_auc = quick_stats(lr, X_train, X_test, y_train, y_test)
rf_tr, rf_te, rf_auc = quick_stats(rf, X_train, X_test, y_train, y_test)
st.write(pd.DataFrame({
    'Model':['Logistic','RandomForest'],
    'Train acc':[lr_tr, rf_tr],
    'Test acc':[lr_te, rf_te],
    'Test ROC AUC':[lr_auc, rf_auc]
}).set_index('Model'))

# -------------------------
# Build sliders UI
# -------------------------
st.header("Interactive sliders (service ratings)")
st.write("Sliders override a base input (median values). Move sliders and observe live predictions.")

# determine slider candidates: prefer service candidates present in chosen_features, else top numeric features
slider_candidates = [c for c in service_candidates if c in chosen_features]
if not slider_candidates:
    slider_candidates = [c for c in chosen_features if pd.api.types.is_numeric_dtype(X_full[c])][:8]

# build sliders
slider_values: Dict[str, Any] = {}
cols = st.columns(2)
for i, feat in enumerate(slider_candidates):
    col = cols[i % 2]
    if feat in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[feat]):
        lo = int(df_raw[feat].min()); hi = int(df_raw[feat].max()); default = int(df_raw[feat].median())
    else:
        ser = X_full[feat]
        lo = int(ser.min()) if pd.api.types.is_numeric_dtype(ser) else 0
        hi = int(ser.max()) if pd.api.types.is_numeric_dtype(ser) else 5
        default = int(ser.median()) if pd.api.types.is_numeric_dtype(ser) else 0
    slider_values[feat] = col.slider(feat, min_value=lo, max_value=hi, value=default, step=1)

with st.expander("Edit other numeric features (optional)"):
    other_numeric = [c for c in chosen_features if c not in slider_candidates and pd.api.types.is_numeric_dtype(X_full[c])]
    other_inputs = {}
    cols2 = st.columns(2)
    for i, feat in enumerate(other_numeric[:10]):
        col = cols2[i % 2]
        ser = X_full[feat]
        lo, hi = float(ser.min()), float(ser.max())
        default = float(ser.median())
        other_inputs[feat] = col.number_input(feat, value=default, min_value=lo, max_value=hi)

# -------------------------
# Build input vector robustly
# -------------------------
st.header("Live prediction")
threshold = st.slider("Threshold for class=1", 0.01, 0.99, 0.5, 0.01)

model_feats: List[str] = st.session_state['model_feature_order']
# base inputs = medians of training data (unscaled medians)
base_input = {}
for f in model_feats:
    if f in X_full.columns:
        base_input[f] = float(X_full[f].median())
    else:
        base_input[f] = 0.0

# apply slider overrides
for sname, sval in slider_values.items():
    if sname in base_input:
        base_input[sname] = float(sval)
    else:
        lowered = sname.lower()
        matches = [m for m in model_feats if lowered in m.lower() or m.lower() in lowered]
        for mm in matches:
            base_input[mm] = float(sval)

# other numeric overrides
for name, val in (other_inputs.items() if 'other_inputs' in locals() else {}):
    if name in base_input:
        base_input[name] = float(val)

# build DataFrame in model order
input_df = pd.DataFrame([base_input], columns=model_feats)

# scale with saved scaler if present
scaler_saved = st.session_state.get('scaler', None)
if scaler_saved is not None and numeric_cols:
    try:
        input_df[numeric_cols] = scaler_saved.transform(input_df[numeric_cols])
    except Exception:
        # fallback: attempt to fit a quick scaler on X_train numeric and transform
        try:
            tmp_s = StandardScaler()
            tmp_s.fit(pd.concat([st.session_state['X_train'][numeric_cols], st.session_state['X_test'][numeric_cols]], axis=0))
            input_df[numeric_cols] = tmp_s.transform(input_df[numeric_cols])
            st.session_state['scaler'] = tmp_s
        except Exception:
            pass

if show_debug:
    st.subheader("Debug: input passed to model")
    st.write("Model feature order (len={}):".format(len(model_feats)))
    st.write(model_feats)
    st.dataframe(input_df.T)

# Predictions
prob_lr = lr.predict_proba(input_df)[:,1][0]
prob_rf = rf.predict_proba(input_df)[:,1][0]
cls_lr = int(prob_lr >= threshold)
cls_rf = int(prob_rf >= threshold)

st.write("Predicted probabilities (reflect slider inputs):")
plot_prob_bar(prob_lr, prob_rf)
st.write(f"Logistic predicted class: {cls_lr}  |  RandomForest predicted class: {cls_rf}")

with st.expander("Show test set classification reports & confusion matrices"):
    st.write("Logistic regression (test):")
    st.text(classification_report(y_test, lr.predict(X_test), zero_division=0))
    st.write("Confusion matrix (Logistic):")
    st.write(confusion_matrix(y_test, lr.predict(X_test)))
    st.write("Random Forest (test):")
    st.text(classification_report(y_test, rf.predict(X_test), zero_division=0))
    st.write("Confusion matrix (RF):")
    st.write(confusion_matrix(y_test, rf.predict(X_test)))

# -------------------------
# Sensitivity analysis
# -------------------------
st.header("Sensitivity / What-if")
if slider_values:
    s_feature = st.selectbox("Feature to tweak", options=list(slider_values.keys()))
    current_val = base_input.get(s_feature, 0.0)
    new_val = st.number_input(f"Tweak {s_feature} to", value=int(current_val))
    if st.button("Run sensitivity (compare before / after)"):
        new_input = base_input.copy()
        if s_feature in new_input:
            new_input[s_feature] = float(new_val)
        else:
            lowered = s_feature.lower()
            matches = [m for m in model_feats if lowered in m.lower() or m.lower() in lowered]
            for mm in matches:
                new_input[mm] = float(new_val)
        new_df = pd.DataFrame([new_input], columns=model_feats)
        if scaler_saved is not None and numeric_cols:
            try:
                new_df[numeric_cols] = scaler_saved.transform(new_df[numeric_cols])
            except Exception:
                pass
        new_prob_lr = lr.predict_proba(new_df)[:,1][0]
        new_prob_rf = rf.predict_proba(new_df)[:,1][0]
        st.write(f"Logistic prob change: {new_prob_lr - prob_lr:+.4f} ({prob_lr:.4f} → {new_prob_lr:.4f})")
        st.write(f"RandomForest prob change: {new_prob_rf - prob_rf:+.4f} ({prob_rf:.4f} → {new_prob_rf:.4f})")

# -------------------------
# Feature importance / SHAP
# -------------------------
st.header("Feature importance & explanation")
if use_shap and HAS_SHAP:
    try:
        st.write("Computing SHAP values (TreeExplainer on RF). This may take a moment.")
        explainer = shap.TreeExplainer(rf)
        sample = X_train.sample(n=min(200, X_train.shape[0]), random_state=1)
        shap_vals = explainer.shap_values(sample)[1] if isinstance(explainer.shap_values(sample), list) else explainer.shap_values(sample)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        s = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=True).tail(30)
        fig, ax = plt.subplots(figsize=(8, max(3, len(s)*0.25)))
        s.plot(kind='barh', ax=ax)
        ax.set_title("Top mean |SHAP value|")
        st.pyplot(fig)
        # local explanation
        local_shap = explainer.shap_values(input_df)[1] if isinstance(explainer.shap_values(input_df), list) else explainer.shap_values(input_df)
        df_local = pd.DataFrame(local_shap.reshape(1,-1), columns=input_df.columns).T
        df_local.columns = ['SHAP value']
        st.subheader("Local SHAP values (current input)")
        st.dataframe(df_local.sort_values('SHAP value', ascending=False))
    except Exception as e:
        st.warning(f"SHAP failed: {e}. Showing RF importances instead.")
        plot_rf_importances(rf, X_train.columns)
else:
    st.info("SHAP not enabled or not installed. Displaying RandomForest feature importances.")
    plot_rf_importances(rf, X_train.columns)

st.success("Interactive dashboard ready — move sliders, run sensitivity, enable debug for detailed input mapping.")
