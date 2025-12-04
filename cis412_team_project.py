# app_with_shap.py
# Interactive Streamlit app with sliders, live predictions, sensitivity analysis and SHAP explainer.
# Requires: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, shap (optional but recommended)
import os
import io
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
sns.set_theme()

# Try import shap; handle gracefully if not installed
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

st.set_page_config(layout="wide", page_title="CIS412 — What-If Dashboard (SHAP)")

# -----------------------------
# Helper utilities
# -----------------------------
def load_data(uploaded):
    if uploaded:
        return pd.read_csv(uploaded)
    for p in ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("No dataset found. Upload a CSV via the sidebar.")
    st.stop()

def detect_target(df):
    if 'satisfaction_satisfied' in df.columns:
        return 'satisfaction_satisfied'
    cands = [c for c in df.columns if 'satisfaction' in c.lower()]
    if len(cands) == 1:
        return cands[0]
    # try binary column
    for c in cands:
        vals = set(df[c].dropna().unique())
        if vals <= {0,1}:
            return c
    return None

def preprocess_basic(df):
    df = df.copy()
    # impute numeric by median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    # impute categorical by mode
    for c in df.select_dtypes(include=['object','category']).columns:
        if df[c].isnull().any() and df[c].mode().shape[0]:
            df[c].fillna(df[c].mode()[0], inplace=True)
    # fix negative delays if present
    for d in ['Departure Delay in Minutes','Arrival Delay in Minutes']:
        if d in df.columns:
            df[d] = df[d].apply(lambda x: x if pd.isna(x) else max(x,0))
    return df

def safe_scale(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return scaler

def draw_prob_gauges(prob_lr, prob_rf):
    # simple matplotlib bar gauges for live update
    fig, ax = plt.subplots(figsize=(6,2.4))
    models = ['Logistic', 'RandomForest']
    probs = [float(prob_lr), float(prob_rf)]
    ax.barh(models, probs, color=['#2b8cbe','#fdae6b'])
    ax.set_xlim(0,1)
    for i,p in enumerate(probs):
        ax.text(p+0.02, i, f"{p:.3f}", va='center')
    ax.set_xlabel("Probability of satisfied (=1)")
    ax.set_title("Predicted probability (live)")
    st.pyplot(fig)

def shap_feature_bar(shap_vals, feature_names, top_n=20):
    # shap_vals: matrix (n_examples, n_features)
    # compute mean(|shap|)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)
    s = s.tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, len(s)*0.25)))
    s.plot(kind='barh', ax=ax)
    ax.set_title("Mean |SHAP value| (approx feature importance)")
    st.pyplot(fig)

# -----------------------------
# UI - Sidebar controls
# -----------------------------
st.sidebar.header("Dataset & options")
uploaded = st.sidebar.file_uploader("Upload train.csv (optional)", type=['csv'])
df_raw = load_data(uploaded)

st.sidebar.markdown(f"Dataset shape: {df_raw.shape}")

# preprocessing options
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
test_size = st.sidebar.slider("Test proportion", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))

df = preprocess_basic(df_raw)

target_col = detect_target(df)
if target_col is None:
    st.sidebar.error("Could not auto-detect target column named like 'satisfaction...'. Please ensure it exists.")
    st.stop()
st.sidebar.success(f"Detected target: `{target_col}`")

# ensure binary int target
if df[target_col].dtype != int and df[target_col].dtype != np.int64:
    if set(df[target_col].dropna().unique()) <= {0,1}:
        df[target_col] = df[target_col].astype(int)
    else:
        df[target_col], _ = pd.factorize(df[target_col])
        st.sidebar.info(f"Factorized target `{target_col}` to binary.")

# one-hot encode categorical except target
obj_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c != target_col]
if obj_cols:
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

# -----------------------------
# Auto-detect service rating columns (1-5 scale)
# -----------------------------
# Heuristic: columns with names containing words typical for service ratings and numeric values 1-5
rating_keywords = ['inflight','checkin','food','clean','on-board','baggage','seat','wifi','boarding','comfort','entertainment','service','leg room','ease','gate']
service_cols = []
for c in df.columns:
    lowc = c.lower()
    if any(k in lowc for k in rating_keywords):
        # numeric and mostly in 1..5
        if pd.api.types.is_numeric_dtype(df[c]):
            vals = df[c].dropna().unique()
            # check if distribution fits ratings
            if len(vals) > 1 and (df[c].between(1,5).mean() > 0.6 or df[c].median() <= 5):
                service_cols.append(c)
# dedupe
service_cols = list(dict.fromkeys(service_cols))

st.sidebar.header("Model Feature Selection")
all_features = [c for c in df.columns if c != target_col]
# default: use all features but prefer service columns listed first
default_features = service_cols + [c for c in all_features if c not in service_cols]
chosen_features = st.sidebar.multiselect("Features to include in models", options=all_features, default=default_features[:min(20,len(default_features))])

if not chosen_features:
    st.error("Select at least one feature to train models.")
    st.stop()

# -----------------------------
# Train models
# -----------------------------
X = df[chosen_features].copy()
y = df[target_col].copy()

numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
if scale_numeric and numeric_cols:
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None)

# Train sklearn models (we'll use sklearn logistic for fast live predictions)
lr = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

st.header("Models trained")
col1, col2, col3 = st.columns(3)
col1.metric("Train size", f"{X_train.shape[0]}")
col2.metric("Test size", f"{X_test.shape[0]}")
col3.metric("Features used", f"{len(chosen_features)}")

# show basic evaluations
st.subheader("Performance (quick)")
def quick_metrics(model, Xtr, Xte, ytr, yte):
    tr_acc = model.score(Xtr, ytr)
    te_acc = model.score(Xte, yte)
    try:
        te_prob = model.predict_proba(Xte)[:,1]
        te_auc = roc_auc_score(yte, te_prob)
    except Exception:
        te_auc = np.nan
    return tr_acc, te_acc, te_auc

lr_tr, lr_te, lr_auc = quick_metrics(lr, X_train, X_test, y_train, y_test)
rf_tr, rf_te, rf_auc = quick_metrics(rf, X_train, X_test, y_train, y_test)

st.write(pd.DataFrame({
    'Model':['Logistic', 'RandomForest'],
    'Train Accuracy':[lr_tr, rf_tr],
    'Test Accuracy':[lr_te, rf_te],
    'Test ROC-AUC':[lr_auc, rf_auc]
}).set_index('Model'))

# -----------------------------
# Live sliders for service ratings (main UI)
# -----------------------------
st.header("Interactive sliders (service ratings)")
st.markdown("Adjust the sliders for service ratings (detected from the dataset). Models will predict live.")

# If we couldn't detect service columns, present a fallback numeric set (top numeric features)
if not service_cols:
    st.info("No obvious service-rating columns automatically detected. Use the feature selector in the sidebar to include rating-like columns.")
    # choose top numeric features to show sliders for
    numeric_choice = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    sliders_show = numeric_choice[:6]
else:
    sliders_show = service_cols[:8]  # show up to 8 sliders

# build sliders
st.sidebar.header("Sliders preset")
with st.sidebar.expander("Service slider presets"):
    slider_presets = {
        "All 3s (baseline)": {c:3 for c in sliders_show},
        "Good (4-5)": {c:5 if 'wifi' in c.lower() or 'comfort' in c.lower() else 4 for c in sliders_show},
        "Poor (1-2)": {c:2 for c in sliders_show}
    }
    preset_choice = st.selectbox("Apply slider preset", options=["(none)"] + list(slider_presets.keys()), index=0)
    if preset_choice != "(none)":
        preset_vals = slider_presets[preset_choice]
    else:
        preset_vals = {}

st.write("Adjust sliders (values usually 1-5). If your dataset used a different scale, adjust accordingly.")
cols = st.columns(2)
slider_inputs = {}
for i, feat in enumerate(sliders_show):
    col = cols[i % 2]
    # determine min/max from observed data if possible
    obs = df[feat].dropna()
    lo = int(max(0, min(1, int(np.floor(obs.min()))))) if not obs.empty else 1
    hi = int(max(5, int(np.ceil(obs.max())))) if not obs.empty else 5
    default = preset_vals.get(feat, int(round(obs.median())) if not obs.empty else 3)
    slider_inputs[feat] = col.slider(feat, min_value=lo, max_value=hi, value=default, step=1)

# For other chosen numeric features, show expandable editor
with st.expander("Edit other numeric features (optional)"):
    other_numeric = [c for c in chosen_features if c not in sliders_show and pd.api.types.is_numeric_dtype(X[c])]
    other_inputs = {}
    if other_numeric:
        cols2 = st.columns(2)
        for i, feat in enumerate(other_numeric):
            c = cols2[i % 2]
            ser = df[feat].dropna()
            lo, hi = (float(ser.min()), float(ser.max())) if not ser.empty else (0.0, 1.0)
            default = float(ser.median()) if not ser.empty else 0.0
            other_inputs[feat] = c.number_input(feat, value=default, min_value=lo, max_value=hi)
    else:
        st.write("No other numeric features detected.")

# Assemble one input vector (base) from sliders + defaults
input_vec = {}
for feat in chosen_features:
    if feat in slider_inputs:
        input_vec[feat] = slider_inputs[feat]
    elif feat in other_inputs:
        input_vec[feat] = other_inputs[feat]
    else:
        # default to median of column
        if pd.api.types.is_numeric_dtype(df[feat]):
            input_vec[feat] = float(df[feat].median())
        else:
            # for binary / encoded features, use mode or 0
            if set(df[feat].dropna().unique()) <= {0,1}:
                input_vec[feat] = int(df[feat].mode().iloc[0] if not df[feat].mode().empty else 0)
            else:
                # pick first unique
                input_vec[feat] = float(df[feat].dropna().unique()[0]) if df[feat].dropna().any() else 0.0

# scale numeric inputs if we scaled data earlier
if scale_numeric and numeric_cols:
    try:
        # use scaler fitted on training data (we saved X earlier)
        s = StandardScaler()
        s.fit(np.vstack([X_train[numeric_cols], X_test[numeric_cols]]))
        numeric_arr = np.array([input_vec[nc] for nc in numeric_cols], dtype=float).reshape(1,-1)
        scaled = s.transform(numeric_arr).reshape(-1)
        for i, nc in enumerate(numeric_cols):
            input_vec[nc] = float(scaled[i])
    except Exception:
        pass

# Convert input to DataFrame and align to model
input_df = pd.DataFrame([input_vec], columns=chosen_features)
# predictions
prob_lr = lr.predict_proba(input_df)[:,1][0]
prob_rf = rf.predict_proba(input_df)[:,1][0]

st.subheader("Live-updated prediction")
draw_prob_gauges(prob_lr, prob_rf)

# -----------------------------
# Sensitivity analysis quick actions
# -----------------------------
st.header("Sensitivity / What-if analysis")
st.markdown("Quick experiments: choose a service and a target improvement (e.g. inflight wifi 3 -> 5) and see probability deltas.")

sensitive_feature = st.selectbox("Feature to tweak", options=sliders_show)
current_val = input_vec[sensitive_feature]
if pd.api.types.is_numeric_dtype(df[sensitive_feature]):
    tweak_to = st.slider(f"Tweak {sensitive_feature} to", min_value=int(df[sensitive_feature].min()), max_value=int(df[sensitive_feature].max()), value=int(current_val), step=1)
else:
    tweak_to = st.number_input(f"Tweak {sensitive_feature} to value", value=current_val)

if st.button("Run sensitivity: show delta"):
    # clone input_df and set the feature to tweak_to (and rescale if needed)
    new_input = input_df.copy()
    new_input[sensitive_feature] = tweak_to
    # if scaling applied earlier, try to scale just that feature appropriately
    try:
        s = StandardScaler(); s.fit(np.vstack([X_train[numeric_cols], X_test[numeric_cols]]))
        if sensitive_feature in numeric_cols:
            # scale single column
            idx = numeric_cols.index(sensitive_feature)
            # transform only that column
            new_input[sensitive_feature] = s.transform(np.array([input_vec[numeric_cols]]))[0][idx]
    except Exception:
        pass
    new_prob_lr = lr.predict_proba(new_input)[:,1][0]
    new_prob_rf = rf.predict_proba(new_input)[:,1][0]
    st.write(f"Logistic prob change: {new_prob_lr - prob_lr:+.4f} (from {prob_lr:.4f} -> {new_prob_lr:.4f})")
    st.write(f"RandomForest prob change: {new_prob_rf - prob_rf:+.4f} (from {prob_rf:.4f} -> {new_prob_rf:.4f})")

    # show small bar
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.barh(['Logistic','RandomForest'], [prob_lr, new_prob_lr], color=['#2b8cbe','#fdae6b'])
    ax.set_xlim(0,1)
    ax.set_title(f"{sensitive_feature}: before vs after")
    ax.legend().remove()
    st.pyplot(fig)

# -----------------------------
# SHAP explainer (feature importance)
# -----------------------------
st.header("Feature Importance Explainer (SHAP)")

if HAS_SHAP:
    try:
        with st.spinner("Computing SHAP values (may take a few seconds)..."):
            # Use TreeExplainer for RandomForest (fast)
            explainer = shap.TreeExplainer(rf)
            # compute on small sample from training set to be quick
            sample = X_train.sample(n=min(200, X_train.shape[0]), random_state=1)
            shap_vals = explainer.shap_values(sample)[1] if isinstance(explainer.shap_values(sample), list) else explainer.shap_values(sample)
            # shap_vals shape: (n_samples, n_features)
            shap_feature_bar(shap_vals, sample.columns.tolist(), top_n=min(30, sample.shape[1]))
            st.markdown("**Top features by mean |SHAP| shown above**")
            # per-observation explanation for our input
            st.subheader("Local explanation for current slider input (approx)")
            # compute shap for input_df
            try:
                local_shap = explainer.shap_values(input_df)[1] if isinstance(explainer.shap_values(input_df), list) else explainer.shap_values(input_df)
                # present as DataFrame
                df_local = pd.DataFrame(local_shap.reshape(1,-1), columns=input_df.columns).T
                df_local.columns = ['SHAP value']
                st.dataframe(df_local.sort_values('SHAP value', ascending=False))
            except Exception as e:
                st.warning(f"Local SHAP failed: {e}")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
else:
    st.info("SHAP package not installed. To enable SHAP explainers, install `shap` (pip install shap).")
    # provide fallback: show RF feature_importances_
    feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(feat_imp)*0.12)))
    feat_imp.plot(kind='barh', ax=ax)
    ax.set_title("RandomForest feature importances (fallback)")
    st.pyplot(fig)

st.success("Interactive dashboard ready — tweak sliders, run sensitivity experiments, and inspect feature importance!")
