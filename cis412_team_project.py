# app.py
# Interactive Streamlit app with prediction / what-if tools
# Save and run with: streamlit run app.py

import os
import io
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# statsmodels optional (used for p-values)
try:
    import statsmodels.api as sm
    HAS_STATSM = True
except Exception:
    HAS_STATSM = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sns.set_theme()

st.set_page_config(layout="wide", page_title="CIS412 - Interactive What-If Explorer")

st.title("CIS412 — Interactive What-If Prediction Explorer")
st.markdown(
    """
Use the left sidebar to upload your `train.csv`, select features, and retrain models.
The right panels let you pick an observation (or build a custom one), change feature values,
and immediately see predicted probabilities and class from Logistic Regression and Random Forest.
You can also sweep a single feature to see how probability responds (one-at-a-time sensitivity).
"""
)

# -------------------------
# Utility helper functions
# -------------------------
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df_local = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded file loaded")
        return df_local
    # fallback paths
    for p in ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.sidebar.error("No data found. Upload a CSV in the sidebar.")
    st.stop()

def preprocess(df: pd.DataFrame, impute_strategy="median", scale_numeric=True) -> pd.DataFrame:
    df = df.copy()
    # impute numeric
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            if impute_strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    # impute categorical
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            if df[col].mode().shape[0] > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    # convert negative delays -> 0 if present
    for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x if pd.isna(x) else max(x, 0))
    # one-hot encode any remaining object/category columns (but don't touch target)
    return df

def get_target_column(df: pd.DataFrame) -> str:
    if 'satisfaction_satisfied' in df.columns:
        return 'satisfaction_satisfied'
    candidates = [c for c in df.columns if 'satisfaction' in c.lower()]
    # if single candidate, return it
    if len(candidates) == 1:
        return candidates[0]
    # else try to find binary
    for c in candidates:
        vals = set(df[c].dropna().unique())
        if vals <= {0, 1}:
            return c
    # if none, ask user
    return None

def safe_add_const(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    if 'const' not in X2.columns:
        X2.insert(0, 'const', 1.0)
    return X2

def train_models(X: pd.DataFrame, y: pd.Series, random_state=42) -> Tuple[LogisticRegression, RandomForestClassifier]:
    """Train sklearn logistic (for predictions) and RF. Return (logistic, rf)."""
    # logistic with balanced class_weight if imbalanced
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    lr.fit(X, y)
    rf.fit(X, y)
    return lr, rf

def evaluate_models(lr, rf, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    # logistic
    y_train_pred_lr = lr.predict(X_train)
    y_test_pred_lr = lr.predict(X_test)
    rows.append({
        "Model": "Logistic",
        "Split": "Train",
        "Accuracy": accuracy_score(y_train, y_train_pred_lr),
        "Precision": precision_score(y_train, y_train_pred_lr, zero_division=0),
        "Recall": recall_score(y_train, y_train_pred_lr, zero_division=0)
    })
    rows.append({
        "Model": "Logistic",
        "Split": "Test",
        "Accuracy": accuracy_score(y_test, y_test_pred_lr),
        "Precision": precision_score(y_test, y_test_pred_lr, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred_lr, zero_division=0)
    })
    # rf
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)
    rows.append({
        "Model": "RandomForest",
        "Split": "Train",
        "Accuracy": accuracy_score(y_train, y_train_pred_rf),
        "Precision": precision_score(y_train, y_train_pred_rf, zero_division=0),
        "Recall": recall_score(y_train, y_train_pred_rf, zero_division=0)
    })
    rows.append({
        "Model": "RandomForest",
        "Split": "Test",
        "Accuracy": accuracy_score(y_test, y_test_pred_rf),
        "Precision": precision_score(y_test, y_test_pred_rf, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred_rf, zero_division=0)
    })
    return pd.DataFrame(rows)

def build_input_widgets(feature_list: List[str], reference_row: pd.Series = None) -> Dict[str, Any]:
    """
    For each feature in feature_list, create a widget and return dict of values.
    If reference_row provided, prefill values from it.
    Support numeric (slider/number_input) and binary (selectbox), and small categorical (selectbox).
    """
    input_vals = {}
    st.markdown("### Build / edit custom observation")
    cols = st.columns(2)
    for i, feat in enumerate(feature_list):
        col = cols[i % 2]
        # determine dtype from global df_types
        dtype = df_types.get(feat, "numeric")
        if reference_row is not None:
            default = reference_row.get(feat, 0.0)
        else:
            default = feature_defaults.get(feat, 0.0)
        if dtype == "binary":
            val = col.selectbox(f"{feat}", options=[0, 1], index=0 if default==0 else 1)
        elif dtype == "categorical":
            opts = categorical_values.get(feat, ["NA"])
            # ensure default is in opts
            if default not in opts:
                default = opts[0]
            val = col.selectbox(f"{feat}", options=opts, index=opts.index(default))
        else:  # numeric
            # use min/max based on observed distribution
            lo, hi = feature_ranges.get(feat, (0.0, 1.0))
            # provide a number_input with reasonable step
            step = (hi - lo) / 100 if (hi - lo) > 0 else 1.0
            val = col.number_input(f"{feat}", value=float(default), min_value=float(lo), max_value=float(hi), step=float(step))
        input_vals[feat] = val
    return input_vals

def predict_from_input(lr_model, rf_model, input_df: pd.DataFrame, threshold=0.5) -> Dict[str, Any]:
    """
    Return predictions/probabilities for logistic and rf on input_df.
    """
    out = {}
    # ensure columns align (skip const for sklearn)
    X_for_sk = input_df.drop(columns=['const'], errors='ignore')
    # logistic (sklearn)
    prob_lr = lr_model.predict_proba(X_for_sk)[:, 1]
    cls_lr = (prob_lr >= threshold).astype(int)
    out['logistic_prob'] = prob_lr
    out['logistic_class'] = cls_lr
    # rf
    prob_rf = rf_model.predict_proba(X_for_sk)[:, 1]
    cls_rf = (prob_rf >= threshold).astype(int)
    out['rf_prob'] = prob_rf
    out['rf_class'] = cls_rf
    return out

def sweep_feature_and_plot(feature: str, base_input: Dict[str, Any], lr_model, rf_model, n_steps=50):
    """Vary one numeric feature across range and plot predicted probability."""
    lo, hi = feature_ranges.get(feature, (0.0, 1.0))
    xs = np.linspace(lo, hi, n_steps)
    probs_lr = []
    probs_rf = []
    base = base_input.copy()
    for x in xs:
        base[feature] = x
        row = pd.DataFrame([base])
        # ensure const and order
        row = row[model_feature_order]
        pred = predict_from_input(lr_model, rf_model, row, threshold=threshold_slider)
        probs_lr.append(pred['logistic_prob'][0])
        probs_rf.append(pred['rf_prob'][0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, probs_lr, label='Logistic Prob', marker=None)
    ax.plot(xs, probs_rf, label='RandomForest Prob', marker=None)
    ax.set_xlabel(feature)
    ax.set_ylabel('Predicted probability (class=1)')
    ax.set_title(f'What-if sweep: {feature}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -------------------------
# Sidebar UI - Data + options
# -------------------------
st.sidebar.header("Data & Preprocessing")
uploaded_file = st.sidebar.file_uploader("Upload train.csv", type=["csv"])
df_raw = load_data(uploaded_file)
st.sidebar.write(f"Data shape: {df_raw.shape}")

impute_strategy = st.sidebar.selectbox("Impute numeric with", options=["median", "mean"], index=0)
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))

# preprocess
df = preprocess(df_raw, impute_strategy=impute_strategy, scale_numeric=scale_numeric)

# identify target
target_col = get_target_column(df)
if target_col is None:
    st.sidebar.error("Cannot auto-detect satisfaction target. Please ensure a 'satisfaction' column exists (e.g., 'satisfaction_satisfied').")
    st.stop()
st.sidebar.markdown(f"**Target detected**: `{target_col}`")

# ensure target is binary int
if df[target_col].dtype != int and df[target_col].dtype != np.int64:
    # attempt to convert
    if set(df[target_col].dropna().unique()) <= {0, 1}:
        df[target_col] = df[target_col].astype(int)
    else:
        # if object like 'satisfied'/'dissatisfied', convert to binary by factorizing (warn user)
        df[target_col], _ = pd.factorize(df[target_col])
        st.sidebar.info(f"Factorized target {target_col} into binary 0/1.")

# one-hot encode remaining object/category columns (except target)
object_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c != target_col]
if object_cols:
    df = pd.get_dummies(df, columns=object_cols, drop_first=True)

# prepare metadata: feature ranges, types
all_features = [c for c in df.columns if c != target_col]
feature_ranges = {}
df_types = {}
categorical_values = {}
feature_defaults = {}
for c in all_features:
    ser = df[c]
    if set(ser.dropna().unique()) <= {0,1}:
        df_types[c] = "binary"
        feature_ranges[c] = (0,1)
        categorical_values[c] = [0,1]
        feature_defaults[c] = int(ser.median()) if ser.notna().any() else 0
    elif pd.api.types.is_numeric_dtype(ser):
        df_types[c] = "numeric"
        lo, hi = float(ser.min()), float(ser.max())
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        feature_ranges[c] = (lo, hi)
        feature_defaults[c] = float(ser.median())
    else:
        df_types[c] = "categorical"
        vals = sorted(list(ser.dropna().unique()))
        categorical_values[c] = vals
        feature_defaults[c] = vals[0]
        feature_ranges[c] = (0, 1)

# Sidebar: choose features to use in models
st.sidebar.header("Model feature selection")
default_features = all_features.copy()
chosen_features = st.sidebar.multiselect("Select features (for modeling & prediction)", options=all_features, default=default_features)

if len(chosen_features) == 0:
    st.sidebar.error("Select at least one feature.")
    st.stop()

# Option: include const automatically (sklearn doesn't need it)
use_scaler = scale_numeric  # we'll scale numeric columns in chosen_features if desired

# Prepare X,y for modeling (sklearn)
X = df[chosen_features].copy()
y = df[target_col].copy()

# Scale numeric features if selected
num_cols_in_X = [c for c in X.columns if df_types.get(c,'numeric') == 'numeric']
if use_scaler and num_cols_in_X:
    scaler = StandardScaler()
    X[num_cols_in_X] = scaler.fit_transform(X[num_cols_in_X])
    # Save scaled default values
    for c in num_cols_in_X:
        feature_defaults[c] = float(X[c].median())

# Add const for statsmodels if used for p-values later
X_const = safe_add_const(X)

# Sidebar: retrain controls
st.sidebar.header("Train / Retrain")
retrain_btn = st.sidebar.button("Train / Retrain models")
# show precomputed train/test split sizes or allow user to train now
# We'll train immediately when button pressed or when app first loads
train_now = retrain_btn or ("models_trained" not in st.session_state)

if train_now:
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None)
    # train sklearn logistic and rf
    lr_model, rf_model = train_models(X_train, y_train, random_state)
    # store in session
    st.session_state['lr_model'] = lr_model
    st.session_state['rf_model'] = rf_model
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['model_feature_order'] = X.columns.tolist()
else:
    if 'lr_model' not in st.session_state:
        st.info("Hit 'Train / Retrain models' to train models with the selected features.")
        st.stop()
    lr_model = st.session_state['lr_model']
    rf_model = st.session_state['rf_model']
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    model_feature_order = st.session_state['model_feature_order']

# Evaluate and show metrics
st.subheader("Model evaluation (train & test)")
metrics_df = evaluate_models(lr_model, rf_model, X_train, X_test, y_train, y_test)
st.dataframe(metrics_df.style.format({"Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}"}))

# show p-values if statsmodels available (optional)
if HAS_STATSM:
    st.subheader("Statsmodels Logit (p-values) [optional]")
    try:
        X_const_for_stats = safe_add_const(X)  # const included
        logit_sm = sm.Logit(y, X_const_for_stats)
        res_sm = logit_sm.fit(disp=False, maxiter=200)
        st.text(res_sm.summary().as_text())
        pvals = res_sm.pvalues.sort_values()
        st.write("Significant (p < 0.05):")
        st.dataframe(pvals[pvals < 0.05].to_frame(name='pvalue'))
    except Exception as e:
        st.warning(f"statsmodels Logit failed to fit for p-values: {e}")

# -------------------------
# Interactive prediction area
# -------------------------
st.header("Interactive Prediction / What-if")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### Choose an observation")
    row_choice_mode = st.radio("Pick a source for the base observation", options=["Use a row from dataset", "Build custom from scratch"], index=0)

    if row_choice_mode == "Use a row from dataset":
        idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df)-1, value=0, step=1)
        base_row_full = df.iloc[idx]
        st.write("Base row (original):")
        st.dataframe(pd.DataFrame(base_row_full).T)
        # create base_input limited to chosen_features
        base_input = {f: (base_row_full[f] if f in base_row_full.index else feature_defaults.get(f, 0)) for f in chosen_features}
        st.markdown("You can edit the values below and click **Predict**")
        input_vals = build_input_widgets(chosen_features, reference_row=pd.Series(base_input))
    else:
        st.markdown("Build custom observation")
        input_vals = build_input_widgets(chosen_features, reference_row=None)
        base_input = input_vals.copy()

    # threshold slider
    st.markdown("### Decision threshold")
    threshold_slider = st.slider("Prediction threshold (prob->class)", 0.01, 0.99, 0.5, 0.01)

    # Predict button
    if st.button("Predict with current input"):
        # convert input_vals into dataframe (one row), ensure column order matches model's training order
        input_df = pd.DataFrame([input_vals])
        # apply same scaling to numeric columns if we scaled earlier
        if use_scaler and num_cols_in_X:
            # we used scaler fitted on X earlier; best attempt: transform using that scaler if available in session
            try:
                # Refit a scaler on X_train numeric to transform the input consistently
                s = StandardScaler()
                s.fit(pd.concat([X_train[num_cols_in_X], X_test[num_cols_in_X]], axis=0))
                input_df[num_cols_in_X] = s.transform(input_df[num_cols_in_X])
            except Exception:
                pass

        # ensure columns order
        input_df = input_df[model_feature_order]
        # attach const for statsmodels-like representation if needed
        input_df_const = safe_add_const(input_df)
        preds = predict_from_input(lr_model, rf_model, input_df_const, threshold=threshold_slider)
        st.markdown("### Predictions")
        st.write("Logistic Regression (sklearn) — probability of class=1:", float(preds['logistic_prob'][0]))
        st.write("Logistic class (threshold {:.2f}):".format(threshold_slider), int(preds['logistic_class'][0]))
        st.write("Random Forest — probability of class=1:", float(preds['rf_prob'][0]))
        st.write("Random Forest class (threshold {:.2f}):".format(threshold_slider), int(preds['rf_class'][0]))

        # show local feature importance approximation: change each numeric feature by +-10% and measure Δprob
        st.markdown("#### Local sensitivity (±10% numeric change) — change in logistic prob")
        sens = {}
        for feat in chosen_features:
            if df_types.get(feat) == 'numeric':
                val = float(input_vals[feat])
                lo = val * 0.9
                hi = val * 1.1
                df_lo = input_df_const.copy(); df_hi = input_df_const.copy()
                df_lo[feat] = lo; df_hi[feat] = hi
                pred_lo = predict_from_input(lr_model, rf_model, df_lo, threshold=threshold_slider)['logistic_prob'][0]
                pred_hi = predict_from_input(lr_model, rf_model, df_hi, threshold=threshold_slider)['logistic_prob'][0]
                sens[feat] = pred_hi - pred_lo
        if sens:
            sens_s = pd.Series(sens).sort_values(ascending=False)
            st.bar_chart(sens_s)
        else:
            st.write("No numeric features to compute ±10% sensitivity.")

with right_col:
    st.markdown("### Prediction explorer visuals")
    st.markdown("You can sweep a numeric feature to see how predicted probability changes (one-at-a-time).")
    sweep_feature = st.selectbox("Feature to sweep (numeric only)", options=[f for f in chosen_features if df_types.get(f)=='numeric'])
    steps = st.slider("Sweep steps", 10, 200, 50)
    st.markdown("Use the 'Predict' button on the left to set the base observation; then click 'Sweep' to plot.")
    if st.button("Sweep feature"):
        # build base_input using last input_vals (we should have one)
        base_input_for_sweep = input_vals.copy()
        # ensure input order and types
        # create input_df to match model_feature_order columns
        for f in model_feature_order:
            if f not in base_input_for_sweep:
                base_input_for_sweep[f] = feature_defaults.get(f, 0)
        # call sweep plotter
        model_feature_order = X.columns.tolist()
        sweep_feature_and_plot(sweep_feature, base_input_for_sweep, lr_model, rf_model, n_steps=steps)

# -----------------------------------------
# Additional interactive: flip label & retrain simulation
# -----------------------------------------
st.header("Simulation: Flip a row's target & retrain (quick experiment)")
st.markdown("Pick a dataset row; flip its target (0→1 or 1→0), retrain on the modified dataset, and compare Test accuracy.")
if st.checkbox("Enable flip-and-retrain experiment"):
    row_idx = st.number_input("Row index to flip", min_value=0, max_value=len(df)-1, value=0, step=1)
    flip_row = df.iloc[row_idx]
    st.write("Original target for this row:", int(flip_row[target_col]))
    flip_action = st.button("Flip & retrain")
    if flip_action:
        df_sim = df.copy()
        df_sim.loc[df_sim.index == flip_row.name, target_col] = 1 - int(df_sim.loc[flip_row.name, target_col])
        X_sim = df_sim[chosen_features]
        y_sim = df_sim[target_col]
        # scale same way
        if use_scaler and num_cols_in_X:
            s = StandardScaler(); X_sim[num_cols_in_X] = s.fit_transform(X_sim[num_cols_in_X])
        Xtr, Xte, ytr, yte = train_test_split(X_sim, y_sim, test_size=test_size, random_state=random_state, stratify=y_sim if len(np.unique(y_sim))>1 else None)
        lr_sim, rf_sim = train_models(Xtr, ytr, random_state)
        acc_lr = lr_sim.score(Xte, yte)
        acc_rf = rf_sim.score(Xte, yte)
        st.write(f"After flip: Logistic Test accuracy = {acc_lr:.3f}; RandomForest Test accuracy = {acc_rf:.3f}")
        st.info("This is a simple local experiment — flipping single rows can have small effects unless the dataset is tiny.")

st.success("Interactive what-if tools ready. Adjust features / inputs and retrain to explore model behavior.")
# app.py
# -*- coding: utf-8 -*-
# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app converted from CIS412_Team_Project.ipynb
Runs preprocessing, statsmodels Logit, RandomForest, displays p-values,
confusion matrices, learning curves, and other diagnostics.
"""

import os
import io
import warnings
from typing import List

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# statsmodels may be slower to fit; we use try/except fallbacks to sklearn where appropriate
try:
    import statsmodels.api as sm
    HAS_STATSM = True
except Exception:
    HAS_STATSM = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier

# suppress warnings in UI
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CIS412 - Customer Satisfaction Explorer", layout="wide")

# ---- Helper functions ----
def show_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    st.write(f"**Shape:** {df.shape}")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("**Describe:**")
    st.dataframe(df.describe(include='all').T)

def plot_confusion_matrix(cm: np.ndarray, xticklabels: List[str], yticklabels: List[str], title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    st.pyplot(fig)

def plot_barh_series(values: pd.Series, xlabel: str = 'Value', title: str = ''):
    fig, ax = plt.subplots(figsize=(10, max(4, len(values)*0.3)))
    values.plot(kind='barh', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)

def safe_add_constant(X: pd.DataFrame, has_constant: str = 'add') -> pd.DataFrame:
    if HAS_STATSM:
        return sm.add_constant(X, has_constant=has_constant)
    else:
        # simple fallback: add 'const' column of 1s if missing
        X2 = X.copy()
        if 'const' not in X2.columns:
            X2.insert(0, 'const', 1.0)
        return X2

def try_fit_logit(Y, X):
    """
    Attempt to fit a statsmodels Logit. If statsmodels not available or fails,
    return (result_obj, error_message). result_obj will be None on failure.
    """
    if not HAS_STATSM:
        return None, "statsmodels not installed; install it to see p-values and a statsmodels summary."
    try:
        model = sm.Logit(Y, X)
        res = model.fit(disp=False, maxiter=200)
        return res, None
    except Exception as e:
        return None, str(e)


# ---- App UI ----
st.title("CIS412 - Customer Satisfaction Explorer")
st.markdown("""
This Streamlit app reproduces the original notebook workflow:
- Preprocessing (imputation, encoding, scaling)
- Logistic regression (statsmodels Logit when available)
- A focused logistic on a small feature subset
- Random Forest classifier and comparisons
- Visual diagnostics: p-values, coefficients, confusion matrices, learning curves
""")

# Data upload
st.sidebar.header("Data")
st.sidebar.write("Upload `train.csv` or the app will attempt to read `./train.csv` or `./data/train.csv`.")
uploaded_file = st.sidebar.file_uploader("Upload train.csv", type=["csv"])

# options
scale_numeric = st.sidebar.checkbox("Scale numeric features (StandardScaler)", value=True)
remove_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)
impute_strategy = st.sidebar.selectbox("Impute numeric with", ["median", "mean"], index=0)
test_size = st.sidebar.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))

# load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV read successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    fallback_paths = ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]
    df = None
    for p in fallback_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                st.info(f"Loaded data from {p}")
                break
            except Exception:
                continue
    if df is None:
        st.warning("No data found. Please upload train.csv using the uploader in the sidebar.")
        st.stop()

st.header("Raw Data Overview")
st.write("Raw Data (first 5 rows):")
st.dataframe(df.head())
show_dataframe_info(df, name="Original df")

# 4. Remove duplicates
st.subheader("4. Remove duplicates")
duplicate_count = int(df.duplicated().sum())
st.write(f"Duplicate rows: {duplicate_count}")
if remove_duplicates and duplicate_count > 0:
    df = df.drop_duplicates()
    st.write(f"Dropped duplicates. New shape: {df.shape}")

# 5. Missing values
st.subheader("5. Missing values")
missing_summary = df.isnull().sum()
if (missing_summary > 0).any():
    st.write("Missing values per column (only >0 shown):")
    st.write(missing_summary[missing_summary > 0])
else:
    st.write("No missing values detected.")

# 6. Imputation
st.subheader("6. Impute missing values")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().any():
        if impute_strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().any():
        if df[col].mode().shape[0] > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

st.write("Imputation complete. Any missing values remaining?")
st.write(int(df.isnull().sum().sum()))

# 7. Encode categorical variables
st.subheader("7. Encode categorical variables")
object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
st.write("Object/category columns to convert to dummies:", object_cols)
if len(object_cols) > 0:
    df = pd.get_dummies(df, columns=object_cols, drop_first=True)
    st.write("One-hot encoding applied.")
else:
    st.write("No object/category columns remain.")
st.write("Shape after encoding:", df.shape)

# 8. Clean delays
st.subheader("8. Clean numeric features (negative delays -> 0)")
delay_cols = [c for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes'] if c in df.columns]
for col in delay_cols:
    df[col] = df[col].apply(lambda x: x if pd.isna(x) else max(x, 0))

# 9. Scale numeric features
st.subheader("9. Scale numeric features (optional)")
features_to_scale = [c for c in ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'] if c in df.columns]
if scale_numeric and features_to_scale:
    try:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        st.write("Scaled features:", features_to_scale)
    except Exception as e:
        st.warning(f"Scaling failed: {e}")
else:
    st.write("No scaling performed or no matching features to scale.")

st.write("After cleaning – any missing values left?", int(df.isnull().sum().sum()))

# save cleanedCSV (optional)
cleaned_path = os.path.join(os.getcwd(), "train_cleaned.csv")
try:
    df.to_csv(cleaned_path, index=False)
    st.write(f"Cleaned dataset saved to: {cleaned_path}")
except Exception:
    pass

# Identify target
st.header("Identify Target & Features")
if 'satisfaction_satisfied' in df.columns:
    target_col = 'satisfaction_satisfied'
else:
    sat_cols = [c for c in df.columns if 'satisfaction' in c.lower()]
    if len(sat_cols) == 1:
        target_col = sat_cols[0]
        st.info(f"Using {target_col} as target.")
    else:
        # attempt to find a binary column candidate
        cand = None
        for c in sat_cols:
            vals = set(df[c].dropna().unique())
            if vals <= {0,1}:
                cand = c
                break
        if cand is not None:
            target_col = cand
            st.info(f"Using {target_col} as target (binary).")
        else:
            st.error("Cannot identify a satisfaction column automatically. Please ensure 'satisfaction_satisfied' exists in your data.")
            st.stop()

st.write("Target column selected:", target_col)
y = df[target_col].astype(int)

# Drop obvious non-features
drop_candidates = ['Unnamed: 0', 'id', 'satisfaction_satisfied', target_col]
drop_cols = [c for c in drop_candidates if c in df.columns]
X = df.drop(columns=drop_cols)

st.write("Features shape:", X.shape)
st.dataframe(X.head())

# Add constant and convert bools -> ints
X_const = safe_add_constant(X)
bool_cols = X_const.select_dtypes(include='bool').columns.tolist()
for c in bool_cols:
    X_const[c] = X_const[c].astype(int)

# Convert any remaining non-numeric columns to numeric where possible or drop
non_numeric = X_const.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    st.warning("Converting or dropping non-numeric columns: " + ", ".join(non_numeric))
    for c in non_numeric:
        try:
            X_const[c] = pd.to_numeric(X_const[c])
        except Exception:
            X_const.drop(columns=[c], inplace=True)

st.subheader("Features with constant (first rows)")
st.dataframe(X_const.head())

# Fit logistic regression using statsmodels if available
st.header("Fit Logistic Regression (statsmodels.Logit)")
if not HAS_STATSM:
    st.error("statsmodels not installed — statsmodels Logit will not run. Install statsmodels to see p-values and summary.")
else:
    logit_res, err = try_fit_logit(y, X_const)
    if logit_res is None:
        st.error(f"Logit fit failed: {err}")
        # try removing zero-variance columns and retry
        nzv = [c for c in X_const.columns if X_const[c].nunique() <= 1]
        if nzv:
            st.info(f"Dropping zero-variance columns and retrying: {nzv}")
            X_const = X_const.drop(columns=nzv)
            logit_res, err = try_fit_logit(y, X_const)
        if logit_res is None:
            st.error(f"Retry failed: {err}")
    if logit_res is not None:
        st.subheader("Model Summary (statsmodels)")
        st.text(logit_res.summary().as_text())

        pvalues = logit_res.pvalues
        significant = pvalues[pvalues < 0.05].sort_values()
        nonsignificant = pvalues[pvalues >= 0.05].sort_values(ascending=False)

        st.write("Significant variables (p < 0.05):")
        st.dataframe(significant.to_frame(name='p-value'))

        st.write("Non-significant variables (p >= 0.05):")
        st.dataframe(nonsignificant.to_frame(name='p-value'))

        if not significant.empty:
            st.subheader("Significant Variables - p-values (log scale)")
            sorted_sig = significant.sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_sig)*0.25)))
            ax.barh(sorted_sig.index, sorted_sig.values)
            ax.set_xscale('log')
            ax.set_xlabel('P-value (log scale)')
            ax.set_title('Significant Variables (p-value < 0.05)')
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(fig)

# Focused small-model analysis (df_copy from notebook)
st.header("Focused small-model analysis (selected features)")

selected_columns = [
    'Checkin service', 'Food and drink', 'Inflight service',
    'Cleanliness', 'On-board service', 'Baggage handling'
]
selected_columns = [c for c in selected_columns if c in df.columns]
if target_col not in selected_columns:
    selected_columns.append(target_col)

if len(selected_columns) < 2:
    st.error("Not enough selected columns found in dataset to run the focused analysis. Add columns or upload appropriate data.")
else:
    df_copy = df[selected_columns].copy()
    df_copy[target_col] = df_copy[target_col].astype(int)

    st.write("df_copy first rows:")
    st.dataframe(df_copy.head())
    buffer = io.StringIO()
    df_copy.info(buf=buffer)
    st.text(buffer.getvalue())

    X_small = df_copy.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_small = df_copy[target_col]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=test_size,
                                                        random_state=random_state, stratify=y_small if len(np.unique(y_small))>1 else None)

    X_train_const = safe_add_constant(X_train)
    X_test_const = safe_add_constant(X_test)

    st.write("Train/Test shapes:", X_train.shape, X_test.shape)

    # Fit logistic on df_copy
    logit_results = None
    if HAS_STATSM:
        logit_results, err = try_fit_logit(y_train, X_train_const)
        if logit_results is None:
            st.warning(f"statsmodels Logit failed for df_copy: {err}")
    # fallback to sklearn logistic if statsmodels not available or failed
    if logit_results is None:
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            # fabricate a small wrapper with similar attributes for predictions
            class SklearnLogitWrapper:
                def __init__(self, model):
                    self.model = model
                    self.params = pd.Series(np.concatenate(([0], model.coef_.ravel())), index=['const'] + list(X_train.columns))
                    # pvalues not available
                def predict(self, X):
                    return self.model.predict_proba(X.drop(columns=['const'], errors='ignore'))[:, 1]
            logit_results = None
            lr_wrapper = SklearnLogitWrapper(lr)
        except Exception as e:
            st.error(f"Failed to fit fallback sklearn LogisticRegression: {e}")
            st.stop()
    else:
        lr_wrapper = None

    # Evaluate logistic predictions
    if logit_results is not None:
        y_pred_prob = logit_results.predict(X_test_const)
        y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        y_pred_prob = lr_wrapper.predict(X_test_const)
        y_pred = (y_pred_prob >= 0.5).astype(int)

    st.write("Logistic (focused model) - Test accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix (logistic focused):")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm,
                          xticklabels=['Pred 0', 'Pred 1'],
                          yticklabels=['True 0', 'True 1'],
                          title='Confusion Matrix (Logistic - focused)')

    st.write("Classification Report (logistic focused):")
    st.text(classification_report(y_test, y_pred, digits=3))

    try:
        auc_val = roc_auc_score(y_test, y_pred_prob)
        st.write(f"ROC-AUC (logistic focused): {auc_val:.3f}")
    except Exception:
        st.write("ROC-AUC unavailable (possible constant predictions).")

    # Random Forest on small dataset
    st.subheader("Random Forest (same subset)")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    y_test_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    st.write(f"RF - Train acc: {accuracy_score(y_train, y_train_pred_rf):.3f}")
    st.write(f"RF - Test acc: {accuracy_score(y_test, y_test_pred_rf):.3f}")
    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    plot_confusion_matrix(cm_rf, xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], title="RF Confusion Matrix")
    st.write("RF classification report (test):")
    st.text(classification_report(y_test, y_test_pred_rf, digits=3))
    try:
        st.write(f"RF ROC-AUC (test): {roc_auc_score(y_test, y_test_prob_rf):.3f}")
    except Exception:
        pass

    # Feature importances
    st.subheader("Random Forest Feature Importances")
    feat_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    st.dataframe(feat_imp.to_frame(name='importance'))
    plot_barh_series(feat_imp, xlabel='Importance', title='RF Feature Importances')

    # Significant coefficients from statsmodels if available
    st.subheader("Significant Coefficients (logistic focused)")
    if logit_results is not None:
        coeffs = logit_results.params
        pvals = logit_results.pvalues
        sig = coeffs[pvals < 0.05].drop('const', errors='ignore')
        if not sig.empty:
            coeff_df = pd.DataFrame({'Coefficient': sig, 'P-value': pvals[pvals < 0.05].drop('const', errors='ignore')})
            coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
            sorted_coeffs = coeff_df.sort_values(by='Abs_Coefficient', ascending=True)
            st.dataframe(sorted_coeffs[['Coefficient', 'P-value']])
            # plot
            fig, ax = plt.subplots(figsize=(10, max(3, len(sorted_coeffs)*0.3)))
            colors = ['#5DA5FF' if v>0 else '#FF6F6F' for v in sorted_coeffs['Coefficient']]
            ax.barh(sorted_coeffs.index, sorted_coeffs['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Significant Feature Coefficients (excluding const)')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.write("No significant coefficients (p < 0.05) found for this focused model.")
    else:
        st.info("Detailed coefficient/p-value reporting requires statsmodels; using sklearn fallback.")

    # Learning curves
    st.header("Learning Curves (focused subset)")
    train_sizes = np.linspace(0.1, 1.0, 10)
    num_train = (train_sizes * len(X_train)).astype(int)
    log_train_acc = []
    log_test_acc = []
    rf_train_acc = []
    rf_test_acc = []

    for n in num_train:
        if n < 2:
            log_train_acc.append(np.nan); log_test_acc.append(np.nan)
            rf_train_acc.append(np.nan); rf_test_acc.append(np.nan)
            continue
        Xt = X_train.iloc[:n]; yt = y_train.iloc[:n]
        # logistic subset (try statsmodels first)
        if HAS_STATSM:
            try:
                res_sub, _ = try_fit_logit(yt, safe_add_constant(Xt))
                preds_train = (res_sub.predict(safe_add_constant(Xt)) >= 0.5).astype(int)
                preds_test = (res_sub.predict(safe_add_constant(X_test)) >= 0.5).astype(int)
                log_train_acc.append(accuracy_score(yt, preds_train))
                log_test_acc.append(accuracy_score(y_test, preds_test))
            except Exception:
                log_train_acc.append(np.nan); log_test_acc.append(np.nan)
        else:
            # sklearn fallback
            try:
                from sklearn.linear_model import LogisticRegression
                lr_sub = LogisticRegression(max_iter=1000)
                lr_sub.fit(Xt, yt)
                log_train_acc.append(lr_sub.score(Xt, yt))
                log_test_acc.append(lr_sub.score(X_test, y_test))
            except Exception:
                log_train_acc.append(np.nan); log_test_acc.append(np.nan)
        # RF subset
        try:
            rf_sub = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            rf_sub.fit(Xt, yt)
            rf_train_acc.append(accuracy_score(yt, rf_sub.predict(Xt)))
            rf_test_acc.append(accuracy_score(y_test, rf_sub.predict(X_test)))
        except Exception:
            rf_train_acc.append(np.nan); rf_test_acc.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_train, log_train_acc, marker='o', label='Logistic - Train')
    ax.plot(num_train, log_test_acc, marker='o', label='Logistic - Test')
    ax.plot(num_train, rf_train_acc, marker='o', label='RF - Train')
    ax.plot(num_train, rf_test_acc, marker='o', label='RF - Test')
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curves (focused subset)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Comparison table
    st.header("Model comparison (focused subset)")
    # ensure we have predictions
    try:
        # logistic predictions
        if logit_results is not None:
            y_train_prob_log = logit_results.predict(X_train_const)
            y_train_pred_log = (y_train_prob_log >= 0.5).astype(int)
            y_test_prob_log = logit_results.predict(X_test_const)
            y_test_pred_log = (y_test_prob_log >= 0.5).astype(int)
        else:
            y_train_pred_log = lr.predict(X_train)
            y_test_pred_log = lr.predict(X_test)
        results = []
        results.append({'Model': 'Logistic Regression', 'Split': 'Train',
                        'Accuracy': accuracy_score(y_train, y_train_pred_log),
                        'Precision': precision_score(y_train, y_train_pred_log, zero_division=0),
                        'Recall': recall_score(y_train, y_train_pred_log, zero_division=0)})
        results.append({'Model': 'Logistic Regression', 'Split': 'Test',
                        'Accuracy': accuracy_score(y_test, y_test_pred_log),
                        'Precision': precision_score(y_test, y_test_pred_log, zero_division=0),
                        'Recall': recall_score(y_test, y_test_pred_log, zero_division=0)})
        results.append({'Model': 'Random Forest', 'Split': 'Train',
                        'Accuracy': accuracy_score(y_train, y_train_pred_rf),
                        'Precision': precision_score(y_train, y_train_pred_rf, zero_division=0),
                        'Recall': recall_score(y_train, y_train_pred_rf, zero_division=0)})
        results.append({'Model': 'Random Forest', 'Split': 'Test',
                        'Accuracy': accuracy_score(y_test, y_test_pred_rf),
                        'Precision': precision_score(y_test, y_test_pred_rf, zero_division=0),
                        'Recall': recall_score(y_test, y_test_pred_rf, zero_division=0)})
        results_df_full = pd.DataFrame(results)
        st.dataframe(results_df_full.style.format({"Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}"}))
    except Exception as e:
        st.warning(f"Could not build comparison table: {e}")

st.success("Analysis complete. Adjust options in the sidebar and re-run as needed.")
