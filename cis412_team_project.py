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
