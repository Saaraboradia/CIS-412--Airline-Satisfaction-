# app.py
"""
Interactive Streamlit app for CIS412 Airline Satisfaction
- Robust prediction alignment: sliders -> model input vector
- Saves models, scaler, and feature order to st.session_state
- Optional SHAP explainer (if shap is installed)
"""

import os
import io
import warnings
from typing import Dict, Any, List

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
st.set_page_config(layout="wide", page_title="CIS412 — Interactive What-If App")

# optional shap
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# Helper functions
# -------------------------
def load_csv(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for p in ["./train.csv", "./data/train.csv", "/content/sample_data/train.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("No dataset found. Upload train.csv in the sidebar or add train.csv to the repo.")
    st.stop()

def detect_target_col(df: pd.DataFrame) -> str:
    if 'satisfaction_satisfied' in df.columns:
        return 'satisfaction_satisfied'
    cands = [c for c in df.columns if 'satisfaction' in c.lower()]
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        vals = set(df[c].dropna().unique())
        if vals <= {0,1}:
            return c
    return None

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # numeric imputation median
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

def draw_prob_bar(prob_lr, prob_rf):
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.barh(['Logistic','RandomForest'], [prob_lr, prob_rf], color=['#2b8cbe','#fdae6b'])
    ax.set_xlim(0,1)
    for i,p in enumerate([prob_lr, prob_rf]):
        ax.text(p + 0.02, i, f"{p:.3f}", va='center')
    ax.set_xlabel("Predicted probability (class=1)")
    st.pyplot(fig)

def plot_feature_importances_from_rf(rf, feature_names):
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(imp)*0.12)))
    imp.plot(kind='barh', ax=ax)
    ax.set_title("RandomForest feature importances (fallback if SHAP missing)")
    st.pyplot(fig)

# -------------------------
# UI - sidebar
# -------------------------
st.sidebar.header("Dataset & app options")
uploaded_file = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"])
df_raw = load_csv(uploaded_file)
st.sidebar.write("Dataset shape:", df_raw.shape)

scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
show_prediction_debug = st.sidebar.checkbox("Show prediction debug info", value=False)
use_shap = st.sidebar.checkbox("Compute SHAP explainer (may be slow)", value=False)

# -------------------------
# Preprocess & feature creation
# -------------------------
st.title("CIS412 — Interactive What-If Dashboard (corrected)")
st.markdown("This app builds models from a CSV, exposes sliders for service features, and predicts live. "
            "If slider changes do not affect predictions, enable *Show prediction debug info* in the sidebar.")

df_clean = basic_preprocess(df_raw)

target_col = detect_target_col(df_clean)
if target_col is None:
    st.error("Could not detect a satisfaction target column (e.g., 'satisfaction_satisfied'). Please upload data with such a column.")
    st.stop()
st.write("Detected target:", target_col)

# One-hot encode categorical columns except target
obj_cols = [c for c in df_clean.select_dtypes(include=['object','category']).columns if c != target_col]
if obj_cols:
    df_processed = pd.get_dummies(df_clean, columns=obj_cols, drop_first=True)
else:
    df_processed = df_clean.copy()

# crea
