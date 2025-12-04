# --- BEGIN REPLACEMENT TOP SECTION ---
import os
import io
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sns.set_theme()
st.set_page_config(layout="wide", page_title="CIS412 — Interactive What-If Dashboard")

# Helper: load CSV either from uploader or from repo fallback paths
def load_csv(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[pd.DataFrame]:
    """Return DataFrame if uploaded_file or a fallback file exists, else None."""
    # If user uploaded, try that first
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Uploaded CSV read successfully.")
            return df
        except Exception as e:
            st.sidebar.error(f"Uploaded file read failed: {e}")
            return None

    # Fallback: check common repo paths
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

    # no file found
    st.sidebar.warning("No CSV found. Please upload train.csv via the uploader above or add train.csv to the repo.")
    return None

# Create uploader (single place). Give an explicit key to avoid duplicate-element errors.
uploaded_file = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"], key="uploader_main")

# Now load the dataframe using the helper
df_raw = load_csv(uploaded_file)

# If no df loaded, stop the app but show instructions
if df_raw is None:
    st.title("CIS412 — Interactive What-If Dashboard")
    st.write("No data loaded. Please upload `train.csv` using the uploader in the sidebar or add a `train.csv` file to the repo.")
    st.stop()

# --- END REPLACEMENT TOP SECTION ---


# One-hot encode categorical columns except target
obj_cols = [c for c in df_clean.select_dtypes(include=['object','category']).columns if c != target_col]
if obj_cols:
    df_processed = pd.get_dummies(df_clean, columns=obj_cols, drop_first=True)
else:
    df_processed = df_clean.copy()

# crea
# app_debug.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")
st.title("APP DEBUG: UI + Interaction Checker")

st.sidebar.header("Debug controls")
uploaded = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"])
show_session = st.sidebar.checkbox("Show session_state", value=True)
run_quick = st.sidebar.button("Run quick interaction test")

# Try to read fallback files if no upload
def load_csv(uploaded):
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV read")
            return df
        except Exception as e:
            st.sidebar.error(f"Upload read failed: {e}")
            return None
    # fallback repo paths
    for p in ["./train.csv","./data/train.csv","/content/sample_data/train.csv"]:
        try:
            df = pd.read_csv(p)
            st.sidebar.info(f"Loaded CSV from {p}")
            return df
        except Exception:
            pass
    st.sidebar.warning("No CSV found in repo or uploaded. Upload one or add train.csv to repo.")
    return None

# Load data
df = load_csv(uploaded)

# Always show the sidebar instruction so user sees it
st.sidebar.write("Make sure to expand the sidebar (top-left hamburger).")

# Show dataset info
if df is not None:
    st.subheader("Data preview (first 5 rows)")
    st.dataframe(df.head())
    st.subheader("Columns")
    st.write(list(df.columns))
else:
    st.info("No dataframe loaded yet. Upload or add train.csv to repo.")

# Show session state
if show_session:
    st.subheader("Session state")
    st.write(dict(st.session_state))

# A simple slider + live "model"
st.subheader("Interaction test")
s = st.slider("Test slider (move me)", 0, 10, 5)
st.write("Slider value:", s)

# A small dummy logistic regression trained on toy data so predictions change live
if run_quick:
    st.write("Running quick model training on toy data...")
    X = np.arange(0,11).reshape(-1,1)
    y = (X.ravel() > 4).astype(int)
    model = LogisticRegression().fit(X, y)
    pred_prob = model.predict_proba(np.array([[s]]))[:,1][0]
    pred_class = int(pred_prob >= 0.5)
    st.write(f"Pred prob (toy model): {pred_prob:.4f}, pred class: {pred_class}")
else:
    st.write("Click 'Run quick interaction test' in the sidebar to train a tiny model and predict with current slider.")

# Try/except to show any top-level exceptions
try:
    1 + 1  # nop
except Exception as e:
    st.error(f"Top-level exception: {e}")
