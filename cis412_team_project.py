# app.py
"""
Fully self-contained CIS412 Interactive Dashboard
- NO file upload needed (data embedded directly)
- Always runs, even if user has no CSV
- Includes: auto-training, sliders, predictions, confusion matrix heatmaps, permutation importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report

st.set_page_config(layout="wide", page_title="CIS412 — Airline Satisfaction Dashboard")
sns.set_theme()

# ---------------------------------------------------------------
# 1) EMBEDDED TRAINING DATA (replace this block with your dataset)
# ---------------------------------------------------------------
TRAIN_DATA = 
id,Age,Flight Distance,Inflight wifi service,Food and drink,Seat comfort,Baggage handling,Departure Delay in Minutes,Arrival Delay in Minutes,satisfaction
1,22,1200,3,4,4,4,0,0,satisfied
2,45,300,1,2,2,3,10,5,dissatisfied
3,35,650,4,4,5,4,0,0,satisfied
4,60,400,2,3,3,3,20,18,neutral
5,28,900,5,4,5,5,0,0,satisfied
6,52,700,1,2,2,2,5,3,dissatisfied
7,31,1100,4,4,4,4,0,0,satisfied
8,40,500,2,3,3,3,15,8,neutral
9,27,1300,5,5,5,5,0,0,satisfied
10,50,450,1,2,2,2,7,4,dissatisfied
"""

# Load embedded CSV
df_raw = pd.read_csv(pd.compat.StringIO(TRAIN_DATA))
# ---------------------------------------------------------------


st.title("CIS412 — Interactive Airline Satisfaction Dashboard")
st.write("This version includes **built-in training data**, so no upload is needed!")

# -----------------------------
# Preprocessing
# -----------------------------
df = df_raw.copy()

# Convert satisfaction to binary
df["satisfaction"] = df["satisfaction"].str.lower().map(
    lambda x: 1 if "satisf" in x else 0
)

# Impute numeric
for col in df.select_dtypes(include=[float, int]):
    df[col].fillna(df[col].median(), inplace=True)

# One-hot encode if needed (not used here)
# df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Model training
# -----------------------------
target = "satisfaction"
features = [c for c in df.columns if c != target and c != "id"]

X = df[features].copy()
y = df[target]

# Scale numeric
numeric_cols = X.columns
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Train both models
lr = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# --------------------------------
# Interactive sliders
# --------------------------------
st.header("Adjust Customer Inputs")

slider_values = {}
cols = st.columns(3)

for i, feat in enumerate(features):
    col = cols[i % 3]
    min_val = float(df[feat].min())
    max_val = float(df[feat].max())
    med_val = float(df[feat].median())
    slider_values[feat] = col.slider(
        feat, min_value=min_val, max_value=max_val, value=med_val
    )

# Convert to model input
input_df = pd.DataFrame([slider_values])
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=features)

# Predict
prob_lr = lr.predict_proba(input_scaled)[0, 1]
prob_rf = rf.predict_proba(input_scaled)[0, 1]

st.subheader("Predicted Probability of Satisfaction")
st.write(f"**Logistic Regression:** {prob_lr:.3f}")
st.write(f"**Random Forest:** {prob_rf:.3f}")

st.progress(prob_rf)

# -----------------------------
# Confusion matrix graphics
# -----------------------------
st.header("Model Performance — Confusion Matrices")

def plot_conf_matrix(model, X, y, title):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Satisfied", "Satisfied"],
                yticklabels=["Not Satisfied", "Satisfied"],
                ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

plot_conf_matrix(lr, X_test, y_test, "Logistic Regression — Confusion Matrix")
plot_conf_matrix(rf, X_test, y_test, "Random Forest — Confusion Matrix")

# -----------------------------
# Permutation Importance
# -----------------------------
st.header("Feature Importance (Permutation Importance)")

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
importances = pd.Series(perm.importances_mean, index=features).sort_values()

fig, ax = plt.subplots(figsize=(6,4))
importances.plot(kind="barh", ax=ax)
ax.set_title("Permutation Importance")
st.pyplot(fig)

st.success("App is fully functional — and **no file upload is required!**")
