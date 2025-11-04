import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import (
    load_data, clean_and_impute_data,
    encode_and_scale, compute_correlation, top_correlated_features
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Churn Modeling Dashboard", layout="wide")
sns.set_style("whitegrid")

# ---------- FIXED STYLING ----------


# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;margin-bottom:0;'>Customer Churn Prediction Dashboard</h1>
<p style='text-align:center;margin-top:0;'>
Using <b>Logistic Regression</b> on the <i>Churn Modeling</i> dataset to identify customers likely to leave.
</p>
<div style="background:linear-gradient(90deg,#0077ff,#7dd3fc);
height:6px;border-radius:3px;margin-bottom:15px;"></div>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
st.markdown("### 📊 Dataset Overview")

# Refresh button 
if st.button("🔄"):
    df = load_data()
    clean_and_impute_data(df)
    st.success("✅ Dataset reloaded successfully! All visuals will use the latest data.")
else:
    df = load_data()
    clean_and_impute_data(df)

# Show preview of data
st.dataframe(df.head())
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")


# ---------- RUN MODEL ----------
if st.button("▶️ Run Logistic Regression"):

    numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary']
    catdata = ['Geography','Gender']

    X_processed, y = encode_and_scale(df, numdata, catdata)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # KPI CARDS
    k1, k2 = st.columns(2)
    k1.metric("Model Accuracy", f"{acc*100:.2f}%")
    k2.metric("ROC-AUC Score", f"{auc:.2f}")
    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------- FIGURES ----------
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ROC Curve
    fig_roc, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve")

    # Probability Distribution
    fig_prob, ax = plt.subplots()
    sns.histplot(y_proba, bins=20, kde=True, color="#64b5f6", ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability of Churn")

    # Top Correlated Features
    corr = compute_correlation(X_processed, y)
    top_features = top_correlated_features(corr)
    fig_corr, ax = plt.subplots()
    sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm", ax=ax)
    ax.set_title("Top Correlated Features with Target")
    ax.set_xlabel("Correlation Coefficient")

    # Precision-Recall Curve
    fig_pr, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve")

    # Align Age with test indices
    df = df.reset_index(drop=True)
    _, X_test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    df_test = df.iloc[X_test_idx]

    # Age vs Probability
    fig_age, ax = plt.subplots()
    sns.scatterplot(x=df_test["Age"], y=y_proba, alpha=0.6, color="mediumseagreen", ax=ax)
    ax.set_title("Age vs Predicted Churn Probability")
    ax.set_xlabel("Age"); ax.set_ylabel("Predicted Churn Probability")

    # ---------- GRID LAYOUT ----------
    # Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.pyplot(fig_cm)
    r1c2.pyplot(fig_roc)
    r1c3.pyplot(fig_prob)

    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.pyplot(fig_corr)
    r2c2.pyplot(fig_pr)
    r2c3.pyplot(fig_age)

    # ---------- INTERPRETATIONS ----------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    ### 🧩 Model Insights
    | Chart | Interpretation |
    |:------|:----------------|
    | **Confusion Matrix** | Diagonal cells show correct predictions; off-diagonal = misclassifications |
    | **ROC Curve** | Curve closer to top-left = better performance |
    | **Probability Distribution** | Confidence spread; near 1 → high churn likelihood |
    | **Top Correlations** | Strongest churn-influencing features |
    | **Precision-Recall** | Trade-off between precision & recall |
    | **Age vs Probability** | Relationship between age and churn likelihood |
    """)

else:
    st.info("👆 Click **Run Logistic Regression** to view model results and dashboard charts.")
