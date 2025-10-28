# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay

# # ========== Load and preprocess data ==========
# df = load_data()
# clean_and_impute_data(df)

# # ========== Sidebar ==========
# st.sidebar.markdown("#  Churn Modeling Dashboard")
# st.sidebar.markdown("""
# This dashboard analyzes the **Churn Modeling dataset** using **Logistic Regression**  
# and visualizes relationships between features and customer churn.

# **Features include:**
# - CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
# - Categorical: Geography, Gender  
# - Target: Exited (0 = Stayed, 1 = Churned)

# **Instructions**:  
# 1. Explore dataset & visualizations below.  
# 2. Press **Run Logistic Regression** to train and view model results.
# """)

# # ========== Main Page ==========
# st.markdown("##  Dataset Overview")
# st.dataframe(df.head())

# # ---------------------------------------------------------------------
# # 🧭 NEW PLOT 1: CHURN DISTRIBUTION
# # ---------------------------------------------------------------------
# st.markdown("###  Churn Distribution")
# fig, ax = plt.subplots()
# sns.countplot(x='Exited', data=df, palette='Set2', ax=ax)
# ax.set_xticklabels(['Stayed (0)', 'Churned (1)'])
# ax.set_ylabel("Count")
# st.pyplot(fig)
# st.markdown("""
# **Interpretation**:  
# This chart shows how many customers stayed versus churned.  
# If churned customers are significantly fewer, the dataset is imbalanced — which may affect model learning.
# """)

# # ---------------------------------------------------------------------
# # 🧭 NEW PLOT 2: AGE DISTRIBUTION BY CHURN
# # ---------------------------------------------------------------------
# st.markdown("###  Age Distribution by Churn")
# fig, ax = plt.subplots()
# sns.kdeplot(x='Age', hue='Exited', data=df, fill=True, common_norm=False, palette='Set1', ax=ax)
# ax.set_xlabel("Age")
# ax.set_ylabel("Density")
# st.pyplot(fig)
# st.markdown("""
# **Interpretation**:  
# This shows how the age distribution differs between customers who stayed and those who churned.  
# Peaks or separations between the curves indicate which age groups are more likely to leave.
# """)

# # ---------------------------------------------------------------------
# # 🧭 NEW PLOT 3: BALANCE VS SALARY SCATTER
# # ---------------------------------------------------------------------
# st.markdown("###  Balance vs Estimated Salary (Colored by Churn)")
# fig, ax = plt.subplots()
# sns.scatterplot(x='Balance', y='EstimatedSalary', hue='Exited', data=df, palette='coolwarm', alpha=0.6, ax=ax)
# st.pyplot(fig)
# st.markdown("""
# **Interpretation**:  
# Each point represents a customer.  
# The color indicates churn (red = churned, blue = stayed).  
# Patterns or clusters show how salary and balance levels relate to churn.
# """)

# # ---------------------------------------------------------------------
# # 🧭 NEW PLOT 4: GEOGRAPHY VS CHURN
# # ---------------------------------------------------------------------
# st.markdown("###  Churn Rate by Geography")
# fig, ax = plt.subplots()
# sns.barplot(x='Geography', y='Exited', data=df, estimator='mean', palette='viridis', ax=ax)
# ax.set_ylabel("Average Churn Rate")
# st.pyplot(fig)
# st.markdown("""
# **Interpretation**:  
# Displays the average churn rate per country.  
# Higher bars mean customers from that region are more likely to leave.  
# Useful for identifying location-based churn trends.
# """)

# # ========== Logistic Regression Section ==========
# if st.button("Run Logistic Regression"):
#     st.markdown("##  Logistic Regression Training & Evaluation")

#     # Prepare data
#     numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
#                'HasCrCard','IsActiveMember','EstimatedSalary']
#     catdata = ['Geography','Gender']
#     X_processed, y = encode_and_scale(df, numdata, catdata)

#     X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

#     # Train model
#     model = LogisticRegression(max_iter=500)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:,1]

#     # ---- Accuracy & ROC-AUC ----
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_proba)
#     st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")

#     # ---- Confusion Matrix ----
#     st.markdown("### Confusion Matrix")
#     fig, ax = plt.subplots()
#     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Actual")
#     st.pyplot(fig)
#     st.markdown("""
#     **Interpretation**:  
#     The diagonal cells show correct predictions.  
#     Off-diagonal cells represent misclassifications.  
#     More blue on the diagonal = better model accuracy.
#     """)

#     # ---- ROC Curve ----
#     st.markdown("### ROC Curve")
#     fig, ax = plt.subplots()
#     RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
#     st.pyplot(fig)
#     st.markdown("""
#     **Interpretation**:  
#     The closer the curve is to the top-left corner, the better the model distinguishes churn vs non-churn.  
#     A higher AUC value means stronger classification power.
#     """)

#     # ---- Predicted Probability Distribution ----
#     st.markdown("### Predicted Probability Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(y_proba, bins=20, kde=True, color='skyblue', ax=ax)
#     ax.set_xlabel("Predicted Probability of Churn")
#     ax.set_ylabel("Number of Customers")
#     st.pyplot(fig)
#     st.markdown("""
#     **Interpretation**:  
#     This shows how confident the model is in predicting churn.  
#     Values near 0 mean low churn likelihood, while values near 1 indicate high risk of churn.
#     """)

    
#     # ---- Top Correlated Features ----
#     corr = compute_correlation(X_processed, y)
#     top_features = top_correlated_features(corr)
#     st.markdown("###  Top Correlated Features with Target")
#     st.write(top_features)
#     st.markdown("""
#     **Interpretation**:  
#     Shows the top features most linearly related to the target (`Exited`).  
#     Helps identify which attributes most influence customer churn.
#     """)

#     # ---------------------------------------------------------------------
#     # 🧭 NEW PLOT 5: CORRELATION HEATMAP
#     # ---------------------------------------------------------------------
#     st.markdown("###  Correlation Heatmap (All Features)")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
#     st.pyplot(fig)
#     st.markdown("""
#     **Interpretation**:  
#     Displays pairwise correlations among all features.  
#     Red = positive correlation, Blue = negative correlation.  
#     Useful for spotting multicollinearity or strong feature relationships.
#     """)

#----------------------------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from back import (
#     load_data, clean_and_impute_data,
#     encode_and_scale, compute_correlation, top_correlated_features
# )
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, confusion_matrix, roc_auc_score,
#     RocCurveDisplay, PrecisionRecallDisplay
# )

# # ---------- PAGE CONFIG ----------
# st.set_page_config(page_title="Churn Modeling Dashboard", layout="wide")
# sns.set_style("darkgrid")

# # ---------- HEADER / INFO BAR ----------
# st.markdown("""
# <h1 style='text-align:center;'>Customer Churn Prediction Dashboard</h1>
# <p style='text-align:center;'>
# Using <b>Logistic Regression</b> on the <i>Churn Modeling</i> dataset to identify customers likely to leave.
# </p>
# <hr>
# """, unsafe_allow_html=True)

# # ---------- LOAD DATA ----------
# df = load_data()
# clean_and_impute_data(df)

# # ---------- DATA OVERVIEW ----------
# st.markdown("### 📊 Dataset Overview")
# st.dataframe(df.head())

# # ---------- RUN BUTTON ----------
# if st.button("▶️ Run Logistic Regression"):

#     numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
#                'HasCrCard','IsActiveMember','EstimatedSalary']
#     catdata = ['Geography','Gender']
#     X_processed, y = encode_and_scale(df, numdata, catdata)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_processed, y, test_size=0.2, random_state=42)

#     model = LogisticRegression(max_iter=500)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:,1]

#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_proba)

#     # KPI cards
#     c1, c2 = st.columns(2)
#     c1.metric("Model Accuracy", f"{acc*100:.2f}%")
#     c2.metric("ROC-AUC Score", f"{auc:.2f}")
#     st.markdown("---")

#     # ---------- FIGURES ----------
#     # 1. Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     fig_cm, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax)
#     ax.set_title("Confusion Matrix")
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

#     # 2. ROC Curve
#     fig_roc, ax = plt.subplots()
#     RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
#     ax.set_title("ROC Curve")

#     # 3. Probability Distribution
#     fig_prob, ax = plt.subplots()
#     sns.histplot(y_proba, bins=20, kde=True, color="skyblue", ax=ax)
#     ax.set_title("Predicted Probability Distribution")
#     ax.set_xlabel("Predicted Probability of Churn")

#     # 4. Top Correlated Features
#     corr = compute_correlation(X_processed, y)
#     top_features = top_correlated_features(corr)
#     fig_corr, ax = plt.subplots()
#     sns.barplot(x=top_features.values, y=top_features.index, palette="cool", ax=ax)
#     ax.set_title("Top Correlated Features with Target")
#     ax.set_xlabel("Correlation Coefficient")

#     # 5. Precision-Recall Curve
#     fig_pr, ax = plt.subplots()
#     PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
#     ax.set_title("Precision-Recall Curve")

#     # Match Age/Balance to test set rows only
#     df = df.reset_index(drop=True)
#     X_processed_df = pd.DataFrame(X_processed)

#     # Identify test indices
#     _, X_test_idx = train_test_split(
#     df.index, test_size=0.2, random_state=42
#     )

#     df_test = df.iloc[X_test_idx]

#     # 6. Age vs Predicted Probability
#     fig_age, ax = plt.subplots()
#     sns.scatterplot(x=df_test["Age"], y=y_proba, alpha=0.6, color="mediumseagreen", ax=ax)
#     ax.set_title("Age vs Predicted Churn Probability")
#     ax.set_xlabel("Age"); ax.set_ylabel("Predicted Churn Probability")

#     # 7. Balance vs Predicted Probability
#     fig_bal, ax = plt.subplots()
#     sns.scatterplot(x=df_test["Balance"], y=y_proba, alpha=0.6, color="coral", ax=ax)
#     ax.set_title("Balance vs Predicted Churn Probability")
#     ax.set_xlabel("Account Balance"); ax.set_ylabel("Predicted Churn Probability")


#     # ---------- GRID LAYOUT ----------
#     # Row 1 – Core performance plots
#     col1, col2, col3 = st.columns(3)
#     col1.pyplot(fig_cm)
#     col2.pyplot(fig_roc)
#     col3.pyplot(fig_prob)

#     # Row 2 – Feature insights
#     col4, col5, col6 = st.columns(3)
#     col4.pyplot(fig_corr)
#     col5.pyplot(fig_pr)
#     col6.pyplot(fig_age)

#     # Row 3 – Additional pattern view
#     col7, _ , _ = st.columns([1,1,1])
#     col7.pyplot(fig_bal)

#     # ---------- INTERPRETATIONS ----------
#     st.markdown("---")
#     st.markdown("""
# ### 🧩 Interpretations
# - **Confusion Matrix** → Diagonal cells = correct predictions; off-diagonal = errors.  
# - **ROC Curve** → Curve closer to top-left means better classification.  
# - **Probability Distribution** → Shows how confident the model is (near 1 = high churn risk).  
# - **Top Correlations** → Features most linked to churn.  
# - **Precision-Recall** → Balance between model precision and recall.  
# - **Age vs Probability** → Older customers tend to have higher or lower churn probabilities depending on data spread.  
# - **Balance vs Probability** → Relationship between account balance and likelihood to churn.
# """)

# else:
#     st.info("👆 Click 'Run Logistic Regression' to view model results and dashboard charts.")

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
st.markdown("""
<style>
body, .stApp {
    background-color: #f5f6fa;
    color: #1e1e1e;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #1e1e1e;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #000000; /* changed from blue to black */
    font-weight: 600;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
hr {
    border: 1px solid #dcdcdc;
}
</style>
""", unsafe_allow_html=True)

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

# Refresh button for dynamic reloading
if st.button("🔄 Refresh Dataset"):
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
