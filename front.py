import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay

# ========== Load and preprocess data ==========
df = load_data()
clean_and_impute_data(df)

# ========== Sidebar ==========
st.sidebar.markdown("# 📊 Churn Modeling Dashboard")
st.sidebar.markdown("""
This dashboard analyzes the **Churn Modeling dataset** using **Logistic Regression**  
and visualizes relationships between features and customer churn.

**Features include:**
- CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
- Categorical: Geography, Gender  
- Target: Exited (0 = Stayed, 1 = Churned)

**Instructions**:  
1. Explore dataset & visualizations below.  
2. Press **Run Logistic Regression** to train and view model results.
""")

# ========== Main Page ==========
st.markdown("## 📄 Dataset Overview")
st.dataframe(df.head())

# ---------------------------------------------------------------------
# 🧭 NEW PLOT 1: CHURN DISTRIBUTION
# ---------------------------------------------------------------------
st.markdown("### ⚖️ Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Exited', data=df, palette='Set2', ax=ax)
ax.set_xticklabels(['Stayed (0)', 'Churned (1)'])
ax.set_ylabel("Count")
st.pyplot(fig)
st.markdown("""
**Interpretation**:  
This chart shows how many customers stayed versus churned.  
If churned customers are significantly fewer, the dataset is imbalanced — which may affect model learning.
""")

# ---------------------------------------------------------------------
# 🧭 NEW PLOT 2: AGE DISTRIBUTION BY CHURN
# ---------------------------------------------------------------------
st.markdown("### 👥 Age Distribution by Churn")
fig, ax = plt.subplots()
sns.kdeplot(x='Age', hue='Exited', data=df, fill=True, common_norm=False, palette='Set1', ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
st.pyplot(fig)
st.markdown("""
**Interpretation**:  
This shows how the age distribution differs between customers who stayed and those who churned.  
Peaks or separations between the curves indicate which age groups are more likely to leave.
""")

# ---------------------------------------------------------------------
# 🧭 NEW PLOT 3: BALANCE VS SALARY SCATTER
# ---------------------------------------------------------------------
st.markdown("### 💰 Balance vs Estimated Salary (Colored by Churn)")
fig, ax = plt.subplots()
sns.scatterplot(x='Balance', y='EstimatedSalary', hue='Exited', data=df, palette='coolwarm', alpha=0.6, ax=ax)
st.pyplot(fig)
st.markdown("""
**Interpretation**:  
Each point represents a customer.  
The color indicates churn (red = churned, blue = stayed).  
Patterns or clusters show how salary and balance levels relate to churn.
""")

# ---------------------------------------------------------------------
# 🧭 NEW PLOT 4: GEOGRAPHY VS CHURN
# ---------------------------------------------------------------------
st.markdown("### 🌍 Churn Rate by Geography")
fig, ax = plt.subplots()
sns.barplot(x='Geography', y='Exited', data=df, estimator='mean', palette='viridis', ax=ax)
ax.set_ylabel("Average Churn Rate")
st.pyplot(fig)
st.markdown("""
**Interpretation**:  
Displays the average churn rate per country.  
Higher bars mean customers from that region are more likely to leave.  
Useful for identifying location-based churn trends.
""")

# ========== Logistic Regression Section ==========
if st.button("Run Logistic Regression"):
    st.markdown("## 🤖 Logistic Regression Training & Evaluation")

    # Prepare data
    numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary']
    catdata = ['Geography','Gender']
    X_processed, y = encode_and_scale(df, numdata, catdata)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # ---- Accuracy & ROC-AUC ----
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")

    # ---- Confusion Matrix ----
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    The diagonal cells show correct predictions.  
    Off-diagonal cells represent misclassifications.  
    More blue on the diagonal = better model accuracy.
    """)

    # ---- ROC Curve ----
    st.markdown("### ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    The closer the curve is to the top-left corner, the better the model distinguishes churn vs non-churn.  
    A higher AUC value means stronger classification power.
    """)

    # ---- Predicted Probability Distribution ----
    st.markdown("### Predicted Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(y_proba, bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Predicted Probability of Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    This shows how confident the model is in predicting churn.  
    Values near 0 mean low churn likelihood, while values near 1 indicate high risk of churn.
    """)

    
    # ---- Top Correlated Features ----
    corr = compute_correlation(X_processed, y)
    top_features = top_correlated_features(corr)
    st.markdown("### 🔝 Top Correlated Features with Target")
    st.write(top_features)
    st.markdown("""
    **Interpretation**:  
    Shows the top features most linearly related to the target (`Exited`).  
    Helps identify which attributes most influence customer churn.
    """)

    # ---------------------------------------------------------------------
    # 🧭 NEW PLOT 5: CORRELATION HEATMAP
    # ---------------------------------------------------------------------
    st.markdown("### 🌡️ Correlation Heatmap (All Features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    Displays pairwise correlations among all features.  
    Red = positive correlation, Blue = negative correlation.  
    Useful for spotting multicollinearity or strong feature relationships.
    """)
