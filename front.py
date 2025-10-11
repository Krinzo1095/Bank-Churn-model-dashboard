import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features

# =================== LOAD DATA ===================
df = load_data()
clean_and_impute_data(df)

# =================== SIDEBAR ===================
st.sidebar.markdown("# Churn Modeling Dashboard")
st.sidebar.markdown("""
This dashboard uses the **Churn Modeling dataset**.  
It applies **Logistic Regression** to predict customer churn (`Exited`).  

**Dataset Features**:
- CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
- Categorical: Geography, Gender  
- Target: Exited (0 = Stayed, 1 = Churned)  

**Instructions**:  
1. Inspect the dataset below.  
2. Press **Run Logistic Regression** to train the model and see performance metrics and plots.
""")

# =================== MAIN PAGE ===================
st.markdown("## 📄 Dataset Overview")
st.dataframe(df.head())

if st.button("Run Logistic Regression"):
    st.markdown("## 🤖 Logistic Regression Training & Evaluation")

    # Prepare data
    numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary']
    catdata = ['Geography','Gender']
    X_processed, y = encode_and_scale(df, numdata, catdata)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Accuracy & ROC-AUC
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")

    # =================== Confusion Matrix ===================
    cm = confusion_matrix(y_test, y_pred)
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("**Interpretation**: Rows = Actual, Columns = Predicted. Blue diagonal cells show correctly predicted samples. Off-diagonal cells show misclassifications.")

    # =================== ROC Curve ===================
    st.markdown("### ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    st.pyplot(fig)
    st.markdown("**Interpretation**: X-axis = False Positive Rate, Y-axis = True Positive Rate. A curve closer to top-left indicates better performance. Area under curve (AUC) is also shown above.")

    # =================== Predicted Probability Distribution ===================
    st.markdown("### Predicted Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(y_proba, bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Predicted Probability of Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    st.markdown("**Interpretation**: Shows how confident the model is in predicting churn. Values near 0 = low chance of churn, near 1 = high chance of churn. Peaks show where most predictions lie.")

    # =================== Top Correlated Features ===================
    corr = compute_correlation(X_processed, y)
    top_features = top_correlated_features(corr)
    st.markdown("### 🔝 Top Correlated Features with Target")
    st.write(top_features)
    st.markdown("**Interpretation**: Features at the top have the strongest correlation with customer churn (Exited).")
