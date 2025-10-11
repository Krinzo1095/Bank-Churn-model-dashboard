import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features

# Load and clean data
df = load_data()
clean_and_impute_data(df)

st.sidebar.markdown("# 🗂️ Model Controls")
st.sidebar.markdown("""
This dashboard uses the Churn Modeling dataset.  
Adjust the model and hyperparameters below to see how the predictions and performance metrics change.
""")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Random Forest", "ANN"]
)

# Hyperparameter sliders
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 50, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
elif model_choice == "ANN":
    epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50, step=10)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)

# Display dataset overview
st.markdown("## 📄 Dataset Overview")
st.dataframe(df.head())

with st.expander("ℹ️ Preprocessing Details"):
    st.markdown("""
    - Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`  
    - Numeric missing values replaced with mean  
    - Categorical missing values replaced with mode  
    - Scaled numeric features using StandardScaler  
    - OneHotEncoded categorical features: `Geography`, `Gender`
    """)

st.info("""
- Top correlated features with churn: CreditScore, Age, Geography, IsActiveMember  
- ANN model converges well with standardized numeric features  
- RandomForest highlights feature importance clearly
""")

# ===================== MODEL RUN SECTION =====================
if st.button("Run Model"):
    st.markdown("## 🤖 Model Training & Evaluation")

    # Prepare data for modeling
    numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary']
    catdata = ['Geography','Gender']

    X_processed, y = encode_and_scale(df, numdata, catdata)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Logistic Regression
    if model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)

    # Random Forest
    elif model_choice == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)

        # Feature importance plot
        importances = model.feature_importances_
        feature_names = numdata + list(model.feature_names_in_[-len(catdata):])
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        st.markdown("### Feature Importance")
        st.bar_chart(fi_df.set_index("Feature"))

    # ANN
    elif model_choice == "ANN":
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

        model = MLPClassifier(hidden_layer_sizes=(50,50), learning_rate_init=learning_rate,
                              max_iter=epochs, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)

    st.markdown("---")

    # Optional: Top correlated features display
    corr = compute_correlation(X_processed, y)
    top_features = top_correlated_features(corr)
    st.markdown("### 🔝 Top Correlated Features with Target")
    st.write(top_features)

# ===================== END MODEL RUN =====================

if __name__ == "__main__":
    st.write("App loaded successfully.")
