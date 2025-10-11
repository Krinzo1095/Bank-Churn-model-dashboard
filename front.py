import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features

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

# Hyperparameter sliders (example)
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 50, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
elif model_choice == "ANN":
    epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50, step=10)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)

st.markdown("## 📄 Dataset Overview")
st.markdown("This dashboard uses the Churn Modeling dataset, containing customer features like Age, Geography, CreditScore, etc., and the target column `Exited` indicating churn.")
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
