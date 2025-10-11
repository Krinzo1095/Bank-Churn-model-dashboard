import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # needed before import
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer

# loading the data set
def load_data():
    # automatically find CSV file in same folder as this script
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Churn Modeling.csv")
    df = pd.read_csv(file_path)
    return df

def clean_and_impute_data(df):
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    target = 'Exited'
    numdata = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary']
    catdata = ['Geography','Gender']

    # numeric → mean
    for col in numdata:
        df[col] = df[col].fillna(df[col].mean())
    # categorical → mode
    for col in catdata:
        df[col] = df[col].fillna(df[col].mode()[0])

def encode_and_scale(df, numdata, catdata):
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numdata),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), catdata)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

def compute_correlation(X_processed, y):
    processed_df = pd.DataFrame(X_processed)
    processed_df['Exited'] = y.values
    corr = processed_df.corr()
    return corr

def top_correlated_features(corr, target='Exited', n=5):
    top_corr = corr[target].sort_values(ascending=False).head(n)
    return top_corr
