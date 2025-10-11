import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # needed before import
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer

# loading the data set
def load_data():
    df=pd.read_csv('/Users/krishsolanki/fds mini/Churn Modeling.csv')
    return df

def clean_and_impute_data(df):
    df=df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    target='Exited'
    numdata=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
    catdata=['Geography','Gender']
    all_features=numdata+catdata+[target]

    # numeric -> mean
    for col in numdata:
        df[col] = df[col].fillna(df[col].mean())
    # categorical -> mode
    for col in catdata:
        df[col] = df[col].fillna(df[col].mode()[0])

def encode_and_scale(df,numdata,catdata):
    X=df.drop('Exited',axis=1)
    y=df['Exited']

    # Create the ColumnTransformer
    preprocessor=ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numdata),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), catdata)]
    )

    # Fit and transform the features
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

# column dropping, imputation, encoding and scalling DONE!

def compute_correlation(X_processed, y):
    # Combine processed features and target into one DataFrame
    processed_df = pd.DataFrame(X_processed)
    processed_df['Exited'] = y.values

    # Compute correlation matrix
    corr = processed_df.corr()

    return corr

def top_correlated_features(corr, target='Exited', n=5):
    top_corr = corr[target].sort_values(ascending=False).head(n)
    return top_corr
