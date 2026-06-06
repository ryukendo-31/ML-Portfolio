import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
def main():
    print("start")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors= "coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    X = df.drop(columns=['customerID', 'Churn'])
    y = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)


    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            # drop='first' prevents multicollinearity 
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features) 
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nTraining Decision Tree...")
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
    ])

    dt_pipeline.fit(X_train, y_train)
    dt_preds = dt_pipeline.predict(X_test)
    dt_probs = dt_pipeline.predict_proba(X_test)[:, 1]

    print("=== Single Decision Tree Performance ===")
    print(classification_report(y_test, dt_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, dt_probs):.4f}\n")


if __name__ == "__main__":
    main()