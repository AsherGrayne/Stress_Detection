import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_PATH = "datasets/merged_data_datetime_broken.csv"
LABEL = "label"

FEATURES = [
    'X','Y','Z','EDA','HR','TEMP',
    'datetime_year','datetime_month','datetime_day','datetime_hour','datetime_dow'
]

def load_data():
    df = pd.read_csv(CSV_PATH)
    
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    X = df[FEATURES].astype(float)
    y = df[LABEL]
    
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def fit_eval(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} - Accuracy: {acc:.4f}")
    return model

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    os.makedirs("saved_models", exist_ok=True)
    
    print("\nTraining Logistic Regression...")
    logreg = LogisticRegression(max_iter=2000)
    pipe_logreg = fit_eval(
        logreg,
        X_train, y_train,
        X_test, y_test,
        name="LogisticRegression"
    )
    joblib.dump(pipe_logreg, "saved_models/logistic_regression.joblib")
    print("Saved Logistic Regression model to saved_models/logistic_regression.joblib")
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    pipe_rf = fit_eval(
        rf,
        X_train, y_train,
        X_test, y_test,
        name="RandomForest"
    )
    joblib.dump(pipe_rf, "saved_models/random_forest.joblib")
    print("Saved Random Forest model to saved_models/random_forest.joblib")
    
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main()

