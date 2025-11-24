import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb

CSV_PATH = "merged_data_datetime_broken.csv"
LABEL = "label"

EPOCHS = [100,150,200,500,600,2000,4000,6000,8000,10000]

def load_data():
    df = pd.read_csv(CSV_PATH)
    features = ['X','Y','Z','EDA','HR','TEMP','datetime_year','datetime_month','datetime_day','datetime_hour','datetime_dow']
    X = df[features].astype(float)
    y = df[LABEL]
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def main():
    X_train, X_test, y_train, y_test = load_data()
    accuracies = []

    for e in EPOCHS:
        print(f"[XGBoost] n_estimators={e}")
        model = xgb.XGBClassifier(
            n_estimators=e,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(" ->", acc)
        accuracies.append(acc)

    os.makedirs("epoch", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(EPOCHS, accuracies, marker='o')
    plt.xscale("log")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title("XGBoost Accuracy vs Epochs")
    plt.grid(True)
    plt.savefig("epoch/xgboost_accuracy.png")

if __name__ == "__main__":
    main()
