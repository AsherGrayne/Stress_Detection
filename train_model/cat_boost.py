import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    accs = []

    for e in EPOCHS:
        print(f"[AdaBoost] n_estimators={e}")
        model = AdaBoostClassifier(n_estimators=e, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(" ->", acc)
        accs.append(acc)

    os.makedirs("epoch", exist_ok=True)
    plt.plot(EPOCHS, accs, marker='o')
    plt.xscale("log")
    plt.title("AdaBoost Accuracy vs Epochs")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("epoch/adaboost_accuracy.png")

if __name__ == "__main__":
    main()
