import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------------
# DIRECTLY EDIT THIS ONLY
CSV_PATH = "merged_data_datetime_broken.csv"   # <--- set your CSV file here
LABEL = "label"

# Number of "initiators" (here used as n_estimators of RF)
INITIATORS = [100,150,200,500,600,2000,4000,6000,8000,10000]
# -------------------------------------------------------

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

def main():
    X_train, X_test, y_train, y_test = load_data()
    accuracies = []

    for n in INITIATORS:
        print(f"[Random Forest] n_estimators = {n}")
        model = RandomForestClassifier(
            n_estimators=n,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f" -> Accuracy: {acc:.4f}")

        accuracies.append(acc)

    # Save plot
    os.makedirs("epoch", exist_ok=True)
    output_file = "epoch/random_forest_initiators_accuracy.png"

    plt.figure(figsize=(9,5))
    plt.plot(INITIATORS, accuracies, marker='o')
    plt.xscale("log")
    plt.xlabel("Number of Initiators (n_estimators)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest: Accuracy vs Number of Initiators")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)

    print(f"\nSaved graph to: {output_file}")

if __name__ == "__main__":
    main()
