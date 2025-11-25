import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_PATH = "merged_data_datetime_broken.csv"
LABEL = "label"

INITIATORS = [100,150,200,500,600,2000,4000,6000,8000,10000]

FEATURES = [
    'X','Y','Z','EDA','HR','TEMP',
    'datetime_year','datetime_month','datetime_day','datetime_hour','datetime_dow'
]

def load_data():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURES].astype(float)
    y = df[LABEL]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def predict_in_batches(model, X_test, batch_size=10000):
    predictions = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test.iloc[i:i+batch_size] if isinstance(X_test, pd.DataFrame) else X_test[i:i+batch_size]
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds)
    return np.array(predictions)

def main():
    X_train, X_test, y_train, y_test = load_data()
    accuracies = []
    successful_n = []

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()

    for n in INITIATORS:
        print(f"[KNN] n_neighbors = {n}")
        
        test_sample_size = None
        if n > 2000:
            test_sample_size = max(1000, len(X_test) // 10)
            print(f"  Using test sample size: {test_sample_size} (to avoid memory issues)")
            X_test_sample = X_test.sample(n=test_sample_size, random_state=42)
            y_test_sample = y_test.loc[X_test_sample.index]
        else:
            X_test_sample = X_test
            y_test_sample = y_test

        try:
            model = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
            model.fit(X_train, y_train)

            if len(X_test_sample) > 10000:
                preds = predict_in_batches(model, X_test_sample, batch_size=5000)
            else:
                preds = model.predict(X_test_sample)
            
            acc = accuracy_score(y_test_sample, preds)
            print(f" -> Accuracy: {acc:.4f}")
            accuracies.append(acc)
            successful_n.append(n)
            
        except MemoryError as e:
            print(f" -> Memory Error: Skipping n_neighbors={n} (requires too much memory)")
            print(f"   Error: {str(e)[:100]}")
            accuracies.append(None)
            successful_n.append(n)
            continue
        except Exception as e:
            print(f" -> Error: {e}")
            accuracies.append(None)
            successful_n.append(n)
            continue

    os.makedirs("epoch", exist_ok=True)
    out_file = "epoch/knn_initiators_accuracy.png"

    valid_n = [n for n, acc in zip(successful_n, accuracies) if acc is not None]
    valid_acc = [acc for acc in accuracies if acc is not None]
    
    if valid_n:
        plt.figure(figsize=(9,5))
        plt.plot(valid_n, valid_acc, marker='o')
        plt.xscale("log")
        plt.xlabel("Number of Initiators (n_neighbors)")
        plt.ylabel("Accuracy")
        plt.title("KNN: Accuracy vs Number of Initiators")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_file)
        print(f"\nSaved graph to: {out_file}")
        print(f"Successfully tested {len(valid_n)}/{len(INITIATORS)} n_neighbors values")
    else:
        print("\nNo successful runs to plot. All n_neighbors values failed due to memory issues.")

if __name__ == "__main__":
    main()
