import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "merged_data_datetime_broken.csv")
LABEL = "label"
FEATURES = [
    'X', 'Y', 'Z', 'EDA', 'HR', 'TEMP',
    'datetime_year', 'datetime_month', 'datetime_day', 'datetime_hour', 'datetime_dow'
]

MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
METRICS_DIR = os.path.join(PROJECT_ROOT, "model_metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def load_data(csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    X = df[FEATURES].astype(float)
    y = df[LABEL].astype(int)
    
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train_individual_models(X_train, y_train, X_test, y_test):
    models = {}
    metrics = {}
    
    print("\n" + "="*60)
    print("Training Individual Models")
    print("="*60)
    
    print("\n[1/6] Training Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        models['Random Forest'] = rf
        metrics['Random Forest'] = {
            'accuracy': float(rf_acc),
            'predictions': rf_pred.tolist()
        }
        print(f"  ✓ Accuracy: {rf_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n[2/6] Training Logistic Regression...")
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        models['Logistic Regression'] = lr
        metrics['Logistic Regression'] = {
            'accuracy': float(lr_acc),
            'predictions': lr_pred.tolist()
        }
        print(f"  ✓ Accuracy: {lr_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n[3/6] Training XGBoost...")
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['XGBoost'] = xgb_model
        metrics['XGBoost'] = {
            'accuracy': float(xgb_acc),
            'predictions': xgb_pred.tolist()
        }
        print(f"  ✓ Accuracy: {xgb_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n[4/6] Training LightGBM...")
    try:
        lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        models['LightGBM'] = lgb_model
        metrics['LightGBM'] = {
            'accuracy': float(lgb_acc),
            'predictions': lgb_pred.tolist()
        }
        print(f"  ✓ Accuracy: {lgb_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n[5/6] Training AdaBoost...")
    try:
        ada = AdaBoostClassifier(n_estimators=200, random_state=42)
        ada.fit(X_train, y_train)
        ada_pred = ada.predict(X_test)
        ada_acc = accuracy_score(y_test, ada_pred)
        models['AdaBoost'] = ada
        metrics['AdaBoost'] = {
            'accuracy': float(ada_acc),
            'predictions': ada_pred.tolist()
        }
        print(f"  ✓ Accuracy: {ada_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n[6/6] Training KNN...")
    try:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        models['KNN'] = knn
        metrics['KNN'] = {
            'accuracy': float(knn_acc),
            'predictions': knn_pred.tolist()
        }
        print(f"  ✓ Accuracy: {knn_acc:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return models, metrics

def save_models_and_metrics(models, metrics):
    print("\n" + "="*60)
    print("Saving Models and Metrics")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.joblib'
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        print(f"  ✓ Saved: {filepath}")
    
    best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['accuracy'])
    best_accuracy = metrics[best_model_name]['accuracy']
    
    metrics_data = {
        'timestamp': timestamp,
        'individual_models': metrics,
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'model_names': list(models.keys())
    }
    
    metrics_path = os.path.join(METRICS_DIR, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    latest_metrics_path = os.path.join(METRICS_DIR, 'latest_metrics.json')
    with open(latest_metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"  ✓ Saved: {metrics_path}")
    print(f"  ✓ Saved: {latest_metrics_path}")
    
    return metrics_data

def main():
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    print(f"Dataset: {CSV_PATH}")
    print(f"Features: {', '.join(FEATURES)}")
    print(f"Label: {LABEL}")
    
    try:
        print("\nLoading data...")
        X_train, X_test, y_train, y_test = load_data()
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        models, metrics = train_individual_models(X_train, y_train, X_test, y_test)
        
        if len(models) == 0:
            print("\n✗ No models were successfully trained!")
            return
        
        metrics_data = save_models_and_metrics(models, metrics)
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"\nModel Accuracies:")
        for name in sorted(metrics.keys(), key=lambda x: metrics[x]['accuracy'], reverse=True):
            print(f"  {name:20s}: {metrics[name]['accuracy']:.4f}")
        
        best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['accuracy'])
        best_accuracy = metrics[best_model_name]['accuracy']
        print(f"\nBest Model: {best_model_name} ({best_accuracy:.4f})")
        
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

