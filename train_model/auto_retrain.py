import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from train_ensemble import load_data, train_individual_models, save_models_and_metrics
except ImportError:
    from train_model.train_ensemble import load_data, train_individual_models, save_models_and_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_COLLECTION_FILE = os.path.join(PROJECT_ROOT, "collected_data.json")
RETRAIN_CONFIG_FILE = os.path.join(PROJECT_ROOT, "retrain_config.json")
METRICS_DIR = os.path.join(PROJECT_ROOT, "model_metrics")

os.makedirs(METRICS_DIR, exist_ok=True)

class AutoRetrainer:
    def __init__(self):
        self.config = self.load_config()
        self.collected_data = self.load_collected_data()
    
    def load_config(self):
        default_config = {
            "min_samples_for_retrain": 100,
            "retrain_interval_hours": 24,
            "min_accuracy_improvement": 0.01,
            "keep_best_model": True,
            "last_retrain": None,
            "retrain_count": 0
        }
        
        if os.path.exists(RETRAIN_CONFIG_FILE):
            with open(RETRAIN_CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def save_config(self):
        with open(RETRAIN_CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_collected_data(self):
        if os.path.exists(DATA_COLLECTION_FILE):
            with open(DATA_COLLECTION_FILE, 'r') as f:
                return json.load(f)
        return {
            "samples": [],
            "last_updated": None
        }
    
    def save_collected_data(self):
        self.collected_data["last_updated"] = datetime.now().isoformat()
        with open(DATA_COLLECTION_FILE, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
    
    def add_sample(self, sensor_data, predictions, user_feedback=None):
        sample = {
            "timestamp": datetime.now().isoformat(),
            "sensor_data": sensor_data,
            "predictions": predictions,
            "user_feedback": user_feedback
        }
        
        self.collected_data["samples"].append(sample)
        
        max_samples = 10000
        if len(self.collected_data["samples"]) > max_samples:
            self.collected_data["samples"] = self.collected_data["samples"][-max_samples:]
        
        self.save_collected_data()
    
    def should_retrain(self):
        samples_count = len(self.collected_data.get("samples", []))
        
        if samples_count < self.config["min_samples_for_retrain"]:
            return False, f"Not enough samples ({samples_count}/{self.config['min_samples_for_retrain']})"
        
        if self.config["last_retrain"]:
            last_retrain = datetime.fromisoformat(self.config["last_retrain"])
            hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
            
            if hours_since < self.config["retrain_interval_hours"]:
                return False, f"Too soon since last retrain ({hours_since:.1f}h/{self.config['retrain_interval_hours']}h)"
        
        return True, "Ready for retraining"
    
    def prepare_training_data(self, original_csv_path=None):
        if original_csv_path is None:
            original_csv_path = os.path.join(PROJECT_ROOT, "datasets", "merged_data_datetime_broken.csv")
        
        try:
            original_cwd = os.getcwd()
            train_model_dir = os.path.join(PROJECT_ROOT, "train_model")
            if os.path.exists(train_model_dir):
                os.chdir(train_model_dir)
            
            if not os.path.isabs(original_csv_path):
                original_csv_path = os.path.join(PROJECT_ROOT, original_csv_path)
            
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = load_data(original_csv_path)
            
            os.chdir(original_cwd)
            print(f"Original dataset: {len(X_train_orig)} training, {len(X_test_orig)} test samples")
        except Exception as e:
            print(f"Warning: Could not load original dataset: {e}")
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = None, None, None, None
        
        samples = self.collected_data.get("samples", [])
        if len(samples) == 0:
            print("No collected samples available")
            return X_train_orig, X_test_orig, y_train_orig, y_test_orig
        
        print(f"Collected samples: {len(samples)}")
        
        collected_features = []
        collected_labels = []
        
        for sample in samples:
            sensor = sample["sensor_data"]
            predictions = sample["predictions"]
            
            if sample.get("user_feedback") is not None:
                label = sample["user_feedback"]
            else:
                rf_pred = predictions.get("rf_prediction", 0)
                lr_pred = predictions.get("lr_prediction", 0)
                label = int(np.round((rf_pred + lr_pred) / 2))
            
            try:
                features = [
                    float(sensor.get('accel_x', 0)),
                    float(sensor.get('accel_y', 0)),
                    float(sensor.get('accel_z', 0)),
                    float(sensor.get('eda', 6.77)),  # Default EDA
                    float(sensor.get('bpm', 70)),
                    float(sensor.get('temperature', 25)),
                    int(sensor.get('datetime_year', datetime.now().year)),
                    int(sensor.get('datetime_month', datetime.now().month)),
                    int(sensor.get('datetime_day', datetime.now().day)),
                    int(sensor.get('datetime_hour', datetime.now().hour)),
                    int(sensor.get('datetime_dow', datetime.now().weekday()))
                ]
                collected_features.append(features)
                collected_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping sample due to error: {e}")
                continue
        
        if len(collected_features) == 0:
            print("No valid collected samples")
            return X_train_orig, X_test_orig, y_train_orig, y_test_orig
        
        feature_names = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 
                        'datetime_year', 'datetime_month', 'datetime_day', 
                        'datetime_hour', 'datetime_dow']
        X_collected = pd.DataFrame(collected_features, columns=feature_names)
        y_collected = pd.Series(collected_labels)
        
        print(f"Valid collected samples: {len(X_collected)}")
        
        if X_train_orig is not None:
            X_combined = pd.concat([X_train_orig, X_collected], ignore_index=True)
            y_combined = pd.concat([y_train_orig, y_collected], ignore_index=True)
            
            from sklearn.utils import shuffle
            X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
            
            from sklearn.model_selection import train_test_split
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42
            )
            
            print(f"Combined dataset: {len(X_train_new)} training, {len(X_test_new)} test samples")
            return X_train_new, X_test_new, y_train_new, y_test_new
        else:
            from sklearn.model_selection import train_test_split
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                X_collected, y_collected,                 test_size=0.2, stratify=y_collected, random_state=42
            )
            return X_train_new, X_test_new, y_train_new, y_test_new
    
    def evaluate_model_performance(self):
        try:
            metrics_file = os.path.join(METRICS_DIR, 'latest_metrics.json')
            if not os.path.exists(metrics_file):
                return None
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if 'best_model' in metrics and 'best_accuracy' in metrics:
                return metrics['best_accuracy']
            
            return None
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None
    
    def retrain(self):
        print("\n" + "="*60)
        print("AUTO-RETRAINING SYSTEM")
        print("="*60)
        
        should_retrain, reason = self.should_retrain()
        if not should_retrain:
            print(f"\n⏸ Retraining not needed: {reason}")
            return False
        
        print(f"\n✓ Retraining conditions met: {reason}")
        
        current_acc = self.evaluate_model_performance()
        if current_acc:
            print(f"Current model accuracy: {current_acc:.4f}")
        
        print("\nPreparing training data...")
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        if X_train is None or len(X_train) == 0:
            print("✗ No training data available")
            return False
        
        print("\nTraining models...")
        models, metrics = train_individual_models(X_train, y_train, X_test, y_test)
        
        if len(models) == 0:
            print("✗ Model training failed")
            return False
        
        best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['accuracy'])
        best_accuracy = metrics[best_model_name]['accuracy']
        
        improvement = best_accuracy - (current_acc if current_acc else 0)
        
        if self.config["keep_best_model"] and current_acc and improvement < self.config["min_accuracy_improvement"]:
            print(f"\n⏸ New best model accuracy ({best_accuracy:.4f}) doesn't improve enough (+{improvement:.4f})")
            print(f"   Keeping current model (threshold: +{self.config['min_accuracy_improvement']:.4f})")
            return False
        
        print("\nSaving new models...")
        save_models_and_metrics(models, metrics)
        
        self.config["last_retrain"] = datetime.now().isoformat()
        self.config["retrain_count"] = self.config.get("retrain_count", 0) + 1
        self.save_config()
        
        print("\n" + "="*60)
        print("RETRAINING COMPLETE")
        print("="*60)
        if current_acc:
            print(f"Previous best accuracy: {current_acc:.4f}")
        print(f"New best model: {best_model_name}")
        print(f"New best accuracy: {best_accuracy:.4f}")
        if current_acc:
            print(f"Improvement: {improvement:+.4f}")
        print(f"Retrain count: {self.config['retrain_count']}")
        
        return True

def main():
    retrainer = AutoRetrainer()
    retrainer.retrain()

if __name__ == "__main__":
    main()

