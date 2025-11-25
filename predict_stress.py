import joblib
import pandas as pd
import os
from datetime import datetime

MODEL_PATHS = {
    "Random Forest": "saved_models/random_forest.joblib",
    "Logistic Regression": "saved_models/logistic_regression.joblib"
}

def parse_datetime(dt_str):
    try:
        dt = pd.to_datetime(dt_str)
        return {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'dow': dt.weekday()
        }
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        now = datetime.now()
        return {
            'year': now.year,
            'month': now.month,
            'day': now.day,
            'hour': now.hour,
            'dow': now.weekday()
        }

def prepare_features(X, Y, Z, EDA, HR, TEMP, datetime_str):
    dt_info = parse_datetime(datetime_str)
    
    features_dict = {
        'X': [float(X)],
        'Y': [float(Y)],
        'Z': [float(Z)],
        'EDA': [float(EDA)],
        'HR': [float(HR)],
        'TEMP': [float(TEMP)],
        'datetime_year': [int(dt_info['year'])],
        'datetime_month': [int(dt_info['month'])],
        'datetime_day': [int(dt_info['day'])],
        'datetime_hour': [int(dt_info['hour'])],
        'datetime_dow': [int(dt_info['dow'])]
    }
    
    features_df = pd.DataFrame(features_dict)
    return features_df, dt_info

def predict_stress(model, features):
    try:
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0]
            stress_prob = probability[1] if len(probability) > 1 else probability[0]
        else:
            stress_prob = None
        
        return prediction, stress_prob
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def load_model(model_name):
    try:
        model_path = MODEL_PATHS.get(model_name)
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def main():
    X = -78.0
    Y = 15.0
    Z = -17.0
    EDA = 0.169362
    HR = 88.87
    TEMP = 29.79
    id_val = "5C"
    datetime_str = "2020-06-24 14:22:37.718749952"
    
    print("Input Values:")
    print(f"  X: {X}")
    print(f"  Y: {Y}")
    print(f"  Z: {Z}")
    print(f"  EDA: {EDA}")
    print(f"  HR: {HR}")
    print(f"  TEMP: {TEMP}")
    print(f"  ID: {id_val}")
    print(f"  Datetime: {datetime_str}")
    print("\n" + "="*50 + "\n")
    
    features_df, dt_info = prepare_features(X, Y, Z, EDA, HR, TEMP, datetime_str)
    
    print("Prepared Features:")
    print(features_df.to_string())
    print("\n" + "="*50 + "\n")
    
    rf_model = load_model("Random Forest")
    lr_model = load_model("Logistic Regression")
    
    if rf_model:
        rf_prediction, rf_probability = predict_stress(rf_model, features_df)
        if rf_prediction is not None:
            rf_status = "STRESS DETECTED" if rf_prediction == 1 else "NO STRESS"
            rf_conf = rf_probability * 100 if rf_probability is not None else 0.0
            print(f"Random Forest:")
            print(f"  Status: {rf_status}")
            print(f"  Confidence: {rf_conf:.2f}%")
        else:
            print("Random Forest: Prediction failed")
    else:
        print("Random Forest: Model not loaded")
    
    print("\n" + "-"*50 + "\n")
    
    if lr_model:
        lr_prediction, lr_probability = predict_stress(lr_model, features_df)
        if lr_prediction is not None:
            lr_status = "STRESS DETECTED" if lr_prediction == 1 else "NO STRESS"
            lr_conf = lr_probability * 100 if lr_probability is not None else 0.0
            print(f"Logistic Regression:")
            print(f"  Status: {lr_status}")
            print(f"  Confidence: {lr_conf:.2f}%")
        else:
            print("Logistic Regression: Prediction failed")
    else:
        print("Logistic Regression: Model not loaded")
    
    print("\n" + "="*50 + "\n")
    
    output_data = {
        'bpm': HR,
        'temperature': TEMP,
        'accel_x': X / 100.0,
        'accel_y': Y / 100.0,
        'accel_z': Z / 100.0,
        'datetime_year': dt_info['year'],
        'datetime_month': dt_info['month'],
        'datetime_day': dt_info['day'],
        'datetime_hour': dt_info['hour'],
        'datetime_dow': dt_info['dow']
    }
    
    print("Output Format (matching latest_sensor_data.json structure):")
    print(f"  bpm: {output_data['bpm']}")
    print(f"  temperature: {output_data['temperature']}")
    print(f"  accel_x: {output_data['accel_x']}")
    print(f"  accel_y: {output_data['accel_y']}")
    print(f"  accel_z: {output_data['accel_z']}")
    print(f"  datetime_year: {output_data['datetime_year']}")
    print(f"  datetime_month: {output_data['datetime_month']}")
    print(f"  datetime_day: {output_data['datetime_day']}")
    print(f"  datetime_hour: {output_data['datetime_hour']}")
    print(f"  datetime_dow: {output_data['datetime_dow']}")

if __name__ == "__main__":
    main()

