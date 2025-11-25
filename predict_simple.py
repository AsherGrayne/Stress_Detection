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
    return features_df

def predict(model, features):
    try:
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        return None

def load_model(model_name):
    try:
        model_path = MODEL_PATHS.get(model_name)
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception as e:
        return None

X = -78.0
Y = 15.0
Z = -17.0
EDA = 0.169362
HR = 88.87
TEMP = 29.79
datetime_str = "2020-06-24 14:22:37.718749952"

features_df = prepare_features(X, Y, Z, EDA, HR, TEMP, datetime_str)

rf_model = load_model("Random Forest")
lr_model = load_model("Logistic Regression")

rf_prediction = predict(rf_model, features_df) if rf_model else None
lr_prediction = predict(lr_model, features_df) if lr_model else None

print(rf_prediction)
print(lr_prediction)

