import streamlit as st
import json
import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title="Stress Detection Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'hr_readings' not in st.session_state:
    st.session_state.hr_readings = []
if 'temperature_readings' not in st.session_state:
    st.session_state.temperature_readings = []
if 'accel_x_readings' not in st.session_state:
    st.session_state.accel_x_readings = []
if 'accel_y_readings' not in st.session_state:
    st.session_state.accel_y_readings = []
if 'accel_z_readings' not in st.session_state:
    st.session_state.accel_z_readings = []
if 'simulated_data_index' not in st.session_state:
    st.session_state.simulated_data_index = 0
if 'simulated_data_df' not in st.session_state:
    st.session_state.simulated_data_df = None

MAX_READINGS = 50

MODEL_PATHS = {
    "Random Forest": "saved_models/random_forest.joblib",
    "Logistic Regression": "saved_models/logistic_regression.joblib"
}

@st.cache_resource
def load_model(model_name):
    try:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            return None
        
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None

@st.cache_data
def load_eda_from_dataset():
    try:
        dataset_path = os.path.join("datasets", "merged_data_datetime_broken.csv")
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if 'EDA' in df.columns and len(df) > 0:
                eda_value = float(df['EDA'].iloc[0])
                return eda_value
        return 6.77
    except Exception as e:
        return 6.77

def prepare_features(sensor_data):
    eda_value = load_eda_from_dataset()
    
    features_dict = {
        'X': [float(sensor_data['accel_x'])],
        'Y': [float(sensor_data['accel_y'])],
        'Z': [float(sensor_data['accel_z'])],
        'EDA': [float(eda_value)],
        'HR': [float(sensor_data['bpm'])],
        'TEMP': [float(sensor_data['temperature'])],
        'datetime_year': [int(sensor_data['datetime_year'])],
        'datetime_month': [int(sensor_data['datetime_month'])],
        'datetime_day': [int(sensor_data['datetime_day'])],
        'datetime_hour': [int(sensor_data['datetime_hour'])],
        'datetime_dow': [int(sensor_data['datetime_dow'])]
    }
    
    features_df = pd.DataFrame(features_dict)
    return features_df

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
        st.error(f"Prediction error: {e}")
        return None, None

@st.cache_data
def load_simulated_dataset():
    try:
        csv_path = os.path.join("datasets", "balanced_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        return None
    except Exception as e:
        return None

def parse_datetime_for_simulated(dt_str):
    try:
        if pd.isna(dt_str):
            now = datetime.now()
            return {
                'year': now.year,
                'month': now.month,
                'day': now.day,
                'hour': now.hour,
                'dow': now.weekday(),
                'timestamp': now.isoformat() + 'Z'
            }
        
        dt_str = str(dt_str).strip()
        
        if ':' in dt_str and len(dt_str.split(':')) == 2:
            parts = dt_str.split(':')
            hour = int(float(parts[0]))
            minute_sec = parts[1].split('.')[0]
            minute = int(float(minute_sec)) if minute_sec else 0
            second = int(float(parts[1].split('.')[1])) if '.' in parts[1] and len(parts[1].split('.')) > 1 else 0
            
            now = datetime.now()
            dt = datetime(now.year, now.month, now.day, hour, minute, second)
            
            return {
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'dow': dt.weekday(),
                'timestamp': dt.isoformat() + 'Z'
            }
        
        dt = pd.to_datetime(dt_str)
        return {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'dow': dt.weekday(),
            'timestamp': dt.isoformat() + 'Z'
        }
    except Exception as e:
        now = datetime.now()
        return {
            'year': now.year,
            'month': now.month,
            'day': now.day,
            'hour': now.hour,
            'dow': now.weekday(),
            'timestamp': now.isoformat() + 'Z'
        }

def generate_simulated_data():
    if st.session_state.simulated_data_df is None:
        st.session_state.simulated_data_df = load_simulated_dataset()
    
    if st.session_state.simulated_data_df is None or len(st.session_state.simulated_data_df) == 0:
        return None
    
    if st.session_state.simulated_data_index >= len(st.session_state.simulated_data_df):
        st.session_state.simulated_data_index = 0
    
    row = st.session_state.simulated_data_df.iloc[st.session_state.simulated_data_index]
    dt_info = parse_datetime_for_simulated(row.get('datetime'))
    
    sensor_data = {
        'bpm': float(row['HR']) if pd.notna(row['HR']) else 0.0,
        'temperature': float(row['TEMP']) if pd.notna(row['TEMP']) else 0.0,
        'accel_x': float(row['X']) / 100.0 if pd.notna(row['X']) else 0.0,
        'accel_y': float(row['Y']) / 100.0 if pd.notna(row['Y']) else 0.0,
        'accel_z': float(row['Z']) / 100.0 if pd.notna(row['Z']) else 0.0,
        'datetime_year': dt_info['year'],
        'datetime_month': dt_info['month'],
        'datetime_day': dt_info['day'],
        'datetime_hour': dt_info['hour'],
        'datetime_dow': dt_info['dow'],
        'timestamp': dt_info['timestamp'],
        'last_updated': datetime.now().isoformat()
    }
    
    st.session_state.simulated_data_index = (st.session_state.simulated_data_index + 1) % len(st.session_state.simulated_data_df)
    
    return sensor_data

def fetch_from_thingspeak():
    """Fetch latest sensor data directly from ThingSpeak API"""
    CHANNEL_ID = "3160510"
    READ_API_KEY = "Z7ZCVHQMMHBUUD2L"
    FETCH_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"
    
    try:
        response = requests.get(FETCH_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        bpm = data.get("field1")
        temperature = data.get("field2")
        accel_x = data.get("field3")
        accel_y = data.get("field4")
        accel_z = data.get("field5")
        timestamp = data.get("created_at")

        datetime_year = None
        datetime_month = None
        datetime_day = None
        datetime_hour = None
        datetime_dow = None
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                datetime_year = dt.year
                datetime_month = dt.month
                datetime_day = dt.day
                datetime_hour = dt.hour
                datetime_dow = dt.weekday()
            except Exception as e:
                st.warning(f"⚠️ Error parsing timestamp: {e}")

        if all([bpm, temperature, accel_x, accel_y, accel_z]):
            sensor_data = {
                'bpm': float(bpm) if bpm else None,
                'temperature': float(temperature) if temperature else None,
                'accel_x': float(accel_x) if accel_x else None,
                'accel_y': float(accel_y) if accel_y else None,
                'accel_z': float(accel_z) if accel_z else None,
                'datetime_year': datetime_year,
                'datetime_month': datetime_month,
                'datetime_day': datetime_day,
                'datetime_hour': datetime_hour,
                'datetime_dow': datetime_dow,
                'timestamp': timestamp,
                'last_updated': datetime.now().isoformat()
            }
            return sensor_data
        else:
            return None

    except Exception as e:
        st.error(f"❌ Error fetching data from ThingSpeak: {e}")
        return None

def load_sensor_data(data_source='real'):
    if data_source == 'simulated':
        return generate_simulated_data()
    else:
        # Fetch directly from ThingSpeak API
        return fetch_from_thingspeak()

def add_sensor_reading(sensor_data, reading_list, value_key, timestamp_key='last_updated'):
    if sensor_data and value_key in sensor_data and timestamp_key in sensor_data:
        reading = {
            'value': sensor_data[value_key],
            'timestamp': sensor_data[timestamp_key]
        }
        
        if len(reading_list) == 0 or reading_list[-1]['timestamp'] != reading['timestamp']:
            reading_list.append(reading)
            
            if len(reading_list) > MAX_READINGS:
                reading_list[:] = reading_list[-MAX_READINGS:]

st.title("Stress Detection Dashboard")

col_main_toggle, col_switch = st.columns([3, 1])

with col_main_toggle:
    toggle_state = st.toggle("", value=False, label_visibility="hidden", key="data_source_toggle")
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin: 20px 0;">
        <span style="font-size: 16px; font-weight: 500; color: #2c3e50;">Data Source:</span>
        <span style="font-size: 14px; font-weight: 600; color: #2c3e50; min-width: 120px;">{"Simulated data" if toggle_state else "Real data"}</span>
    </div>
    """, unsafe_allow_html=True)

with col_switch:
    if toggle_state:
        reading_active = st.toggle("Start Reading", value=False, key="simulated_reading_toggle")
        if reading_active:
            st.success("Simulated data reading ON")
        else:
            st.info("Simulated data reading OFF")
    else:
        reading_active = st.toggle("Start Reading", value=False, key="real_reading_toggle")
        if reading_active:
            st.success("Real data reading ON")
        else:
            st.info("Real data reading OFF")

data_source_key = 'simulated' if toggle_state else 'real'
data_active = reading_active

st.markdown("---")

st.header("Stress Detection")

if not data_active:
    sensor_data_for_prediction = None
else:
    sensor_data_for_prediction = load_sensor_data(data_source_key)

rf_model = load_model("Random Forest")
lr_model = load_model("Logistic Regression")

if rf_model is None:
    st.error("Failed to load Random Forest model. Check that saved_models/random_forest.joblib exists and is compatible with scikit-learn 1.3.2")
if lr_model is None:
    st.error("Failed to load Logistic Regression model. Check that saved_models/logistic_regression.joblib exists and is compatible with scikit-learn 1.3.2")

col_rf, col_lr = st.columns(2)

with col_rf:
    if not data_active:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
            <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Random Forest</h3>
            <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                <p>Enable reading to see predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif rf_model and sensor_data_for_prediction:
        try:
            features = prepare_features(sensor_data_for_prediction)
            rf_prediction, rf_probability = predict_stress(rf_model, features)
        except Exception as e:
            st.error(f"Error preparing features for Random Forest: {e}")
            rf_prediction, rf_probability = None, None
        
        if rf_prediction is not None:
            rf_prediction_int = int(rf_prediction)
            if rf_prediction_int == 0:
                rf_status = "Normal"
                rf_tile_color = "#2ed573"
                rf_text_color = "#ffffff"
            elif rf_prediction_int == 1:
                rf_status = "Mildly Stressed"
                rf_tile_color = "#ffd700"
                rf_text_color = "#000000"
            else:
                rf_status = "Highly Stressed"
                rf_tile_color = "#ff4757"
                rf_text_color = "#ffffff"
            
            st.markdown(f"""
            <div style="background-color: {rf_tile_color}; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
                <h3 style="text-align: center; color: {rf_text_color}; margin-bottom: 1rem;">Random Forest</h3>
                <div style="text-align: center; font-size: 2rem; color: {rf_text_color}; font-weight: 700; margin: 0.5rem 0;">
                    {rf_prediction_int}
                </div>
                <div style="text-align: center; font-size: 1.5rem; color: {rf_text_color}; font-weight: 600; margin: 1rem 0;">
                    {rf_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
                <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Random Forest</h3>
                <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                    <p>Prediction error</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
            <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Random Forest</h3>
            <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                <p>Loading model...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_lr:
    if not data_active:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
            <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Logistic Regression</h3>
            <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                <p>Enable reading to see predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lr_model and sensor_data_for_prediction:
        try:
            features = prepare_features(sensor_data_for_prediction)
            lr_prediction, lr_probability = predict_stress(lr_model, features)
        except Exception as e:
            st.error(f"Error preparing features for Logistic Regression: {e}")
            lr_prediction, lr_probability = None, None
        
        if lr_prediction is not None:
            lr_prediction_int = int(lr_prediction)
            if lr_prediction_int == 0:
                lr_status = "Normal"
                lr_tile_color = "#2ed573"
                lr_text_color = "#ffffff"
            elif lr_prediction_int == 1:
                lr_status = "Mildly Stressed"
                lr_tile_color = "#ffd700"
                lr_text_color = "#000000"
            else:
                lr_status = "Highly Stressed"
                lr_tile_color = "#ff4757"
                lr_text_color = "#ffffff"
            
            st.markdown(f"""
            <div style="background-color: {lr_tile_color}; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
                <h3 style="text-align: center; color: {lr_text_color}; margin-bottom: 1rem;">Logistic Regression</h3>
                <div style="text-align: center; font-size: 2rem; color: {lr_text_color}; font-weight: 700; margin: 0.5rem 0;">
                    {lr_prediction_int}
                </div>
                <div style="text-align: center; font-size: 1.5rem; color: {lr_text_color}; font-weight: 600; margin: 1rem 0;">
                    {lr_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
                <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Logistic Regression</h3>
                <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                    <p>Prediction error</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;">
            <h3 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Logistic Regression</h3>
            <div style="text-align: center; font-size: 1.2rem; color: #2c3e50;">
                <p>Loading model...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.header("Sensor readings")

if not data_active:
    sensor_data = None
else:
    sensor_data = load_sensor_data(data_source_key)

if sensor_data:
    add_sensor_reading(sensor_data, st.session_state.hr_readings, 'bpm')
    add_sensor_reading(sensor_data, st.session_state.temperature_readings, 'temperature')
    add_sensor_reading(sensor_data, st.session_state.accel_x_readings, 'accel_x')
    add_sensor_reading(sensor_data, st.session_state.accel_y_readings, 'accel_y')
    add_sensor_reading(sensor_data, st.session_state.accel_z_readings, 'accel_z')
    
    sensor_configs = [
        {'key': 'hr_readings', 'name': 'Heart Rate (HR)', 'unit': 'BPM', 'color': '#1f77b4'},
        {'key': 'temperature_readings', 'name': 'Temperature', 'unit': '°C', 'color': '#ff7f0e'},
        {'key': 'accel_x_readings', 'name': 'Accelerometer X', 'unit': 'g', 'color': '#2ca02c'},
        {'key': 'accel_y_readings', 'name': 'Accelerometer Y', 'unit': 'g', 'color': '#d62728'},
        {'key': 'accel_z_readings', 'name': 'Accelerometer Z', 'unit': 'g', 'color': '#9467bd'}
    ]
    
    for config in sensor_configs:
        readings_list = st.session_state[config['key']]
        
        if readings_list:
            st.subheader(f"{config['name']} Readings")
            st.write(f"Total readings: {len(readings_list)}/{MAX_READINGS}")
            
            total_readings = len(readings_list)
            display_rows = 4
            
            readings_display = []
            for reading in readings_list:
                readings_display.append({
                    config['name']: reading['value'],
                    'Timestamp': reading['timestamp']
                })
            
            if total_readings > display_rows:
                readings_to_display = readings_display[-display_rows:]
            else:
                readings_to_display = readings_display
            
            st.dataframe(readings_to_display, use_container_width=True, hide_index=True)
            
            if len(readings_list) > 0:
                timestamps = []
                values = []
                
                for reading in readings_list:
                    try:
                        timestamp_dt = datetime.fromisoformat(reading['timestamp'])
                        timestamps.append(timestamp_dt)
                        values.append(reading['value'])
                    except:
                        timestamps.append(reading['timestamp'])
                        values.append(reading['value'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=config['name'],
                    line=dict(color=config['color'], width=2),
                    marker=dict(size=6, color=config['color'])
                ))
                
                fig.update_layout(
                    title=f"{config['name']} Over Time",
                    xaxis_title='Timestamp',
                    yaxis_title=f"{config['name']} ({config['unit']})",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        if data_source_key == 'simulated':
            st.warning("Simulated Data")
        else:
            st.warning("Data from ThingspeakAPI.")

st.markdown("---")

st.header("Stress Log")

if 'stress_log' not in st.session_state:
    st.session_state.stress_log = []

if not data_active:
    sensor_data_for_stress_log = None
else:
    sensor_data_for_stress_log = load_sensor_data(data_source_key)

if sensor_data_for_stress_log and rf_model and lr_model:
    features = prepare_features(sensor_data_for_stress_log)
    rf_prediction, rf_probability = predict_stress(rf_model, features)
    lr_prediction, lr_probability = predict_stress(lr_model, features)
    
    if rf_prediction is not None and lr_prediction is not None:
        rf_prediction_int = int(rf_prediction)
        lr_prediction_int = int(lr_prediction)
        
        if rf_prediction_int > 0 or lr_prediction_int > 0:
            current_timestamp = sensor_data_for_stress_log.get('last_updated', datetime.now().isoformat())
            
            rf_status_label = "Normal" if rf_prediction_int == 0 else ("Mildly Stressed" if rf_prediction_int == 1 else "Highly Stressed")
            lr_status_label = "Normal" if lr_prediction_int == 0 else ("Mildly Stressed" if lr_prediction_int == 1 else "Highly Stressed")
            
            log_entry = {
                'timestamp': current_timestamp,
                'rf_prediction': rf_prediction_int,
                'rf_status': rf_status_label,
                'rf_confidence': rf_probability * 100 if rf_probability is not None else 0.0,
                'lr_prediction': lr_prediction_int,
                'lr_status': lr_status_label,
                'lr_confidence': lr_probability * 100 if lr_probability is not None else 0.0
            }
            
            if len(st.session_state.stress_log) == 0 or st.session_state.stress_log[-1]['timestamp'] != log_entry['timestamp']:
                st.session_state.stress_log.append(log_entry)
                
                if len(st.session_state.stress_log) > MAX_READINGS:
                    st.session_state.stress_log[:] = st.session_state.stress_log[-MAX_READINGS:]

if st.session_state.stress_log:
    log_display = []
    for entry in st.session_state.stress_log:
        log_display.append({
            'Timestamp': entry['timestamp'],
            'RF Prediction': entry.get('rf_prediction', 'N/A'),
            'RF Status': entry['rf_status'],
            'LR Prediction': entry.get('lr_prediction', 'N/A'),
            'LR Status': entry['lr_status']
        })
    
    st.dataframe(log_display, use_container_width=True, hide_index=True)
else:
    st.info("No stress detected yet. The log will record entries when stress is detected.")

time.sleep(3)
st.rerun()
