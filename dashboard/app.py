"""
Streamlit Dashboard for Stress Detection
Easiest to deploy: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Stress Detection Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONFIGURATION
# ============================================
THINGSPEAK_CHANNEL_ID = "3160510"
THINGSPEAK_READ_API_KEY = "Z7ZCVHQMMHBUUD2L"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_API_KEY}"

MODEL_PATHS = {
    "Random Forest": "saved_models/random_forest.joblib",
    "Logistic Regression": "saved_models/logistic_regression.joblib",
    "Gradient Boosting": "saved_models/gradient_boosting.joblib",
    "MLP Classifier": "saved_models/mlp_classifier.joblib"
}

# Feature names matching the model training
FEATURES = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP',
            'datetime_year', 'datetime_month', 'datetime_day', 'datetime_hour', 'datetime_dow']

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'X': 'Accelerometer X (float)',
    'Y': 'Accelerometer Y (float)',
    'Z': 'Accelerometer Z (float)',
    'EDA': 'Electrodermal Activity (float) - From datasets folder',
    'HR': 'Heart Rate in BPM (float)',
    'TEMP': 'Temperature in °C (float)',
    'datetime_year': 'Year (int)',
    'datetime_month': 'Month (int, 1-12)',
    'datetime_day': 'Day (int, 1-31)',
    'datetime_hour': 'Hour (int, 0-23)',
    'datetime_dow': 'Day of week (int, 0-6, Monday=0)'
}

# Initialize session state for historical data
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []

MAX_HISTORY = 100  # Maximum number of data points to keep

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_resource
def load_model(model_name):
    """Load ML model from joblib file."""
    try:
        model_path = MODEL_PATHS.get(model_name)
        if model_path:
            return joblib.load(model_path)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def fetch_local_data():
    """Fetch latest data from raspberry.py (reads from local JSON file)."""
    data_file = 'latest_sensor_data.json'
    
    try:
        # Check if file exists
        if not os.path.exists(data_file):
            return None
        
        # Read JSON file
        with open(data_file, 'r') as f:
            sensor_data = json.load(f)
        
        # Verify data is recent (within last 30 seconds)
        if 'last_updated' in sensor_data:
            try:
                last_update = datetime.fromisoformat(sensor_data['last_updated'])
                time_diff = (datetime.now() - last_update).total_seconds()
                if time_diff > 30:
                    st.warning(f"Data is {time_diff:.0f} seconds old. Is raspberry.py running?")
            except:
                pass
        
        return sensor_data
        
    except json.JSONDecodeError as e:
        st.error(f"Error reading data file (invalid JSON): {e}")
        return None
    except Exception as e:
        st.error(f"Error reading data file: {e}")
        return None

@st.cache_data
def load_eda_from_dataset():
    """
    Load EDA value from datasets folder.
    Returns a single EDA value from the merged_data_datetime_broken.csv file.
    """
    try:
        dataset_path = os.path.join("datasets", "merged_data_datetime_broken.csv")
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if 'EDA' in df.columns and len(df) > 0:
                # Return the first EDA value from the dataset
                eda_value = float(df['EDA'].iloc[0])
                return eda_value
        return 6.77  # Default fallback value based on dataset sample
    except Exception as e:
        st.warning(f"Could not load EDA from dataset: {e}. Using default value.")
        return 6.77

def prepare_features(sensor_data):
    """
    Prepare feature vector for model prediction using exact attribute names.
    Features must match training data: X, Y, Z, EDA, HR, TEMP, datetime_*
    All values taken from latest_sensor_data.json except EDA which comes from datasets folder.
    Returns a DataFrame with column names matching the training data.
    """
    # Load EDA value from datasets folder
    eda_value = load_eda_from_dataset()
    
    # Prepare features as DataFrame with column names matching training data
    # Parameter mapping:
    # X - Accelerometer X (float) from accel_x
    # Y - Accelerometer Y (float) from accel_y
    # Z - Accelerometer Z (float) from accel_z
    # EDA - Electrodermal Activity (float) from datasets folder
    # HR - Heart Rate in BPM (float) from bpm
    # TEMP - Temperature in °C (float) from temperature
    # datetime_year - Year (int)
    # datetime_month - Month (int, 1-12)
    # datetime_day - Day (int, 1-31)
    # datetime_hour - Hour (int, 0-23)
    # datetime_dow - Day of week (int, 0-6, Monday=0)
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
    
    return features_df, eda_value

def predict_stress(model, features):
    """Make stress prediction using loaded model."""
    try:
        prediction = model.predict(features)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0]
            stress_prob = probability[1] if len(probability) > 1 else probability[0]
        else:
            stress_prob = None
        
        return prediction, stress_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def add_to_history(sensor_data, prediction, probability, eda_value):
    """Add current data point to historical data."""
    history_entry = {
        'timestamp': datetime.now(),
        'bpm': sensor_data['bpm'],
        'temperature': sensor_data['temperature'],
        'accel_x': sensor_data['accel_x'],
        'accel_y': sensor_data['accel_y'],
        'accel_z': sensor_data['accel_z'],
        'eda': eda_value,
        'prediction': prediction,
        'probability': probability if probability is not None else 0.0,
        'stress_status': 'STRESS' if prediction == 1 else 'NO STRESS'
    }
    
    st.session_state.historical_data.append(history_entry)
    
    # Keep only last MAX_HISTORY entries
    if len(st.session_state.historical_data) > MAX_HISTORY:
        st.session_state.historical_data = st.session_state.historical_data[-MAX_HISTORY:]

def create_time_series_graphs(historical_data):
    """Create time-series graphs for all input values."""
    if not historical_data:
        return None
    
    df = pd.DataFrame(historical_data)
    
    # Create subplots for all sensors (4 rows x 2 cols to include EDA)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Heart Rate (BPM) Over Time',
            'Temperature (°C) Over Time',
            'Accelerometer X Over Time',
            'Accelerometer Y Over Time',
            'Accelerometer Z Over Time',
            'EDA Over Time',
            'Stress Probability Over Time',
            'Stress Status Over Time'
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.1
    )
    
    # Heart Rate
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bpm'],
            mode='lines+markers',
            name='BPM',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    # Accelerometer X
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['accel_x'],
            mode='lines+markers',
            name='Accel X',
            line=dict(color='#95E1D3', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # Accelerometer Y
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['accel_y'],
            mode='lines+markers',
            name='Accel Y',
            line=dict(color='#F38181', width=2),
            marker=dict(size=4)
        ),
        row=2, col=2
    )
    
    # Accelerometer Z
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['accel_z'],
            mode='lines+markers',
            name='Accel Z',
            line=dict(color='#AA96DA', width=2),
            marker=dict(size=4)
        ),
        row=3, col=1
    )
    
    # EDA (From datasets folder)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['eda'],
            mode='lines+markers',
            name='EDA',
            line=dict(color='#FFA07A', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(255, 160, 122, 0.2)'
        ),
        row=3, col=2
    )
    
    # Stress Probability over time
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['probability'] * 100,
            mode='lines+markers',
            name='Stress Probability',
            line=dict(color='#FF6B9D', width=3),
            marker=dict(size=6, color=df['probability'] * 100, 
                       colorscale='RdYlGn', showscale=True, 
                       cmin=0, cmax=100),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 157, 0.2)',
            text=[f"Status: {s}<br>Probability: {p*100:.1f}%" 
                  for s, p in zip(df['stress_status'], df['probability'])],
            hovertemplate='%{text}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Add threshold line at 50%
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="50% Threshold", row=4, col=1)
    
    # Stress Status (bar chart)
    stress_colors = ['#4CAF50' if s == 'NO STRESS' else '#F44336' for s in df['stress_status']]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=[1 if s == 'STRESS' else 0 for s in df['stress_status']],
            name='Stress Status',
            marker_color=stress_colors,
            text=[s for s in df['stress_status']],
            textposition='outside'
        ),
        row=4, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=2)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=2)
    
    fig.update_yaxes(title_text="BPM", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=2)
    fig.update_yaxes(title_text="Accel X", row=2, col=1)
    fig.update_yaxes(title_text="Accel Y", row=2, col=2)
    fig.update_yaxes(title_text="Accel Z", row=3, col=1)
    fig.update_yaxes(title_text="EDA (μS)", row=3, col=2)
    fig.update_yaxes(title_text="Stress Probability (%)", row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Stress Status", row=4, col=2, range=[-0.1, 1.1])
    
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text="Sensor Data Time Series - All Input Values",
        title_x=0.5
    )
    
    return fig

# ============================================
# MAIN DASHBOARD
# ============================================

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
    }
    .model-card-rf {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .model-card-lr {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Stress Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">Real-time stress detection using machine learning models</p>', unsafe_allow_html=True)

# Data source indicator
st.info("**Data Source:** Reading from `raspberry.py` (local JSON file). Make sure `raspberry.py` is running!")

# Sidebar
st.sidebar.header("Configuration")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (15s)", value=False)
refresh_interval = 15 if auto_refresh else 0

# Clear history button
if st.sidebar.button("Clear History"):
    st.session_state.historical_data = []
    st.rerun()

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    st.rerun()

# Show history count
st.sidebar.info(f"Data Points: {len(st.session_state.historical_data)}/{MAX_HISTORY}")

# Load both models for Stress Detection section
rf_model = load_model("Random Forest")
lr_model = load_model("Logistic Regression")

if rf_model is None or lr_model is None:
    st.error("Could not load models. Please check model files.")
    st.stop()

st.sidebar.success("Models loaded successfully")

# Fetch data from raspberry.py
sensor_data = fetch_local_data()

if sensor_data:
    # Prepare features
    features, eda_value = prepare_features(sensor_data)
    
    # ============================================
    # SECTION 1: STRESS DETECTION
    # ============================================
    st.markdown('<h2 class="section-header">Stress Detection</h2>', unsafe_allow_html=True)
    
    # Make predictions with both models
    rf_prediction, rf_probability = predict_stress(rf_model, features)
    lr_prediction, lr_probability = predict_stress(lr_model, features)
    
    # Display both models side by side
    col_rf, col_lr = st.columns(2)
    
    with col_rf:
        # Random Forest Card
        rf_status = "STRESS DETECTED" if rf_prediction == 1 else "NO STRESS"
        rf_status_color = "#ff4757" if rf_prediction == 1 else "#2ed573"
        rf_conf = rf_probability * 100 if rf_probability is not None else 0.0
        
        st.markdown(f"""
        <div class="model-card model-card-rf">
            <h3 style="margin-top: 0; font-size: 1.5rem; font-weight: 600;">Random Forest</h3>
            <div class="prediction-text" style="color: {rf_status_color};">
                {rf_status}
            </div>
            <div class="confidence-text" style="color: white;">
                {rf_conf:.1f}%
            </div>
            <p style="margin-bottom: 0; opacity: 0.9;">Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_lr:
        # Logistic Regression Card
        lr_status = "STRESS DETECTED" if lr_prediction == 1 else "NO STRESS"
        lr_status_color = "#ff4757" if lr_prediction == 1 else "#2ed573"
        lr_conf = lr_probability * 100 if lr_probability is not None else 0.0
        
        st.markdown(f"""
        <div class="model-card model-card-lr">
            <h3 style="margin-top: 0; font-size: 1.5rem; font-weight: 600;">Logistic Regression</h3>
            <div class="prediction-text" style="color: {lr_status_color};">
                {lr_status}
            </div>
            <div class="confidence-text" style="color: white;">
                {lr_conf:.1f}%
            </div>
            <p style="margin-bottom: 0; opacity: 0.9;">Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use Random Forest as primary model for historical data
    primary_prediction = rf_prediction
    primary_probability = rf_probability
    
    # Add to historical data (using Random Forest as primary)
    add_to_history(sensor_data, primary_prediction, primary_probability, eda_value)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # SECTION 2: SENSOR READINGS
    # ============================================
    st.markdown('<h2 class="section-header">Sensor Readings</h2>', unsafe_allow_html=True)
    
    # Define sensor parameters with their display names and units
    sensor_params = [
        {'key': 'bpm', 'name': 'Heart Rate (HR)', 'unit': 'BPM', 'color': '#FF6B6B'},
        {'key': 'temperature', 'name': 'Temperature (TEMP)', 'unit': '°C', 'color': '#4ECDC4'},
        {'key': 'accel_x', 'name': 'Accelerometer X', 'unit': 'g', 'color': '#95E1D3'},
        {'key': 'accel_y', 'name': 'Accelerometer Y', 'unit': 'g', 'color': '#F38181'},
        {'key': 'accel_z', 'name': 'Accelerometer Z', 'unit': 'g', 'color': '#AA96DA'},
        {'key': 'eda', 'name': 'Electrodermal Activity (EDA)', 'unit': 'μS', 'color': '#FFA07A', 'simulated': True}
    ]
    
    # Create individual graphs for each sensor parameter
    for sensor in sensor_params:
        st.markdown(f"### {sensor['name']}")
        
        # Get current value directly from latest_sensor_data.json
        if sensor['key'] == 'eda':
            current_value = eda_value
        else:
            # Ensure we get the exact value from JSON as float
            raw_value = sensor_data.get(sensor['key'])
            if raw_value is not None:
                current_value = float(raw_value)
            else:
                current_value = 0.0
        
        # Create two columns: graph on left, table on right
        graph_col, table_col = st.columns([2, 1])
        
        with graph_col:
            # Create line graph showing ONLY the current value from latest_sensor_data.json
            fig = go.Figure()
            
            # Get the exact value from sensor_data (latest_sensor_data.json)
            if sensor['key'] == 'eda':
                display_value = eda_value
            else:
                display_value = float(sensor_data.get(sensor['key'], 0.0))
            
            # Create line graph with current value from JSON file
            # Use last_updated as x-axis for proper line graph display
            last_updated_str = sensor_data.get('last_updated', datetime.now().isoformat())
            try:
                timestamp_dt = datetime.fromisoformat(last_updated_str)
            except:
                timestamp_dt = datetime.now()
            
            fig.add_trace(go.Scatter(
                x=[timestamp_dt],
                y=[display_value],
                mode='lines+markers',
                name=sensor['name'],
                line=dict(color=sensor['color'], width=3),
                marker=dict(size=10, color=sensor['color']),
                hovertemplate=f"<b>{sensor['name']}</b><br>Time: %{{x}}<br>Value: {display_value:.3f} {sensor['unit']}<extra></extra>"
            ))
            
            # Add horizontal reference line at zero for accelerometer values
            if 'accel' in sensor['key']:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f"{sensor['name']} - Current Value from latest_sensor_data.json",
                xaxis_title="Time",
                yaxis_title=f"Value ({sensor['unit']})",
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                xaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the exact value being shown
            st.caption(f"**Current Value from JSON:** {display_value:.6f} {sensor['unit']}")
            
            if sensor.get('simulated', False):
                st.caption(f"Note: {sensor['name']} value loaded from datasets folder")
        
        with table_col:
            # Create table with sensor data
            table_data = {
                'Parameter': [sensor['name']],
                'Value': [f"{current_value:.3f}"],
                'Unit': [sensor['unit']],
                'Source': ['From datasets' if sensor.get('simulated', False) else 'Real-time']
            }
            
            # Add timestamp information
            table_data['Timestamp'] = [sensor_data.get('timestamp', 'N/A')]
            table_data['Last Updated'] = [sensor_data.get('last_updated', 'N/A')]
            
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            
            # Additional info box
            st.info(f"""
            **Current Value:** {current_value:.3f} {sensor['unit']}
            
            **Status:** {'From datasets folder' if sensor.get('simulated', False) else 'Real-time'}
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary metrics row
    st.markdown("---")
    st.subheader("Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Heart Rate (BPM)",
            f"{sensor_data['bpm']:.1f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Temperature (°C)",
            f"{sensor_data['temperature']:.1f}",
            delta=None
        )
    
    with col3:
        # Average confidence from both models
        avg_conf = ((rf_conf if rf_probability is not None else 0) + 
                   (lr_conf if lr_probability is not None else 0)) / 2
        st.metric(
            "Average Confidence",
            f"{avg_conf:.1f}%",
            delta=None
        )
    
    with col4:
        # Agreement status
        if rf_prediction == lr_prediction:
            agreement = "Models Agree"
            agreement_color = "green"
        else:
            agreement = "Models Disagree"
            agreement_color = "orange"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: #f8f9fa; 
                    border-radius: 10px; border-left: 4px solid {agreement_color};">
            <p style="margin: 0; color: {agreement_color}; font-weight: 600;">{agreement}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Comparison Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Comparison table
        comparison_data = {
            'Model': ['Random Forest', 'Logistic Regression'],
            'Prediction': [
                'STRESS' if rf_prediction == 1 else 'NO STRESS',
                'STRESS' if lr_prediction == 1 else 'NO STRESS'
            ],
            'Confidence': [
                f"{rf_conf:.2f}%" if rf_probability is not None else "N/A",
                f"{lr_conf:.2f}%" if lr_probability is not None else "N/A"
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col_right:
        # Feature values with descriptions
        st.markdown("**Feature Values Used for Prediction:**")
        feature_data = []
        for feat_name in FEATURES:
            feature_data.append({
                'Feature': feat_name,
                'Value': f"{features[feat_name].iloc[0]:.3f}",
                'Type': 'From datasets' if feat_name == 'EDA' else 'Real-time'
            })
        
        feature_df = pd.DataFrame(feature_data)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        # Highlight EDA source
        st.info(f"**EDA (Electrodermal Activity)** value loaded from datasets folder. Current value: {eda_value:.2f} μS")
    
    # Timestamp Information
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Timestamp Information")
    timestamp_data = {
        'Field': ['Date', 'Time', 'Day of Week', 'Raw Timestamp', 'Last Updated'],
        'Value': [
            f"{sensor_data['datetime_year']}-{sensor_data['datetime_month']:02d}-{sensor_data['datetime_day']:02d}",
            f"{sensor_data['datetime_hour']:02d}:00",
            f"{sensor_data['datetime_dow']} (Mon=0, Sun=6)",
            sensor_data.get('timestamp', 'N/A'),
            sensor_data.get('last_updated', 'N/A')
        ]
    }
    timestamp_df = pd.DataFrame(timestamp_data)
    st.dataframe(timestamp_df, use_container_width=True, hide_index=True)
    
else:
    st.warning("Could not fetch data from raspberry.py. Please ensure:")
    st.markdown("""
    1. **raspberry.py is running** in another terminal/process
    2. The file `latest_sensor_data.json` exists in the project root
    3. raspberry.py has permission to write the JSON file
    
    **To start raspberry.py:**
    ```bash
    python raspberry.py
    ```
    """)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p><strong>Data Source:</strong> Dashboard reads from <code>latest_sensor_data.json</code> created by <code>raspberry.py</code></p>
    <p><strong>Note:</strong> EDA (Electrodermal Activity) value is loaded from datasets folder.</p>
</div>
""", unsafe_allow_html=True)

