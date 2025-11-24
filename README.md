# Stress Detection System

ML-powered stress detection system that analyzes real-time sensor data from Raspberry Pi and displays predictions via an interactive Streamlit dashboard.

## Overview

This project implements a real-time stress detection system using machine learning models (Random Forest and Logistic Regression) to analyze physiological and movement sensor data. The system fetches sensor data from ThingSpeak, processes it through trained ML models, and displays stress predictions and sensor readings through a professional web dashboard.

## Features

- **Real-time Stress Detection**: Uses Random Forest and Logistic Regression models to predict stress levels
- **Multi-Sensor Analysis**: Processes heart rate (BPM), temperature, accelerometer (X, Y, Z), and EDA data
- **Interactive Dashboard**: Professional Streamlit dashboard with:
  - Stress detection cards showing model predictions and confidence levels
  - Individual sensor graphs with time-series visualization
  - Tabular data display for each sensor parameter
  - Model comparison and feature analysis
- **ThingSpeak Integration**: Fetches sensor data from ThingSpeak IoT platform
- **Local Data Exchange**: Uses JSON file for inter-process communication between data fetcher and dashboard

## Project Structure

```
Stress_Detection/
├── dashboard/
│   ├── app.py                 # Main Streamlit dashboard
│   ├── gradio_app.py          # Alternative Gradio dashboard
│   ├── requirements.txt        # Python dependencies
│   ├── run_dashboard.bat      # Windows launcher
│   └── run_dashboard.sh       # Linux/Mac launcher
├── datasets/
│   ├── merged_data_datetime_broken.csv  # Training dataset
│   ├── merged_data.csv
│   └── balanced_data.csv
├── saved_models/
│   ├── random_forest.joblib
│   └── logistic_regression.joblib
├── train_model/
│   ├── random_forest.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── sgd.py
│   ├── xg_boost.py
│   └── light_gbm.py
├── raspberry.py               # ThingSpeak data fetcher
├── latest_sensor_data.json    # Local data exchange file
└── README.md
```

## Requirements

- Python 3.8+
- ThingSpeak account with channel configured
- Internet connection for ThingSpeak API access

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stress_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r dashboard/requirements.txt
   ```

3. **Configure ThingSpeak**
   - Update `CHANNEL_ID` and `READ_API_KEY` in `raspberry.py`
   - Ensure your ThingSpeak channel has fields configured:
     - Field 1: BPM (Heart Rate)
     - Field 2: Temperature
     - Field 3: Accelerometer X
     - Field 4: Accelerometer Y
     - Field 5: Accelerometer Z

## Usage

### Step 1: Start Data Fetcher

Run the `raspberry.py` script to fetch data from ThingSpeak and save it to `latest_sensor_data.json`:

```bash
python raspberry.py
```

This script will:
- Fetch sensor data from ThingSpeak every 15 seconds
- Parse datetime components (year, month, day, hour, day of week)
- Save data to `latest_sensor_data.json` for the dashboard to read

### Step 2: Start Dashboard

In a separate terminal, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Or use the provided launcher scripts:
- **Windows**: `dashboard\run_dashboard.bat`
- **Linux/Mac**: `bash dashboard/run_dashboard.sh`

The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Features

### 1. Stress Detection Section
- Displays predictions from both Random Forest and Logistic Regression models
- Shows "STRESS DETECTED" or "NO STRESS" status with color coding
- Displays confidence percentage for each model

### 2. Sensor Readings Section
- Individual line graphs for each sensor parameter:
  - Heart Rate (HR/BPM)
  - Temperature (TEMP)
  - Accelerometer X, Y, Z
  - Electrodermal Activity (EDA) - loaded from datasets folder
- Tabular data display with current values, units, and timestamps
- X-axis uses `last_updated` timestamp from JSON file

### 3. Summary Metrics
- Quick overview of key sensor values
- Average confidence from both models
- Model agreement status

### 4. Model Comparison
- Side-by-side comparison of model predictions
- Feature values used for prediction
- Data source information

## Machine Learning Models

The system uses two trained models:

1. **Random Forest**: Ensemble learning model for stress classification
2. **Logistic Regression**: Linear model for stress probability estimation

### Model Features

The models use the following features (in order):
- `X`: Accelerometer X-axis (float)
- `Y`: Accelerometer Y-axis (float)
- `Z`: Accelerometer Z-axis (float)
- `EDA`: Electrodermal Activity (float) - loaded from datasets folder
- `HR`: Heart Rate in BPM (float)
- `TEMP`: Temperature in Celsius (float)
- `datetime_year`: Year (int)
- `datetime_month`: Month (int, 1-12)
- `datetime_day`: Day (int, 1-31)
- `datetime_hour`: Hour (int, 0-23)
- `datetime_dow`: Day of week (int, 0-6, Monday=0)

## Data Flow

```
ThingSpeak -> raspberry.py -> latest_sensor_data.json -> dashboard/app.py -> ML Models -> Stress Prediction
```

1. **raspberry.py** fetches data from ThingSpeak API
2. Data is parsed and datetime components are extracted
3. Data is saved to `latest_sensor_data.json`
4. **dashboard/app.py** reads from JSON file
5. Features are prepared and passed to ML models
6. Predictions are displayed in the dashboard

## Configuration

### ThingSpeak Setup

Edit `raspberry.py` to configure your ThingSpeak channel:

```python
CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API_KEY = "YOUR_READ_API_KEY"
```

### Dashboard Configuration

The dashboard automatically reads from `latest_sensor_data.json`. Ensure `raspberry.py` is running and has write permissions to create this file.

## Model Training

To train new models, use the scripts in the `train_model/` directory:

```bash
python train_model/random_forest.py
python train_model/logistic_regression.py
```

Trained models are saved to `saved_models/` directory in `.joblib` format.

## Troubleshooting

### Dashboard shows "Could not fetch data"
- Ensure `raspberry.py` is running
- Check that `latest_sensor_data.json` exists in the project root
- Verify file permissions for JSON file creation

### Prediction errors
- Ensure model files exist in `saved_models/` directory
- Check that feature values match expected format
- Verify EDA value is loaded from datasets folder

### ThingSpeak connection issues
- Verify internet connection
- Check ThingSpeak channel ID and API key
- Ensure ThingSpeak channel is public or API key has read permissions

## Technologies Used

- **Python 3.8+**
- **Streamlit**: Web dashboard framework
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **plotly**: Interactive graphs
- **joblib**: Model serialization
- **requests**: HTTP requests for ThingSpeak API

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Authors

[Add author information]

## Acknowledgments

- ThingSpeak for IoT data platform
- scikit-learn for ML framework
- Streamlit for dashboard framework

