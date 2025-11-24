# Dashboard Setup - Reading from raspberry.py

## How It Works

The dashboard now reads data **directly from `raspberry.py`** when it's running, not from ThingSpeak.

### Data Flow:

```
raspberry.py (running)
    ↓ (fetches from ThingSpeak)
    ↓ (saves to JSON file)
latest_sensor_data.json
    ↓ (dashboard reads)
Streamlit Dashboard
    ↓ (makes predictions)
ML Models (joblib)
```

## Setup Instructions

### Step 1: Start raspberry.py

In **Terminal 1**:
```bash
python raspberry.py
```

This will:
- Fetch data from ThingSpeak every 15 seconds
- Save latest data to `latest_sensor_data.json`
- Display sensor readings in terminal

### Step 2: Start Dashboard

In **Terminal 2** (new terminal):
```bash
streamlit run dashboard/app.py
```

The dashboard will:
- Read from `latest_sensor_data.json`
- Display real-time predictions
- Auto-refresh when new data is available

## Important Notes

1. **Both must run simultaneously:**
   - `raspberry.py` must be running to update the JSON file
   - Dashboard reads from that JSON file

2. **File Location:**
   - JSON file: `latest_sensor_data.json` (project root)
   - Created automatically by `raspberry.py`
   - Dashboard reads from same location

3. **If dashboard shows "Could not fetch data":**
   - Check that `raspberry.py` is running
   - Check that `latest_sensor_data.json` exists
   - Check file permissions

## Quick Start

```bash
# Terminal 1: Start data fetcher
python raspberry.py

# Terminal 2: Start dashboard (in new terminal)
streamlit run dashboard/app.py
```

## Troubleshooting

**Dashboard shows "Could not fetch data":**
- ✅ Is `raspberry.py` running?
- ✅ Does `latest_sensor_data.json` exist?
- ✅ Check file permissions

**Data is old/stale:**
- Dashboard warns if data is >30 seconds old
- Check that `raspberry.py` is actively fetching data
- Check ThingSpeak connection in `raspberry.py`

**JSON file not updating:**
- Check `raspberry.py` output for errors
- Verify ThingSpeak API keys are correct
- Check internet connection

