import pandas as pd
import json
import time
import os
from datetime import datetime

CSV_PATH = os.path.join("datasets", "balanced_data.csv")
OUTPUT_FILE = "latest_sensor_data_dataset.json"

def parse_datetime(dt_str):
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

def process_row(row):
    dt_info = parse_datetime(row.get('datetime'))
    
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
    
    return sensor_data

def save_to_json(sensor_data):
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(sensor_data, f, indent=2)
        print(f"Data saved to {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    print(f"Reading data from {CSV_PATH}...")
    
    try:
        df = pd.read_csv(CSV_PATH, chunksize=1)
        
        chunk_count = 0
        for chunk in df:
            if len(chunk) == 0:
                break
            
            row = chunk.iloc[0]
            
            sensor_data = process_row(row)
            
            print(f"\nRow {chunk_count + 1}:")
            print(f"  BPM: {sensor_data['bpm']}")
            print(f"  Temperature: {sensor_data['temperature']}")
            print(f"  Accel X: {sensor_data['accel_x']}, Y: {sensor_data['accel_y']}, Z: {sensor_data['accel_z']}")
            print(f"  Timestamp: {sensor_data['timestamp']}")
            print(f"  Year: {sensor_data['datetime_year']}, Month: {sensor_data['datetime_month']}, Day: {sensor_data['datetime_day']}")
            print(f"  Hour: {sensor_data['datetime_hour']}, Day of Week: {sensor_data['datetime_dow']}")
            
            save_to_json(sensor_data)
            
            chunk_count += 1
            time.sleep(3)
            
    except Exception as e:
        print(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

