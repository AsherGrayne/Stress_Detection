import requests
import time
import json
import os
from datetime import datetime

CHANNEL_ID = "3160510"
READ_API_KEY = "Z7ZCVHQMMHBUUD2L"
FETCH_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"

def fetch_latest():
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
                print(f"⚠️ Error parsing timestamp: {e}")

        if all([bpm, temperature, accel_x, accel_y, accel_z]):
            print(f"\n🕒 Timestamp: {timestamp}")
            print(f"   Year: {datetime_year}, Month: {datetime_month}, Day: {datetime_day}")
            print(f"   Hour: {datetime_hour}, Day of Week: {datetime_dow}")
            print(f"❤️ BPM: {bpm}")
            print(f"🌡️ Temperature: {temperature} °C")
            print(f"⚙️ Accel X: {accel_x},  Y: {accel_y},  Z: {accel_z}")
            
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
            
            data_file = 'latest_sensor_data.json'
            try:
                with open(data_file, 'w') as f:
                    json.dump(sensor_data, f, indent=2)
                print(f"💾 Data saved to {data_file} for dashboard")
            except Exception as e:
                print(f"⚠️ Could not save data file: {e}")
            
            return sensor_data
        else:
            print("⚠️ Missing one or more fields in the latest entry.")

    except Exception as e:
        print("❌ Error fetching data:", e)
        return None

print("📡 Fetching latest BPM, Temperature, and Accel (X, Y, Z) from ThingSpeak...")

while True:
    fetch_latest()
    time.sleep(3)
