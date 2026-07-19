import requests
import time
from datetime import datetime

# ---------------- FIREBASE ----------------
FIREBASE_URL = "https://stress-detection-c2bec-default-rtdb.asia-southeast1.firebasedatabase.app/data.json"

# ---------------- MAIN LOOP ----------------
while True:
    try:
        response = requests.get(FIREBASE_URL)
        data = response.json()

        # System timestamp (PC time)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("----- Received Data -----")
        print("Time:", timestamp)

        temperature = data.get("temperature")
        pulse = data.get("pulse")
        gsr = data.get("gsr")
        accX = data.get("accX")
        accY = data.get("accY")
        accZ = data.get("accZ")

        print("Temperature:", temperature, "°C")
        print("Pulse (BPM):", pulse)
        print("GSR (µS):", gsr)
        print("Acceleration:", accX, accY, accZ)

        # -------- Real ML Inference --------
        # Pass the values to the local Keras Deep Learning Model
        inference_url = "http://127.0.0.1:8081/predict"
        
        # Default missing values to 0.0 to prevent crashes
        payload = {
            "x": accX if accX is not None else 0.0,
            "y": accY if accY is not None else 0.0,
            "z": accZ if accZ is not None else 0.0,
            "eda": gsr if gsr is not None else 0.0,
            "hr": pulse if pulse is not None else 0.0,
            "temp": temperature if temperature is not None else 0.0
        }
        
        try:
            ml_response = requests.post(inference_url, json=payload)
            if ml_response.status_code == 200:
                prediction = ml_response.json()
                category = prediction.get("stressCategory")
                
                # Convert the numerical category to a label
                if category == 0:
                    label = "0 - No Stress"
                elif category == 1:
                    label = "1 - Mild Stress"
                else:
                    label = "2 - High Stress"
                    
                print(f"Deep Learning Output: {label}")
            else:
                print(f"ML API Error: {ml_response.status_code}")
        except Exception as api_err:
            print("Could not reach Python Inference API. Is it running on port 8081?")

        print("------------------------------")

    except Exception as e:
        print("Error fetching data:", e)

    time.sleep(5)