# Stress Detection Dashboard

Lightweight stress monitoring app powered by:

- **Python FastAPI** for Keras/TensorFlow inference
- **Node.js + Express** for the API, telemetry scheduler, dashboard, and optional MongoDB logging
- **Vanilla HTML/CSS/JavaScript** for a professional browser dashboard

The previous Flutter and Spring Boot layers have been replaced by `node-app`.

## Architecture

```text
balanced_data.csv or Firebase
        |
        v
Node.js scheduler/API on :8090
        |
        v
Python inference service on :8081
        |
        v
Browser dashboard + optional MongoDB logs
```

## Run Locally

Terminal 1, start the Python inference service:

```bash
pip install -r inference_service/requirements.txt
python -m uvicorn inference_service.main:app --host 127.0.0.1 --port 8081
```

Terminal 2, start the Node.js app:

```bash
cd node-app
npm install
npm start
```

Open:

```text
http://localhost:8090
```

## API Endpoints

- `GET /api/health`
- `GET /api/telemetry/simulated/latest`
- `GET /api/telemetry/real/latest`
- `GET /api/telemetry/next`
- `GET /api/telemetry/simulated/history`
- `GET /api/telemetry/real/history`

## Configuration

Use environment variables or a `.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8090` | Node app port |
| `STRESS_CSV_PATH` | `../balanced_data.csv` from `node-app` | Simulated dataset path |
| `STRESS_INFERENCE_BASE_URL` | `http://127.0.0.1:8081` | Python inference API |
| `STRESS_INTERVAL_MS` | `5000` | Simulated and real feed interval |
| `STRESS_CHART_HISTORY_SIZE` | `60` | Points retained per chart |
| `FIREBASE_URL` | project Firebase URL | Real device telemetry source |
| `MONGODB_URI` | empty | Enables mild/high stress history logging |

## Docker

```bash
docker compose up --build
```

The dashboard/API runs at `http://localhost:8090`; Python inference remains at `http://localhost:8081`.
