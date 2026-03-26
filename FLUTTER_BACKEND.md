# Stress IoT — Spring Boot, Python inference, Flutter, MongoDB

**Docker:** see [`DOCKER.md`](DOCKER.md) — `docker compose up --build` runs inference + Spring (ML model included in the inference image).

## Architecture

1. **Python** (`inference_service`) — loads `Job/se_widedeep_model.keras` + `Job/scaler.pkl`, exposes `POST /predict` on port **8081**.
2. **Spring Boot** (`spring-backend`) — every **5 seconds** reads the next row from `balanced_data.csv`, runs inference, updates in-memory chart history, exposes **`GET /api/telemetry/simulated/latest`**. When predicted category is **2 (High Stress)**, inserts a document into MongoDB.
3. **Flutter** (`flutter_app`) — home screen has **Simulated** and **Real** (placeholder). **Simulated** polls the latest snapshot every **5 seconds** and draws **six line charts** (time on X-axis, sensor value on Y-axis) plus a **Prediction** card.

## MongoDB (high-stress logs only)

- **Database:** `Stress_Dtabase` (as in your Atlas setup)
- **Collection:** `stresslog`
- **Document fields:** `stressCategory`, `stressLabel`, `loggedAt` (Java `Date` → BSON date), `reading` (embedded `x`, `y`, `z`, `eda`, `hr`, `temp`)

Set the connection URI **via environment variable** (do not commit passwords to public repos):

**PowerShell (use your Atlas user/password):**

```powershell
$env:MONGODB_URI = "mongodb+srv://USER:PASSWORD@cluster0.enwyvnr.mongodb.net/Stress_Dtabase?retryWrites=true&w=majority"
cd spring-backend
mvn spring-boot:run
```

If `MONGODB_URI` is unset, the app still runs; high-stress rows are **not** persisted.

Ensure Atlas **Network Access** allows your current IP (or `0.0.0.0/0` for testing only).

## Run backend

**Terminal 1 — project root `Stress_detection/`:**

```bash
pip install -r inference_service/requirements.txt
python -m uvicorn inference_service.main:app --host 127.0.0.1 --port 8081
```

**Terminal 2 — with `MONGODB_URI` set if you want logging:**

```bash
cd spring-backend
mvn spring-boot:run
```

Default Spring port: **8090** (`application.yml`). Python inference: **8081**.

### Config (`spring-backend/src/main/resources/application.yml`)

| Key | Purpose |
|-----|---------|
| `stress.csv-path` | Path to `balanced_data.csv` (default `../balanced_data.csv` from `spring-backend/`) |
| `stress.inference-base-url` | Python API base URL |
| `stress.simulated-interval-ms` | Simulated tick interval (default `5000`) |
| `stress.chart-history-size` | Points kept per sensor in `series` (default `60`) |
| `stress.mongodb-uri` | `${MONGODB_URI:}` |

## REST API

### `GET /api/telemetry/simulated/latest`

Returns the **current** simulated sample (refreshed on the server every 5s) and rolling **series** for charts:

```json
{
  "observedAt": "2026-03-21T14:32:01.123Z",
  "sequenceIndex": 42,
  "reading": { "x": -34.0, "y": 51.0, "z": 19.0, "eda": 2.71, "hr": 97.98, "temp": 34.0 },
  "predictedStressCategory": 2,
  "predictedStressLabel": "High Stress",
  "series": {
    "X": [{ "t": 1711, "v": -34.0 }],
    "Y": [{ "t": 1711, "v": 51.0 }
    ]
  }
}
```

- **`t`**: epoch milliseconds (chart **X-axis** = time)  
- **`v`**: sensor value (chart **Y-axis**)  
- Labels: **0 — No Stress**, **1 — Mild Stress**, **2 — High Stress**

### `GET /api/telemetry/next`

Still advances the **shared** CSV cursor manually (use mainly for debugging; the Flutter app uses `/simulated/latest`).

### `GET /api/health`

## Flutter app

```bash
cd flutter_app
flutter pub get
flutter run
```

**API base URL** (`lib/config.dart`):

- **Android emulator** default: `http://10.0.2.2:8090` (maps to host loopback)
- **iOS simulator / desktop**: `flutter run --dart-define=API_BASE=http://127.0.0.1:8090`
- **Physical phone**: same Wi‑Fi as PC, e.g. `--dart-define=API_BASE=http://192.168.x.x:8090`

Android **cleartext HTTP** is enabled for dev (`AndroidManifest`). iOS allows **local networking** for HTTP in `Info.plist`.

## CORS

Spring allows all origins on `/api/**` for local development.
