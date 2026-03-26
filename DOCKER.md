# Docker deployment

**Hosting this stack on the public internet** (Flutter Web + HTTPS + CORS): see [`HOSTING.md`](HOSTING.md).

Two containers:

| Service | Image build | Port | Role |
|---------|-------------|------|------|
| **inference** | `docker/Dockerfile.inference` | 8081 | Keras model (`Job/`), scaler, `POST /predict` |
| **backend** | `docker/Dockerfile.spring` | 8090 | Spring Boot API, CSV simulator, calls inference |

MongoDB uses **Atlas** (or any URI); no Mongo container in this compose file.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2
- Repo root: `Stress_detection/` (contains `Job/`, `balanced_data.csv`, `inference_service/`, `spring-backend/`)

## Run

```bash
cd Stress_detection

# Optional: persist high-stress logs to Atlas
cp .env.example .env
# Edit .env — set MONGODB_URI

docker compose up --build
```

- Health: `http://localhost:8090/api/health`
- Simulated feed: `http://localhost:8090/api/telemetry/simulated/latest`
- Inference docs: `http://localhost:8081/docs`

## Environment

| Variable | Where | Purpose |
|----------|--------|---------|
| `MONGODB_URI` | Compose `.env` → `backend` | Atlas connection; if empty, high-stress logging is skipped |
| `STRESS_INFERENCE_BASE_URL` | Set in compose | Internal URL `http://inference:8081` (default) |
| `STRESS_CSV_PATH` | Dockerfile env | `/data/balanced_data.csv` baked into backend image |

## Flutter

Build the app against the **host** API (ports published as above):

```bash
cd flutter_app
flutter run -d windows --dart-define=API_BASE=http://127.0.0.1:8090
# Web:
flutter run -d chrome --dart-define=API_BASE=http://localhost:8090
```

On a **phone**, use your PC’s LAN IP and ensure the firewall allows **8090** (and **8081** only if the app called inference directly — it doesn’t; only 8090 is required for the Flutter client).

## Image sizes

The inference image is large (~2GB+) because of TensorFlow/Keras. For production, consider slim TF builds, multi-stage pruning, or a CPU-only wheel if available.

## Production notes

- Put **TLS** in front (reverse proxy / load balancer).
- Do **not** bake `application-local.yml` with passwords into images; use secrets / `MONGODB_URI`.
- Rotate Atlas credentials if they were ever committed.
