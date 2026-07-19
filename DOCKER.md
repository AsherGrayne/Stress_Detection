# Docker deployment

Two containers:

| Service | Image build | Port | Role |
|---------|-------------|------|------|
| **inference** | `docker/Dockerfile.inference` | 8081 | Keras model (`Job/`), scaler, `POST /predict` |
| **app** | `docker/Dockerfile.node` | 8090 | Node.js API, CSV/Firebase scheduler, web dashboard |

MongoDB uses **Atlas** (or any URI); no Mongo container in this compose file.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2
- Repo root: `Stress_detection/` (contains `Job/`, `balanced_data.csv`, `inference_service/`, `node-app/`)

## Run

```bash
cd Stress_detection

# Optional: persist high-stress logs to Atlas
cp .env.example .env
# Edit .env — set MONGODB_URI

docker compose up --build
```

- Dashboard: `http://localhost:8090`
- Health: `http://localhost:8090/api/health`
- Simulated feed: `http://localhost:8090/api/telemetry/simulated/latest`
- Inference docs: `http://localhost:8081/docs`

## Environment

| Variable | Where | Purpose |
|----------|--------|---------|
| `MONGODB_URI` | Compose `.env` -> `app` | Atlas connection; if empty, mild/high stress logging is skipped |
| `STRESS_INFERENCE_BASE_URL` | Set in compose | Internal URL `http://inference:8081` (default) |
| `STRESS_CSV_PATH` | Dockerfile env | `/app/balanced_data.csv` baked into the Node image |

## Image sizes

The inference image is large (~2GB+) because of TensorFlow/Keras. For production, consider slim TF builds, multi-stage pruning, or a CPU-only wheel if available.

## Production notes

- Put **TLS** in front (reverse proxy / load balancer).
- Do **not** bake passwords into images; use secrets / `MONGODB_URI`.
- Rotate Atlas credentials if they were ever committed.
