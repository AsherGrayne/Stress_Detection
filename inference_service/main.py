"""HTTP inference API for Job/se_widedeep_model.keras (load once, predict per request)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent.parent
JOB_DIR = BASE_DIR / "Job"
PREPROCESS_PATH = JOB_DIR / "preprocessing.json"
SCALER_PATH = JOB_DIR / "scaler.pkl"
MODEL_PATH = JOB_DIR / "se_widedeep_model.keras"

_predictor: "StressPredictor | None" = None


def _strip_quantization_config(obj: object) -> None:
    key = "quantization_config"
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_quantization_config(x)


def load_job_keras_model(path: Path) -> keras.Model:
    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, "r") as zin:
            zin.extractall(tmpdir)
        cfg_path = os.path.join(tmpdir, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        _strip_quantization_config(cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        fixed = os.path.join(tmpdir, "fixed.keras")
        with zipfile.ZipFile(fixed, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in ("metadata.json", "config.json", "model.weights.h5"):
                zout.write(os.path.join(tmpdir, name), name)
        return keras.models.load_model(fixed)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_features_row(
    reading: dict[str, float], feature_columns: list[str]
) -> np.ndarray:
    row = pd.DataFrame([reading])
    out = pd.DataFrame(index=row.index)
    for c in feature_columns:
        if "*" in c:
            a, b = c.split("*", 1)
            out[c] = row[a].astype(np.float64) * row[b].astype(np.float64)
        else:
            out[c] = row[c].astype(np.float64)
    return out.to_numpy(dtype=np.float64)


class StressPredictor:
    def __init__(self) -> None:
        with open(PREPROCESS_PATH, encoding="utf-8") as f:
            prep = json.load(f)
        self._feature_columns: list[str] = prep["feature_columns"]
        self._scaler = joblib.load(SCALER_PATH)
        self._model = load_job_keras_model(MODEL_PATH)

    def predict_category(self, x: float, y: float, z: float, eda: float, hr: float, temp: float) -> int:
        reading = {"X": x, "Y": y, "Z": z, "EDA": eda, "HR": hr, "TEMP": temp}
        raw = build_features_row(reading, self._feature_columns)
        scaled = self._scaler.transform(raw)
        proba = self._model.predict(scaled, verbose=0)
        return int(np.argmax(proba, axis=1)[0])


class SensorReading(BaseModel):
    x: float
    y: float
    z: float
    eda: float
    hr: float
    temp: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _predictor
    _predictor = StressPredictor()
    yield
    _predictor = None


app = FastAPI(title="Stress IoT Inference", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(body: SensorReading) -> dict[str, int]:
    assert _predictor is not None
    cat = _predictor.predict_category(
        body.x, body.y, body.z, body.eda, body.hr, body.temp
    )
    return {"stressCategory": cat}
