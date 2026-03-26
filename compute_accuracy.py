"""Evaluate Job/se_widedeep_model.keras on balanced_data.csv; write accuracy to a .txt file."""
import json
import os
import shutil
import tempfile
import warnings
import zipfile

import joblib
import keras
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CSV_PATH = "balanced_data.csv"
JOB_DIR = "Job"
PREPROCESS_PATH = os.path.join(JOB_DIR, "preprocessing.json")
SCALER_PATH = os.path.join(JOB_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(JOB_DIR, "se_widedeep_model.keras")
OUT_PATH = "job_widedeep_accuracy.txt"
LABEL_COL = "label"


def _strip_quantization_config(obj: object) -> None:
    key = "quantization_config"
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_quantization_config(x)


def load_job_keras_model(path: str) -> keras.Model:
    """Load .keras zip; drop Dense quantization_config so older Keras can deserialize."""
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


def build_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in feature_columns:
        if "*" in c:
            a, b = c.split("*", 1)
            out[c] = df[a].astype(np.float64) * df[b].astype(np.float64)
        else:
            out[c] = df[c].astype(np.float64)
    return out


def main() -> None:
    with open(PREPROCESS_PATH, encoding="utf-8") as f:
        prep = json.load(f)
    feature_columns = prep["feature_columns"]

    df = pd.read_csv(CSV_PATH)
    y = df[LABEL_COL].to_numpy(dtype=np.float64)

    X = build_features(df, feature_columns)
    scaler = joblib.load(SCALER_PATH)
    Xs = scaler.transform(X)

    model = load_job_keras_model(MODEL_PATH)
    proba = model.predict(Xs, batch_size=4096, verbose=0)
    pred = np.argmax(proba, axis=1).astype(np.float64)

    correct = (pred == y).sum()
    n = len(y)
    acc = correct / n

    lines = [
        "balanced_data.csv vs Job/se_widedeep_model.keras (+ Job/scaler.pkl, preprocessing.json)",
        f"Rows evaluated: {n}",
        f"Correct: {int(correct)}",
        f"Accuracy: {acc:.6f}",
    ]
    text = "\n".join(lines) + "\n"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


if __name__ == "__main__":
    main()
