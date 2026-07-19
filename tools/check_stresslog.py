"""One-off: count documents in Atlas stress log collections."""
import os
from pathlib import Path

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def main() -> None:
    uri = os.environ.get("MONGODB_URI") or read_env_file("MONGODB_URI")
    if not uri:
        print("Set MONGODB_URI in the environment or root .env file.")
        return

    client = MongoClient(uri, serverSelectionTimeoutMS=20000)
    db = client["Stress_Detection"]
    for name in ("stress_log_simulated", "stress_log_real"):
        col = db[name]
        n = col.count_documents({})
        print(f"Database: Stress_Detection, collection: {name}")
        print("Total documents:", n)
        for i, d in enumerate(col.find().sort("loggedAt", -1).limit(5)):
            print(f"--- #{i + 1} ---")
            print("  stressCategory:", d.get("stressCategory"))
            print("  stressLabel:", d.get("stressLabel"))
            print("  loggedAt:", d.get("loggedAt"))
    client.close()
    print("Read OK.")


def read_env_file(key: str) -> str:
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


if __name__ == "__main__":
    main()
