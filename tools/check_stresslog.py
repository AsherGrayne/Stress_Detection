"""One-off: count documents in Atlas stresslog (reads URI from spring-backend/application-local.yml)."""
import re
from pathlib import Path

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
YML = ROOT / "spring-backend" / "application-local.yml"


def main() -> None:
    text = YML.read_text(encoding="utf-8")
    m = re.search(r'mongodb-uri:\s*"([^"]+)"', text)
    if not m:
        print("Could not parse mongodb-uri from", YML)
        return
    uri = m.group(1)
    client = MongoClient(uri, serverSelectionTimeoutMS=20000)
    db = client["Stress_Dtabase"]
    col = db["stresslog"]
    n = col.count_documents({})
    print("Database: Stress_Dtabase, collection: stresslog")
    print("Total documents:", n)
    for i, d in enumerate(col.find().sort("loggedAt", -1).limit(5)):
        print(f"--- #{i + 1} ---")
        print("  stressCategory:", d.get("stressCategory"))
        print("  stressLabel:", d.get("stressLabel"))
        print("  loggedAt:", d.get("loggedAt"))
    client.close()
    print("Read OK.")


if __name__ == "__main__":
    main()
