"""Validate Alpaca BFSI dataset schema and count."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "alpaca_bfsi.json"


def validate():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return False
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Dataset must be a JSON array.")
        return False
    required = {"instruction", "output"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Item {i}: must be an object.")
            return False
        if "input" not in item:
            item["input"] = ""
        for key in required:
            if key not in item or not isinstance(item[key], str):
                print(f"Item {i}: missing or invalid '{key}'.")
                return False
        if len(item["instruction"].strip()) == 0 or len(item["output"].strip()) == 0:
            print(f"Item {i}: instruction and output must be non-empty.")
            return False
    if len(data) < 150:
        print(f"Dataset has {len(data)} samples; minimum required is 150.")
        return False
    print(f"Valid: {len(data)} samples, Alpaca format (instruction, input, output).")
    return True


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
