"""Build or rebuild the dataset similarity index (Tier 1). Run after updating data/alpaca_bfsi.json."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity import DatasetSimilarity


def main():
    ds = DatasetSimilarity()
    if ds._load_dataset() is None:
        print("ERROR: Dataset not found or empty. Run scripts/build_dataset.py first.")
        sys.exit(1)
    if not ds._build_index():
        print("ERROR: Failed to build index.")
        sys.exit(1)
    print("Dataset index built successfully at", ds.index_path)


if __name__ == "__main__":
    main()
