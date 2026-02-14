"""One-time setup: install deps, build dataset index, build RAG index, pre-download embedding model."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    r = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, shell=False)
    return r.returncode == 0


def main():
    print("Step 1: Installing dependencies...")
    if not run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"]):
        print("pip install failed. Try: pip install -r requirements.txt")
        return 1

    print("Step 2: Pre-downloading embedding model (sentence-transformers)...")
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        print("  Embedding model ready.")
    except Exception as e:
        print("  Warning:", e)

    print("Step 3: Validating dataset...")
    if not run([sys.executable, "scripts/validate_dataset.py"]):
        print("  Run scripts/build_dataset.py first to generate data/alpaca_bfsi.json")
        return 1

    print("Step 4: Building dataset similarity index (Tier 1)...")
    if not run([sys.executable, "scripts/build_index.py"]):
        print("  build_index failed.")
        return 1

    print("Step 5: Building RAG index (Tier 3)...")
    if not run([sys.executable, "scripts/ingest_rag.py"]):
        print("  ingest_rag failed.")
        return 1

    print("Setup complete. Run: streamlit run demo/app_streamlit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
