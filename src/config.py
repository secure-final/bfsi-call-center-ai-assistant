"""Load configuration from config.yaml and environment."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Default project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    if not path.exists():
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Override with env
    if os.getenv("SIMILARITY_THRESHOLD"):
        cfg.setdefault("similarity", {})["threshold"] = float(os.getenv("SIMILARITY_THRESHOLD"))
    if os.getenv("LOG_LEVEL"):
        cfg.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    return cfg


def _default_config() -> dict:
    return {
        "similarity": {"threshold": 0.88, "embedding_model": "all-MiniLM-L6-v2", "top_k": 1},
        "slm": {"base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "max_new_tokens": 256, "temperature": 0.3},
        "rag": {"top_k": 3, "complex_keywords": ["emi", "interest", "rate", "penalty", "policy"]},
        "guardrails": {"enabled": True},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    }


def get_logging_config(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return cfg.get("logging", {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"})
