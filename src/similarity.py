"""Tier 1: Dataset similarity layer. Return stored response if query matches Alpaca samples."""
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, load_config
from src.logging_config import get_logger

logger = get_logger(__name__)


def _text_for_embedding(instruction: str, input_text: str) -> str:
    if (input_text or "").strip():
        return f"{instruction.strip()} {input_text.strip()}".strip()
    return instruction.strip()


class DatasetSimilarity:
    """Match user query to Alpaca dataset via embeddings; return stored output if above threshold."""

    def __init__(
        self,
        dataset_path: Path | None = None,
        index_path: Path | None = None,
        embedding_model: str | None = None,
        threshold: float | None = None,
        top_k: int = 1,
    ):
        cfg = load_config()
        sim = cfg.get("similarity", {})
        self.dataset_path = Path(dataset_path or sim.get("dataset_path", "data/alpaca_bfsi.json"))
        if not self.dataset_path.is_absolute():
            self.dataset_path = PROJECT_ROOT / self.dataset_path
        self.index_path = Path(index_path or sim.get("index_path", "data/dataset_index"))
        if not self.index_path.is_absolute():
            self.index_path = PROJECT_ROOT / self.index_path
        self.embedding_model_name = embedding_model or sim.get("embedding_model", "all-MiniLM-L6-v2")
        self.threshold = threshold if threshold is not None else float(sim.get("threshold", 0.88))
        self.top_k = top_k or sim.get("top_k", 1)
        self._model = None
        self._index = None
        self._samples = None

    def _load_dataset(self) -> list[dict] | None:
        if self._samples is not None:
            return self._samples
        if not self.dataset_path.exists():
            logger.error("Dataset not found: %s. Run scripts/build_dataset.py and scripts/build_index.py", self.dataset_path)
            return None
        try:
            import json
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self._samples = json.load(f)
            if not self._samples:
                logger.warning("Dataset is empty")
                return None
            logger.info("Loaded %s dataset samples", len(self._samples))
            return self._samples
        except Exception as e:
            logger.exception("Failed to load dataset: %s", e)
            return None

    def _get_embedder(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _build_index(self) -> bool:
        """Build or load Chroma index for (instruction, input) texts. Returns True on success."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
        if self._load_dataset() is None:
            return False
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        persist_dir = str(self.index_path)
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        collection_name = "bfsi_alpaca"
        try:
            coll = client.get_collection(collection_name)
            if coll.count() == len(self._samples):
                self._index = coll
                self._client = client
                logger.info("Loaded existing similarity index at %s", persist_dir)
                return True
        except Exception:
            pass
        embedder = self._get_embedder()
        texts = [_text_for_embedding(s["instruction"], s.get("input", "")) for s in self._samples]
        embeddings = embedder.encode(texts, show_progress_bar=len(texts) > 50)
        ids = [str(i) for i in range(len(self._samples))]
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        coll = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        coll.add(ids=ids, embeddings=embeddings.tolist(), documents=texts)
        self._index = coll
        self._client = client
        logger.info("Built similarity index with %s vectors", len(ids))
        return True

    def query(self, user_query: str) -> tuple[str | None, float | None]:
        """
        Return (stored_output, score) if best match >= threshold; else (None, best_score).
        On any failure returns (None, None).
        """
        if not user_query or not user_query.strip():
            return None, None
        try:
            if self._load_dataset() is None:
                return None, None
            if not self._build_index():
                return None, None
            embedder = self._get_embedder()
            q_emb = embedder.encode([user_query.strip()])
            results = self._index.query(
                query_embeddings=q_emb.tolist(),
                n_results=min(self.top_k, len(self._samples)),
                include=["documents", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return None, None
            dist = results["distances"][0][0]
            similarity = max(0.0, 1.0 - float(dist))
            idx = int(results["ids"][0][0])
            if similarity >= self.threshold:
                output = self._samples[idx]["output"]
                logger.info("Tier 1 match: similarity=%.3f", similarity)
                return output, similarity
            logger.info("Tier 1 no match: best similarity=%.3f (threshold=%.2f)", similarity, self.threshold)
            return None, similarity
        except Exception as e:
            logger.exception("Similarity query failed: %s", e)
            return None, None
