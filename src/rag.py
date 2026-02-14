"""Tier 3: RAG retrieval over knowledge base. Returns context for SLM to generate grounded response."""
from pathlib import Path
from typing import List

from src.config import PROJECT_ROOT, load_config
from src.logging_config import get_logger

logger = get_logger(__name__)


def is_complex_query(query: str, keywords: List[str] | None = None) -> bool:
    """Heuristic: query is complex if it contains any of the configured keywords."""
    cfg = load_config()
    keywords = keywords or cfg.get("rag", {}).get("complex_keywords", [])
    q = (query or "").lower()
    return any(kw.lower() in q for kw in keywords)


class RAGRetriever:
    """Retrieve relevant chunks from knowledge base for a query."""

    def __init__(
        self,
        chroma_path: Path | str | None = None,
        knowledge_path: Path | str | None = None,
        embedding_model: str | None = None,
        top_k: int = 3,
    ):
        cfg = load_config()
        rag = cfg.get("rag", {})
        sim = cfg.get("similarity", {})
        self.chroma_path = Path(chroma_path or rag.get("chroma_path", "data/rag_chroma"))
        if not self.chroma_path.is_absolute():
            self.chroma_path = PROJECT_ROOT / self.chroma_path
        self.knowledge_path = Path(knowledge_path or rag.get("knowledge_path", "knowledge"))
        if not self.knowledge_path.is_absolute():
            self.knowledge_path = PROJECT_ROOT / self.knowledge_path
        self.embedding_model_name = embedding_model or sim.get("embedding_model", "all-MiniLM-L6-v2")
        self.top_k = top_k or rag.get("top_k", 3)
        self._client = None
        self._coll = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_collection(self):
        if self._coll is not None:
            return self._coll
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_path), settings=Settings(anonymized_telemetry=False))
        try:
            self._coll = client.get_collection("bfsi_knowledge", metadata={"hnsw:space": "cosine"})
        except Exception:
            raise RuntimeError("RAG index not found. Run: python scripts/ingest_rag.py")
        self._client = client
        return self._coll

    def retrieve(self, query: str) -> str:
        """Return concatenated context from top-k chunks. Empty if no index or on error."""
        if not query or not query.strip():
            return ""
        try:
            coll = self._get_collection()
        except Exception:
            logger.warning("RAG index missing or error; returning empty context")
            return ""
        try:
            embedder = self._get_embedder()
            q_emb = embedder.encode([query.strip()])
            n = min(self.top_k, coll.count())
            if n == 0:
                return ""
            results = coll.query(
                query_embeddings=q_emb.tolist(),
                n_results=n,
                include=["documents"],
            )
            if not results["documents"] or not results["documents"][0]:
                return ""
            docs = results["documents"][0]
            return "\n\n".join(docs)
        except Exception as e:
            logger.exception("RAG retrieve failed: %s", e)
            return ""
