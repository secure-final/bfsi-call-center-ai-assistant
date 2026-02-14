"""Chunk knowledge docs, embed, and store in Chroma for RAG (Tier 3)."""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, PROJECT_ROOT


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if end < len(words) else len(words)
    return chunks


def main():
    cfg = load_config()
    rag = cfg.get("rag", {})
    knowledge_path = PROJECT_ROOT / rag.get("knowledge_path", "knowledge")
    chroma_path = PROJECT_ROOT / rag.get("chroma_path", "data/rag_chroma")
    chunk_size = rag.get("chunk_size", 512)
    chunk_overlap = rag.get("chunk_overlap", 64)
    sim = cfg.get("similarity", {})
    embedding_model = sim.get("embedding_model", "all-MiniLM-L6-v2")

    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings

    embedder = SentenceTransformer(embedding_model)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
    collection_name = "bfsi_knowledge"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    coll = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    all_chunks = []
    for path in sorted(knowledge_path.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for c in chunks:
            all_chunks.append(c)
    if not all_chunks:
        print("No .md files found under", knowledge_path)
        return
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    ids = [f"c{i}" for i in range(len(all_chunks))]
    coll.add(ids=ids, embeddings=embeddings.tolist(), documents=all_chunks)
    print(f"Ingested {len(all_chunks)} chunks into {chroma_path}")


if __name__ == "__main__":
    main()
