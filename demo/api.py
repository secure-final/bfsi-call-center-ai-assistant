"""FastAPI demo: single endpoint for query → response and tier."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel
from src.logging_config import setup_logging
from src.orchestrator import Orchestrator

setup_logging()
app = FastAPI(title="BFSI Call Center AI Assistant")
orch = Orchestrator()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    tier: str
    sources: str | None = None


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = orch.respond(req.query.strip())
    return QueryResponse(
        response=result.response,
        tier=result.tier,
        sources=result.sources,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
