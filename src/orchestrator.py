"""Orchestrate Tier 1 (dataset) → Tier 2 (SLM) → Tier 3 (RAG) and return final response."""
from dataclasses import dataclass
from typing import Optional

from src.config import load_config
from src.logging_config import get_logger
from src.similarity import DatasetSimilarity
from src.slm import SLMInference
from src.rag import RAGRetriever, is_complex_query
from src.guardrails import guardrail_pre, guardrail_post

logger = get_logger(__name__)


@dataclass
class ResponseResult:
    response: str
    tier: str  # "dataset" | "slm" | "rag"
    sources: Optional[str] = None


class Orchestrator:
    """Single entry point: query → guardrails pre → Tier 1 → Tier 2/3 → guardrails post."""

    def __init__(self):
        self.similarity = DatasetSimilarity()
        self.slm = SLMInference()
        self.rag = RAGRetriever()

    def respond(self, user_query: str) -> ResponseResult:
        """Run pipeline and return response with tier used. Never raises."""
        safe_fallback = ResponseResult(
            response=(
                "Something went wrong on our side. Please try again or contact customer care for assistance."
            ),
            tier="dataset",
        )
        try:
            if not user_query or not user_query.strip():
                return ResponseResult(
                    response="Please ask a banking, loan, or account-related question.",
                    tier="dataset",
                )
            reject_msg, sanitized = guardrail_pre(user_query)
            if reject_msg is not None:
                return ResponseResult(response=reject_msg, tier="dataset")

            stored, score = self.similarity.query(sanitized)
            if stored is not None:
                final = guardrail_post(stored)
                return ResponseResult(response=final, tier="dataset")

            context = None
            if is_complex_query(sanitized):
                context = self.rag.retrieve(sanitized)
                if context:
                    response = self.slm.generate(
                        instruction=sanitized,
                        input_text="",
                        context=context,
                    )
                    final = guardrail_post(response, allowed_context=context)
                    return ResponseResult(
                        response=final, tier="rag", sources=context[:500]
                    )
            response = self.slm.generate(instruction=sanitized, input_text="")
            final = guardrail_post(response)
            return ResponseResult(response=final, tier="slm")
        except Exception as e:
            logger.exception("Orchestrator respond failed: %s", e)
            return safe_fallback
