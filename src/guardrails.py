"""Guardrails: out-of-domain, PII, unsafe/unethical intent, no guessing of financial numbers."""
import re
from typing import Tuple

from src.config import load_config
from src.logging_config import get_logger

logger = get_logger(__name__)

# BFSI-related terms for in-domain check (broad so user queries are accepted)
BFSI_KEYWORDS = [
    "loan", "emi", "interest", "rate", "payment", "account", "bank", "balance",
    "eligibility", "application", "statement", "transfer", "card", "kyc",
    "foreclosure", "prepayment", "tenure", "disbursement", "sanction",
    "home loan", "personal loan", "savings", "fd", "nre", "nro", "nominee",
    "net banking", "customer care", "branch", "complaint", "penalty", "policy",
    "finance", "financial", "insurance", "invest", "deposit", "withdraw",
    "credit", "debit", "atm", "cheque", "draft", "neft", "imps", "rtgs",
    "overdraft", "mortgage", "refinance", "repay", "outstanding", "due",
    "help", "support", "query", "question", "information", "details",
]

# Terms suggesting unethical or illegal intent (e.g. manipulate score, fake documents)
UNSAFE_INTENT_KEYWORDS = [
    "manipulate", "manipulation", "cheat", "fake", "forge", "forged", "falsify",
    "hack", "rig", "game the system", "trick the system", "fraud", "fraudulent",
    "illegal", "unethical", "misrepresent", "hide debt", "conceal",
]


def has_unsafe_intent(query: str) -> bool:
    """True if query suggests unethical or illegal intent (e.g. manipulating credit score)."""
    if not query or not query.strip():
        return False
    q = query.lower()
    return any(kw in q for kw in UNSAFE_INTENT_KEYWORDS)


def is_out_of_domain(query: str) -> bool:
    """True if query appears unrelated to BFSI (banking, loan, account)."""
    if not query or not query.strip():
        return True
    q = query.lower()
    return not any(kw in q for kw in BFSI_KEYWORDS)


def contains_pii(query: str) -> bool:
    """Simple heuristic: detect likely PII (Aadhaar pattern, long digit strings)."""
    if not query:
        return False
    # Aadhaar: 4 groups of 4 digits
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", query):
        return True
    # Long digit string (e.g. account number)
    if re.search(r"\b\d{10,}\b", query):
        return True
    return False


def guardrail_pre(query: str) -> Tuple[str | None, str | None]:
    """
    Pre-processing guardrails. Returns (rejection_message, None) if query should be rejected,
    else (None, sanitized_query). Do not log full query if it may contain PII.
    """
    cfg = load_config()
    if not cfg.get("guardrails", {}).get("enabled", True):
        return None, query
    if contains_pii(query):
        logger.warning("Query rejected: possible PII detected")
        return "For your security, please do not share account numbers or personal IDs in the chat. You may contact our helpline for account-specific queries.", None
    if has_unsafe_intent(query):
        logger.warning("Query rejected: unsafe or unethical intent detected")
        msg = cfg.get("guardrails", {}).get(
            "unsafe_intent_message",
            "We can only assist with legitimate ways to improve or manage your credit score and financial health. We do not provide guidance on manipulating, misrepresenting, or falsifying any information. If you would like to know how to improve your credit score, correct errors in your report, reduce debt, or understand your score, please ask and we will be happy to help."
        )
        return msg, None
    if is_out_of_domain(query):
        msg = cfg.get("guardrails", {}).get("out_of_domain_message", "I can only help with banking, loan, and account-related queries. Please ask a question in that domain.")
        logger.info("Query rejected: out of domain")
        return msg, None
    return None, query


def _sanitize_unsafe_echo(response: str) -> str:
    """If response echoes unsafe intent terms (e.g. 'manipulate'), reframe to safe wording."""
    if not response or not response.strip():
        return response
    r = response
    # Reframe common unsafe echoes in credit/score context to legitimate wording
    replacements = [
        (r"\bmanipulate\s+(?:the\s+)?credit\s+score", "improve your credit score", re.IGNORECASE),
        (r"\bmanipulat(?:e|ing)\s+(?:the\s+)?(?:credit\s+)?system", "improving your credit", re.IGNORECASE),
        (r"\bto\s+manipulate\b", "to improve", re.IGNORECASE),
    ]
    for pattern, repl, flags in replacements:
        r = re.sub(pattern, repl, r)
    return r


def guardrail_post(response: str, allowed_context: str | None = None) -> str:
    """
    Post-processing: sanitize any unsafe intent wording echoed in response, then append disclaimer.
    """
    cfg = load_config()
    if not cfg.get("guardrails", {}).get("enabled", True):
        return response
    response = _sanitize_unsafe_echo(response)
    disclaimer = cfg.get("guardrails", {}).get("disclaimer", "")
    if disclaimer and response:
        return response.rstrip() + "\n\n" + disclaimer
    return response
