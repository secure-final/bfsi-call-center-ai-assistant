"""Quick test: Tier 1 and guardrails (no SLM load). Run after setup."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_tier1_and_guardrails():
    from src.orchestrator import Orchestrator

    orch = Orchestrator()
    # Should hit Tier 1 (dataset)
    r = orch.respond("How is EMI calculated?")
    assert r.response, "Empty response"
    assert r.tier == "dataset", f"Expected tier=dataset, got {r.tier}"
    print("[PASS] Tier 1 (dataset): How is EMI calculated?")

    # Out-of-domain
    r = orch.respond("What is the capital of France?")
    assert "banking" in r.response.lower() or "account" in r.response.lower()
    print("[PASS] Guardrails: out-of-domain rejected")

    # Empty
    r = orch.respond("")
    assert r.response
    print("[PASS] Empty query handled")

    # BFSI query that should match dataset
    r = orch.respond("Where can I see my EMI schedule?")
    assert r.response
    if r.tier == "dataset":
        print("[PASS] Tier 1: EMI schedule")
    else:
        print("[INFO] EMI schedule used tier:", r.tier)

    print("All checks passed (Tier 1 + guardrails).")


if __name__ == "__main__":
    test_tier1_and_guardrails()
