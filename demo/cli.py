"""CLI demo: single query in, print response and tier used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.logging_config import setup_logging
from src.orchestrator import Orchestrator


def main():
    setup_logging()
    orch = Orchestrator()
    print("BFSI Call Center AI Assistant (CLI). Type your query and press Enter. 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        result = orch.respond(q)
        print(f"[{result.tier.upper()}] {result.response}\n")


if __name__ == "__main__":
    main()
