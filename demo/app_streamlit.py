"""Streamlit demo: query input, response and tier displayed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.logging_config import setup_logging
from src.orchestrator import Orchestrator


def main():
    setup_logging()
    st.set_page_config(page_title="BFSI Call Center AI Assistant", layout="centered")
    st.title("BFSI Call Center AI Assistant")
    st.caption("Ask a banking, loan or account-related question.")
    if "orch" not in st.session_state:
        st.session_state.orch = Orchestrator()
    q = st.text_input("Your question", placeholder="e.g. How is EMI calculated?")
    if st.button("Get response") and q:
        with st.spinner("Thinking..."):
            result = st.session_state.orch.respond(q.strip())
        st.success(f"**Tier used:** {result.tier.upper()}")
        st.markdown(result.response)
        if result.sources:
            with st.expander("RAG sources (excerpt)"):
                st.text(result.sources[:800])


if __name__ == "__main__":
    main()
