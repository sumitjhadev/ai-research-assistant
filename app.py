"""Streamlit demo frontend for the AI Research Assistant.

Three tabs:
  - Ask: citation-grounded Q&A over the ingested arXiv corpus
  - Summarize: structured per-paper summary
  - Compare: markdown comparison table across 2-5 papers

Talks to the FastAPI backend over HTTP (BACKEND_URL env var, default
http://localhost:8000). Run with:

    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_papers() -> list[dict[str, Any]]:
    """Fetch the list of ingested papers from the backend (cached for 5 min).

    Returns:
        List of paper metadata dicts, or an empty list on failure.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/papers", timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")
        return []


def render_sidebar(papers: list[dict[str, Any]]) -> None:
    """Render the sidebar with corpus stats and backend connection info.

    Args:
        papers: List of ingested paper metadata dicts.
    """
    st.sidebar.title("📚 Corpus Status")
    st.sidebar.markdown(f"**Backend:** `{BACKEND_URL}`")
    extracted = sum(1 for p in papers if p.get("extracted"))
    st.sidebar.metric("Papers ingested", len(papers))
    st.sidebar.metric("Papers with extracted text", extracted)
    if papers:
        with st.sidebar.expander("View paper list"):
            for p in papers:
                st.markdown(f"- `{p['paper_id']}` — {p['title'][:60]}")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Answers are grounded strictly in retrieved paper excerpts. "
        "If the corpus lacks evidence, the assistant will say so instead of guessing."
    )


def render_ask_tab() -> None:
    """Render the Ask tab: question box, retrieval settings, grounded answer."""
    st.header("Ask a research question")
    st.caption(
        "Every claim in the answer is cited with a [paper_id]. If the corpus doesn't "
        "have the answer, the assistant will say 'not enough information in the corpus'."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What retrieval strategies improve factual accuracy in RAG systems?",
        height=100,
    )
    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=20, value=5)

    if st.button("Ask", type="primary", use_container_width=True):
        if not question or len(question.strip()) < 3:
            st.warning("Please enter a question of at least 3 characters.")
            return
        with st.spinner("Retrieving context and generating a grounded answer..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question.strip(), "k": k},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if resp.status_code == 429:
                    st.warning("Rate limited — please wait a moment and try again.")
                    return
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        st.markdown("### Answer")
        st.markdown(data["answer"])

        check = data.get("citation_check", {})
        if check.get("invalid_ids"):
            st.warning(
                f"⚠️ The answer cited paper IDs not found in the retrieved context: "
                f"{check['invalid_ids']}"
            )
        elif check.get("declined"):
            st.info("The assistant declined to answer due to insufficient evidence in the corpus.")
        else:
            st.success("✅ All citations verified against retrieved sources.")

        if data.get("cached"):
            st.caption("⚡ Served from cache (identical question asked recently).")

        st.markdown("### Sources")
        for src in data.get("sources", []):
            st.markdown(
                f"- **[{src['paper_id']}]** {src['title']} "
                f"(similarity: {src['score']:.3f}) — [PDF]({src['source_url']})"
            )


def render_summarize_tab(papers: list[dict[str, Any]]) -> None:
    """Render the Summarize tab: paper picker + structured summary output.

    Args:
        papers: List of ingested paper metadata dicts, for the selectbox.
    """
    st.header("Summarize a paper")
    st.caption("Generates a structured summary: Methodology, Dataset, Key Results, Limitations.")

    if not papers:
        st.info("No papers ingested yet. Run `python backend/ingest.py` first.")
        return

    options = {f"{p['paper_id']} — {p['title'][:70]}": p["paper_id"] for p in papers}
    choice = st.selectbox("Choose a paper", list(options.keys()))

    if st.button("Summarize", type="primary", use_container_width=True):
        paper_id = options[choice]
        with st.spinner("Generating summary..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/summarize",
                    json={"paper_id": paper_id},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        st.markdown(f"### {data['title']}")
        st.markdown(data["summary"])


def render_compare_tab(papers: list[dict[str, Any]]) -> None:
    """Render the Compare tab: multi-select paper picker + comparison table.

    Args:
        papers: List of ingested paper metadata dicts, for the multiselect.
    """
    st.header("Compare papers")
    st.caption("Select 2-5 papers to generate a Method / Dataset / Model / Results / "
               "Limitations comparison table.")

    if not papers:
        st.info("No papers ingested yet. Run `python backend/ingest.py` first.")
        return

    options = {f"{p['paper_id']} — {p['title'][:70]}": p["paper_id"] for p in papers}
    choices = st.multiselect("Choose 2-5 papers", list(options.keys()))

    if st.button("Compare", type="primary", use_container_width=True):
        if not (2 <= len(choices) <= 5):
            st.warning("Please select between 2 and 5 papers.")
            return
        paper_ids = [options[c] for c in choices]
        with st.spinner("Generating comparison..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/compare",
                    json={"paper_ids": paper_ids},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        st.markdown(data["comparison"])


def main() -> None:
    """Render the full Streamlit app: title, sidebar, and three tabs."""
    st.title("📚 AI Research Assistant")
    st.caption(
        "Agentic RAG over arXiv papers — every answer is traceable back to a specific source."
    )

    papers = fetch_papers()
    render_sidebar(papers)

    tab_ask, tab_summarize, tab_compare = st.tabs(["🔎 Ask", "📝 Summarize", "⚖️ Compare"])
    with tab_ask:
        render_ask_tab()
    with tab_summarize:
        render_summarize_tab(papers)
    with tab_compare:
        render_compare_tab(papers)


if __name__ == "__main__":
    main()
