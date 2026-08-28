"""Citation-grounded RAG: question answering, summarization, comparison.

Pipeline stage: Semantic Retrieval -> LLM Generation -> Citation Verification
                 -> Answer with Sources

All LLM calls funnel through a single `_generate()` helper so swapping
providers later only touches one function. The current provider is Google
Gemini (gemini-2.5-flash) via the google-generativeai SDK.

The system prompt strictly forbids the model from answering outside the
retrieved context: if the retrieved chunks don't support an answer, the
model must say so explicitly rather than guessing.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai

from backend.config import (
    BACKOFF_BASE_SECONDS,
    DEFAULT_TOP_K,
    GEMINI_MODEL_NAME,
    MAX_RETRIES,
    get_logger,
    require_google_api_key,
)
from backend.ingest import load_papers
from backend.vectorstore import get_vector_store

logger = get_logger(__name__)

_NOT_ENOUGH_INFO = "not enough information in the corpus"

_QA_SYSTEM_PROMPT = """You are a research assistant that answers questions strictly using \
the provided excerpts from academic papers. Follow these rules exactly:

1. Answer ONLY using information contained in the provided context chunks below.
2. Every factual claim you make MUST be followed by a citation in the form [paper_id], \
using the paper_id shown next to the chunk it came from.
3. If the context does not contain enough information to answer the question confidently, \
you MUST respond with exactly: "not enough information in the corpus" (optionally followed \
by a brief explanation of what IS covered). Do not guess or use outside knowledge.
4. Do not fabricate paper IDs, numbers, or findings that are not present in the context.
5. Be concise and precise. Prefer short, well-cited sentences over long unsupported prose.
"""

_SUMMARY_SYSTEM_PROMPT = """You are a research assistant summarizing a single academic paper \
using ONLY the provided excerpts from that paper. Produce a structured summary with these \
exact sections (use markdown headers):

## Methodology
## Dataset
## Key Results
## Limitations

If a section cannot be determined from the provided excerpts, write "Not stated in the \
retrieved excerpts" for that section instead of guessing. Cite the paper_id in brackets \
after each claim, e.g. [2005.11401]."""

_COMPARE_SYSTEM_PROMPT = """You are a research assistant comparing multiple academic papers \
using ONLY the provided excerpts. Produce a single markdown table with columns:

| Paper | Method | Dataset | Model | Results | Limitations |

One row per paper, using the paper's arXiv ID to identify it. If information for a cell is \
not present in the provided excerpts, write "Not stated". Do not invent details. After the \
table, add a short bullet-point list of the most notable differences, citing [paper_id] for \
each claim."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its similarity score."""

    paper_id: str
    title: str
    text: str
    score: float
    source_url: str


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks: List of chunk dicts as returned by VectorStore.search().

    Returns:
        A single string with one block per chunk, each tagged with its
        paper_id and title so the model can cite it correctly.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} | paper_id={c['paper_id']} | title=\"{c['title']}\"]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def _generate(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Single chokepoint for all LLM calls. Swap providers by editing only this function.

    Args:
        system_prompt: Instructions establishing the model's role and constraints.
        user_prompt: The task-specific content (context + question).
        temperature: Sampling temperature; kept low for grounded, deterministic answers.

    Returns:
        The generated text response.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is missing, or if generation fails
            after all retries are exhausted.
    """
    api_key = require_google_api_key()
    # transport="rest" avoids gRPC, which can hang indefinitely on networks/
    # containers that block or throttle long-lived HTTP/2 connections (seen
    # in some sandboxed/serverless environments). REST is slightly slower
    # per-call but fails fast and predictably instead of hanging forever.
    genai.configure(api_key=api_key, transport="rest")

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text.strip()
        except Exception as exc:  # noqa: BLE001 - broad on purpose for retry wrapper
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    logger.error("Gemini generation failed after %d attempts", MAX_RETRIES, exc_info=True)
    raise RuntimeError(
        f"LLM generation failed after {MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def verify_citations(answer: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that every [paper_id] citation in the answer maps to a retrieved chunk.

    This is a lightweight guardrail (not a semantic fact-checker): it
    confirms the model didn't cite a paper_id that wasn't actually in the
    retrieved context, which is a common hallucination pattern.

    Args:
        answer: The LLM-generated answer text.
        retrieved_chunks: The chunks that were actually passed to the model.

    Returns:
        A dict with:
          - cited_ids: sorted list of paper_ids cited in the answer
          - valid_ids: those citations that match a retrieved chunk
          - invalid_ids: citations that do NOT match any retrieved chunk
          - is_grounded: True if there are no invalid citations
          - declined: True if the model declined to answer (not enough info)
    """
    cited_ids = sorted(set(re.findall(r"\[([^\[\]]+?)\]", answer)))
    retrieved_ids = {c["paper_id"] for c in retrieved_chunks}
    valid_ids = [cid for cid in cited_ids if cid in retrieved_ids]
    invalid_ids = [cid for cid in cited_ids if cid not in retrieved_ids]
    declined = _NOT_ENOUGH_INFO in answer.lower()

    return {
        "cited_ids": cited_ids,
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
        "is_grounded": len(invalid_ids) == 0,
        "declined": declined,
    }


def ask(question: str, k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Answer a question using citation-grounded retrieval-augmented generation.

    Args:
        question: The user's natural-language question.
        k: Number of chunks to retrieve as context.

    Returns:
        A dict with keys: answer, sources (list of {paper_id, title,
        source_url, score}), citation_check (see verify_citations).
    """
    store = get_vector_store()
    retrieved = store.search(question, k=k)

    if not retrieved:
        return {
            "answer": f"I could not retrieve any relevant context; {_NOT_ENOUGH_INFO}.",
            "sources": [],
            "citation_check": {
                "cited_ids": [],
                "valid_ids": [],
                "invalid_ids": [],
                "is_grounded": True,
                "declined": True,
            },
        }

    context = _format_context(retrieved)
    user_prompt = (
        f"Context excerpts from academic papers:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question following all the rules in your system instructions."
    )
    answer = _generate(_QA_SYSTEM_PROMPT, user_prompt)
    citation_check = verify_citations(answer, retrieved)

    seen_paper_ids: set[str] = set()
    sources = []
    for c in retrieved:
        if c["paper_id"] in seen_paper_ids:
            continue
        seen_paper_ids.add(c["paper_id"])
        sources.append(
            {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "source_url": c["source_url"],
                "score": c["score"],
            }
        )

    return {"answer": answer, "sources": sources, "citation_check": citation_check}


def summarize_paper(paper_id: str, k: int = 12) -> dict[str, Any]:
    """Generate a structured summary of a single paper from its retrieved chunks.

    Args:
        paper_id: The arXiv ID of the paper to summarize.
        k: Number of chunks to retrieve for this paper as context.

    Returns:
        A dict with keys: paper_id, title, summary (markdown text with
        Methodology / Dataset / Key Results / Limitations sections).

    Raises:
        ValueError: If no chunks exist for the given paper_id.
    """
    store = get_vector_store()
    paper_chunks = [c for c in store.metadata if c["paper_id"] == paper_id]
    if not paper_chunks:
        raise ValueError(f"No ingested chunks found for paper_id={paper_id!r}")

    paper_chunks = paper_chunks[:k]
    title = paper_chunks[0]["title"]
    context = _format_context(paper_chunks)
    user_prompt = (
        f'Excerpts from the paper (paper_id={paper_id}, title="{title}"):\n\n{context}\n\n'
        "Write the structured summary now."
    )
    summary = _generate(_SUMMARY_SYSTEM_PROMPT, user_prompt)
    return {"paper_id": paper_id, "title": title, "summary": summary}


def compare_papers(paper_ids: list[str], k_per_paper: int = 6) -> dict[str, Any]:
    """Generate a markdown comparison table across 2-5 papers.

    Args:
        paper_ids: List of 2-5 arXiv paper IDs to compare.
        k_per_paper: Number of chunks to include per paper as context.

    Returns:
        A dict with keys: paper_ids, comparison (markdown table + notes).

    Raises:
        ValueError: If fewer than 2 or more than 5 paper_ids are given, or
            if any paper_id has no ingested chunks.
    """
    if not (2 <= len(paper_ids) <= 5):
        raise ValueError("compare_papers requires between 2 and 5 paper_ids")

    store = get_vector_store()
    all_titles: dict[str, str] = {}
    context_blocks = []

    for pid in paper_ids:
        paper_chunks = [c for c in store.metadata if c["paper_id"] == pid][:k_per_paper]
        if not paper_chunks:
            raise ValueError(f"No ingested chunks found for paper_id={pid!r}")
        all_titles[pid] = paper_chunks[0]["title"]
        context_blocks.append(_format_context(paper_chunks))

    context = "\n\n---\n\n".join(context_blocks)
    papers_list = "\n".join(f'- {pid}: "{all_titles[pid]}"' for pid in paper_ids)
    user_prompt = (
        f"Papers to compare:\n{papers_list}\n\n"
        f"Excerpts:\n\n{context}\n\n"
        "Produce the comparison table and notes now."
    )
    comparison = _generate(_COMPARE_SYSTEM_PROMPT, user_prompt)
    return {"paper_ids": paper_ids, "titles": all_titles, "comparison": comparison}


def list_papers() -> list[dict[str, Any]]:
    """Return metadata for all ingested papers (for the /papers endpoint and UI).

    Returns:
        List of paper metadata dicts as saved by ingest.save_artifacts().
    """
    return load_papers()
