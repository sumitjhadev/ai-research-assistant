"""FastAPI application exposing the RAG pipeline as HTTP endpoints.

Endpoints:
    POST /ask        - citation-grounded question answering
    POST /summarize   - structured per-paper summary
    POST /compare     - markdown comparison table across 2-5 papers
    GET  /papers      - list ingested paper metadata
    GET  /health      - liveness/readiness check

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.config import DEFAULT_TOP_K, MAX_TOP_K, get_logger
from backend.rag import ask, compare_papers, list_papers, summarize_paper

logger = get_logger(__name__)

app = FastAPI(
    title="AI Research Assistant API",
    description="Citation-grounded RAG over arXiv papers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Simple in-memory cache + rate limiting for /ask
#
# Repeated identical questions are served from cache instead of re-calling
# the Gemini API, saving credits. This is intentionally a plain dict guarded
# by a lock (no external cache dependency) since the deployment target is a
# single small container/instance.
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_MIN_SECONDS_BETWEEN_REQUESTS = 1.0  # basic rate-limit guard per-process

_ask_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()
_last_request_time = 0.0
_rate_limit_lock = threading.Lock()


def _cache_key(question: str, k: int) -> str:
    """Build a stable cache key for a given (question, k) pair."""
    raw = f"{question.strip().lower()}|k={k}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> dict[str, Any] | None:
    """Return a cached response if present and not expired."""
    with _cache_lock:
        entry = _ask_cache.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if time.time() - cached_at > _CACHE_TTL_SECONDS:
            del _ask_cache[key]
            return None
        return value


def _set_cached(key: str, value: dict[str, Any]) -> None:
    """Store a response in the cache with the current timestamp."""
    with _cache_lock:
        _ask_cache[key] = (time.time(), value)


def _enforce_rate_limit() -> None:
    """Reject requests arriving faster than _MIN_SECONDS_BETWEEN_REQUESTS.

    Raises:
        HTTPException: 429 if called too soon after the previous request.
    """
    global _last_request_time
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests; please wait a moment before asking again.",
            )
        _last_request_time = now


# ---------------------------------------------------------------------------
# Request / response models with input validation
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask.")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of chunks to retrieve.")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        """Reject whitespace-only questions."""
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SummarizeRequest(BaseModel):
    """Request body for POST /summarize."""

    paper_id: str = Field(..., min_length=3, max_length=64, description="arXiv paper ID.")

    @field_validator("paper_id")
    @classmethod
    def paper_id_not_blank(cls, v: str) -> str:
        """Reject whitespace-only paper IDs."""
        if not v.strip():
            raise ValueError("paper_id must not be blank")
        return v.strip()


class CompareRequest(BaseModel):
    """Request body for POST /compare."""

    paper_ids: list[str] = Field(
        ..., min_length=2, max_length=5, description="2-5 arXiv paper IDs to compare."
    )

    @field_validator("paper_ids")
    @classmethod
    def ids_not_blank(cls, v: list[str]) -> list[str]:
        """Reject blank or duplicate paper IDs."""
        cleaned = [pid.strip() for pid in v if pid.strip()]
        if len(cleaned) != len(v):
            raise ValueError("paper_ids must not contain blank entries")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("paper_ids must not contain duplicates")
        return cleaned


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check.

    Returns:
        A simple status payload for uptime monitoring / load balancers.
    """
    return {"status": "ok"}


@app.get("/papers")
def get_papers() -> list[dict[str, Any]]:
    """List metadata for all ingested papers.

    Returns:
        List of paper metadata dicts (paper_id, title, authors, abstract,
        published, pdf_url, extracted).
    """
    return list_papers()


@app.post("/ask")
def post_ask(payload: AskRequest) -> dict[str, Any]:
    """Answer a question with citation-grounded RAG, using cache + rate limiting.

    Args:
        payload: Validated AskRequest (question, k).

    Returns:
        Dict with answer, sources, and citation_check.

    Raises:
        HTTPException: 429 on rate limit, 503 if the vector store is not
            built, 500 on unexpected generation failures.
    """
    key = _cache_key(payload.question, payload.k)
    cached = _get_cached(key)
    if cached is not None:
        logger.info("Cache hit for question=%r", payload.question)
        return {**cached, "cached": True}

    _enforce_rate_limit()

    try:
        result = ask(payload.question, k=payload.k)
    except FileNotFoundError as exc:
        logger.error("Vector store not built: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vector store not built yet. Run ingestion and vectorstore build first.",
        ) from exc
    except RuntimeError as exc:
        logger.error("RAG ask() failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _set_cached(key, result)
    return {**result, "cached": False}


@app.post("/summarize")
def post_summarize(payload: SummarizeRequest) -> dict[str, Any]:
    """Generate a structured summary for a single paper.

    Args:
        payload: Validated SummarizeRequest (paper_id).

    Returns:
        Dict with paper_id, title, summary.

    Raises:
        HTTPException: 404 if the paper_id has no ingested chunks, 503 if
            the vector store is not built, 500 on generation failure.
    """
    try:
        return summarize_paper(payload.paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Vector store not built yet.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("summarize_paper failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def post_compare(payload: CompareRequest) -> dict[str, Any]:
    """Generate a markdown comparison table across 2-5 papers.

    Args:
        payload: Validated CompareRequest (paper_ids).

    Returns:
        Dict with paper_ids, titles, comparison (markdown).

    Raises:
        HTTPException: 404 if any paper_id has no ingested chunks, 503 if
            the vector store is not built, 500 on generation failure.
    """
    try:
        return compare_papers(payload.paper_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Vector store not built yet.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("compare_papers failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
