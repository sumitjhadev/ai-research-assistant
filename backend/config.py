"""Centralized configuration and logging setup for the AI Research Assistant.

All paths, model names, and tunable parameters live here so the rest of the
codebase never hardcodes a magic string or path. Environment variables are
loaded from a local .env file (never committed) via python-dotenv.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present. Real secrets must never
# be committed; only .env.example (with placeholders) is tracked in git.
load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
PDF_DIR = CORPUS_DIR / "pdfs"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
PAPERS_METADATA_PATH = CORPUS_DIR / "papers.json"
FAISS_INDEX_PATH = CORPUS_DIR / "faiss.index"
FAISS_METADATA_PATH = CORPUS_DIR / "faiss_metadata.jsonl"

for path in (DATA_DIR, CORPUS_DIR, PDF_DIR):
    path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# --------------------------------------------------------------------------
# Chunking configuration
# --------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# --------------------------------------------------------------------------
# Retrieval / generation configuration
# --------------------------------------------------------------------------
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
DEFAULT_ARXIV_QUERY = "retrieval augmented generation"
DEFAULT_MAX_RESULTS = 50

# --------------------------------------------------------------------------
# Retry configuration (exponential backoff)
# --------------------------------------------------------------------------
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.5

# --------------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()


def require_google_api_key() -> str:
    """Return the configured Gemini API key or raise a clear, actionable error.

    Returns:
        The value of the GOOGLE_API_KEY environment variable.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set. This is intentionally a
            clean error message (not a raw stack trace from the SDK) so a
            missing key fails fast and obviously in local dev, CI, and prod.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
            "and set GOOGLE_API_KEY=<your key>, or export it in your shell. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    return GOOGLE_API_KEY


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Create (or fetch) a module-level logger with a consistent format.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A configured logging.Logger instance that writes to stdout with
        timestamps and log levels.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger
