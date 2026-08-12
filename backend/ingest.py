"""arXiv search, PDF download, text extraction, and chunking pipeline.

Pipeline stage: User Query -> arXiv Search -> Paper Retrieval -> PDF Download
                 -> Text Extraction -> Chunking (with metadata)

This module is independently runnable:

    python backend/ingest.py --query "retrieval augmented generation" --max-results 20

It writes two artifacts to data/corpus/:
  - papers.json    : list of paper metadata dicts
  - chunks.jsonl    : one JSON object per text chunk, each tagged with
                       paper_id, title, authors, and source_url

A failed PDF download or text extraction for a single paper is logged and
skipped; it never crashes the overall ingestion run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import arxiv
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.config import (
    BACKOFF_BASE_SECONDS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNKS_PATH,
    DEFAULT_ARXIV_QUERY,
    DEFAULT_MAX_RESULTS,
    MAX_RETRIES,
    PAPERS_METADATA_PATH,
    PDF_DIR,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class Paper:
    """Metadata for a single arXiv paper."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    pdf_path: str = ""
    extracted: bool = False


@dataclass
class Chunk:
    """A single text chunk tagged with the metadata needed for citation."""

    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    source_url: str
    text: str
    chunk_index: int = field(default=0)


def search_arxiv(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    """Query the arXiv API and return structured paper metadata.

    Args:
        query: Free-text arXiv search query (e.g. "retrieval augmented generation").
        max_results: Maximum number of papers to fetch.

    Returns:
        A list of Paper objects (title, authors, abstract, published date,
        pdf_url, arXiv id). Does not download PDFs.
    """
    logger.info("Searching arXiv for query=%r max_results=%d", query, max_results)
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search):
            paper_id = result.get_short_id()
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=result.title.strip().replace("\n", " "),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip().replace("\n", " "),
                    published=result.published.isoformat() if result.published else "",
                    pdf_url=result.pdf_url or "",
                )
            )
    except Exception:
        logger.exception("arXiv search failed for query=%r", query)
        raise

    logger.info("Found %d papers for query=%r", len(papers), query)
    return papers


def _retry_with_backoff(fn, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Run fn(*args, **kwargs) with exponential backoff on failure.

    Args:
        fn: Callable to invoke.
        max_retries: Maximum number of attempts before giving up.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception raised by fn if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentionally broad for retry wrapper
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def download_pdf(paper: Paper, dest_dir: Path = PDF_DIR) -> Path | None:
    """Download a paper's PDF with retry/backoff, returning the local path.

    Args:
        paper: Paper metadata including pdf_url.
        dest_dir: Directory to save the PDF into.

    Returns:
        Path to the downloaded PDF, or None if the download failed after
        all retries (the failure is logged, not raised, so the pipeline
        can continue with the next paper).
    """
    dest_path = dest_dir / f"{paper.paper_id.replace('/', '_')}.pdf"
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("PDF already cached for %s", paper.paper_id)
        return dest_path

    def _do_download() -> Path:
        # Fetch the PDF bytes directly via HTTP rather than relying on the
        # arxiv package's download helper, whose API has changed across
        # versions. paper.pdf_url comes straight from the arXiv API result.
        resp = requests.get(paper.pdf_url, timeout=30)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return dest_path

    try:
        return _retry_with_backoff(_do_download)
    except Exception:
        logger.error("Failed to download PDF for %s after retries", paper.paper_id, exc_info=True)
        return None


def extract_text(pdf_path: Path) -> str | None:
    """Extract clean text from a PDF file.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        Extracted text, or None if extraction failed (logged, not raised).
    """
    try:
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                logger.warning("Failed to extract a page from %s; skipping page", pdf_path)
        text = "\n".join(pages_text)
        text = " ".join(text.split())  # collapse whitespace
        if not text.strip():
            logger.warning("Extracted empty text from %s", pdf_path)
            return None
        return text
    except PdfReadError:
        logger.error("Corrupt or unreadable PDF: %s", pdf_path, exc_info=True)
        return None
    except Exception:
        logger.error("Unexpected error extracting text from %s", pdf_path, exc_info=True)
        return None


def chunk_text(
    text: str,
    paper: Paper,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split text into overlapping sliding-window chunks tagged with metadata.

    Args:
        text: Full extracted text of the paper.
        paper: Paper metadata to attach to each chunk (paper_id, title,
            authors, source URL).
        chunk_size: Target chunk length in characters (~800 by default).
        overlap: Overlap between consecutive chunks in characters (~150).

    Returns:
        A list of Chunk objects covering the full text.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}::chunk{index}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=paper.authors,
                    source_url=paper.pdf_url,
                    text=piece,
                    chunk_index=index,
                )
            )
            index += 1
        if end == text_len:
            break
        start += step

    return chunks


def ingest(query: str, max_results: int) -> tuple[list[Paper], list[Chunk]]:
    """Run the full ingestion pipeline: search, download, extract, chunk.

    Args:
        query: arXiv search query.
        max_results: Maximum number of papers to retrieve.

    Returns:
        Tuple of (papers processed, chunks produced). Papers whose PDF
        download or text extraction failed are still included in the
        papers list (with extracted=False) so ingestion is transparent
        about partial failures, but they contribute no chunks.
    """
    papers = search_arxiv(query, max_results)
    all_chunks: list[Chunk] = []

    for i, paper in enumerate(papers, start=1):
        logger.info("[%d/%d] Processing %s: %s", i, len(papers), paper.paper_id, paper.title[:80])
        pdf_path = download_pdf(paper)
        if pdf_path is None:
            logger.warning("Skipping %s: no PDF available", paper.paper_id)
            continue
        paper.pdf_path = str(pdf_path)

        text = extract_text(pdf_path)
        if text is None:
            logger.warning("Skipping %s: text extraction failed", paper.paper_id)
            continue
        paper.extracted = True

        chunks = chunk_text(text, paper)
        logger.info("Produced %d chunks for %s", len(chunks), paper.paper_id)
        all_chunks.extend(chunks)

    return papers, all_chunks


def save_artifacts(papers: list[Paper], chunks: list[Chunk]) -> None:
    """Persist paper metadata and chunks to disk as JSON / JSONL.

    Args:
        papers: All processed papers (including failures, for transparency).
        chunks: All generated chunks across all successfully processed papers.
    """
    PAPERS_METADATA_PATH.write_text(
        json.dumps([asdict(p) for p in papers], indent=2), encoding="utf-8"
    )
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    logger.info(
        "Saved %d papers to %s and %d chunks to %s",
        len(papers),
        PAPERS_METADATA_PATH,
        len(chunks),
        CHUNKS_PATH,
    )


def load_papers() -> list[dict[str, Any]]:
    """Load previously ingested paper metadata from disk.

    Returns:
        List of paper metadata dicts, or an empty list if no ingestion
        has been run yet.
    """
    if not PAPERS_METADATA_PATH.exists():
        return []
    return json.loads(PAPERS_METADATA_PATH.read_text(encoding="utf-8"))


def load_chunks() -> list[dict[str, Any]]:
    """Load previously generated chunks from disk.

    Returns:
        List of chunk dicts, or an empty list if no ingestion has been run.
    """
    if not CHUNKS_PATH.exists():
        return []
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    """CLI entrypoint: python backend/ingest.py --query "..." --max-results 20."""
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the local corpus.")
    parser.add_argument("--query", type=str, default=DEFAULT_ARXIV_QUERY, help="arXiv search query")
    parser.add_argument(
        "--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="Max papers to retrieve"
    )
    args = parser.parse_args()

    papers, chunks = ingest(args.query, args.max_results)
    save_artifacts(papers, chunks)

    succeeded = sum(1 for p in papers if p.extracted)
    logger.info(
        "Ingestion complete: %d/%d papers extracted successfully, %d total chunks",
        succeeded,
        len(papers),
        len(chunks),
    )


if __name__ == "__main__":
    main()
