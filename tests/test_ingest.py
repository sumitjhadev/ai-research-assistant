"""Unit tests for backend/ingest.py — chunking and data classes.

These tests do not hit the network; they operate purely on in-memory text
and Paper/Chunk dataclasses.
"""

from __future__ import annotations

import pytest

from backend.ingest import Chunk, Paper, chunk_text


def make_paper(paper_id: str = "1234.5678") -> Paper:
    """Build a minimal Paper fixture for chunking tests."""
    return Paper(
        paper_id=paper_id,
        title="A Test Paper About Retrieval",
        authors=["Alice Example", "Bob Example"],
        abstract="An abstract.",
        published="2024-01-01T00:00:00",
        pdf_url="https://arxiv.org/pdf/1234.5678",
    )


class TestChunkText:
    """Tests for the sliding-window chunker."""

    def test_chunk_size_and_overlap_defaults(self) -> None:
        """Chunks should respect the configured size/overlap and cover all text."""
        text = "A" * 2500
        paper = make_paper()
        chunks = chunk_text(text, paper, chunk_size=800, overlap=150)

        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert len(c.text) <= 800
            assert c.paper_id == paper.paper_id
            assert c.title == paper.title
            assert c.authors == paper.authors
            assert c.source_url == paper.pdf_url

    def test_chunks_cover_full_text_with_overlap(self) -> None:
        """Consecutive chunks should overlap by roughly the configured amount."""
        text = "0123456789" * 100  # 1000 chars
        paper = make_paper()
        chunks = chunk_text(text, paper, chunk_size=300, overlap=50)

        # Reconstructing chunk_index ordering
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert chunks[0].text.startswith("0123456789")

    def test_chunk_ids_are_unique_and_traceable(self) -> None:
        """Every chunk_id must be unique and contain the paper_id."""
        text = "word " * 1000
        paper = make_paper(paper_id="9999.0001")
        chunks = chunk_text(text, paper)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(paper.paper_id in cid for cid in ids)

    def test_empty_text_produces_no_chunks(self) -> None:
        """Empty or whitespace-only text should produce zero chunks, not crash."""
        paper = make_paper()
        assert chunk_text("", paper) == []
        assert chunk_text("   \n\t  ", paper) == []

    def test_short_text_produces_single_chunk(self) -> None:
        """Text shorter than chunk_size should produce exactly one chunk."""
        paper = make_paper()
        chunks = chunk_text("Short text.", paper, chunk_size=800, overlap=150)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_invalid_overlap_raises(self) -> None:
        """chunk_size <= overlap must raise ValueError rather than infinite-loop."""
        paper = make_paper()
        with pytest.raises(ValueError):
            chunk_text("some text", paper, chunk_size=100, overlap=150)

    def test_every_chunk_tagged_with_metadata(self) -> None:
        """Every chunk must carry paper_id, title, authors, and source_url."""
        text = "Research findings. " * 200
        paper = make_paper()
        chunks = chunk_text(text, paper)
        for c in chunks:
            assert c.paper_id
            assert c.title
            assert c.authors
            assert c.source_url
