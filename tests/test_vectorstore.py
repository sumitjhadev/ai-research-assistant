"""Unit tests for backend/vectorstore.py — FAISS build/search behavior.

The real sentence-transformers model is NOT downloaded in these tests
(that would be slow and network-dependent in CI). Instead, `embed_texts`
is monkeypatched with a deterministic, dependency-free embedding function
so we can test FAISS indexing/search logic in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.vectorstore as vectorstore_module
from backend.vectorstore import VectorStore


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words-ish fake embedding: no model download needed.

    Encodes each text as a small vector derived from character codes so that
    similar strings produce similar (but not identical) vectors, then
    L2-normalizes exactly like the real pipeline does.
    """
    dim = 16
    vectors = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for ch in text.lower():
            vectors[i, ord(ch) % dim] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real embedding call with the deterministic fake for all tests."""
    monkeypatch.setattr(vectorstore_module, "embed_texts", _fake_embed_texts)


def sample_chunks() -> list[dict]:
    """Build a small fixed corpus of chunk dicts for retrieval tests."""
    return [
        {
            "chunk_id": "p1::chunk0",
            "paper_id": "p1",
            "title": "Retrieval Augmented Generation for NLP",
            "authors": ["A"],
            "source_url": "http://example.com/p1",
            "text": "retrieval augmented generation improves factual accuracy",
            "chunk_index": 0,
        },
        {
            "chunk_id": "p2::chunk0",
            "paper_id": "p2",
            "title": "Transformer Efficiency Techniques",
            "authors": ["B"],
            "source_url": "http://example.com/p2",
            "text": "sparse attention reduces transformer compute cost",
            "chunk_index": 0,
        },
        {
            "chunk_id": "p3::chunk0",
            "paper_id": "p3",
            "title": "Vector Database Benchmarking",
            "authors": ["C"],
            "source_url": "http://example.com/p3",
            "text": "benchmarking vector databases for approximate nearest neighbor search",
            "chunk_index": 0,
        },
    ]


class TestVectorStore:
    """Tests for VectorStore.build / save / load / search."""

    def test_build_creates_index_with_correct_count(self) -> None:
        """After build(), the FAISS index should contain one vector per chunk."""
        store = VectorStore()
        chunks = sample_chunks()
        store.build(chunks)
        assert store.index is not None
        assert store.index.ntotal == len(chunks)
        assert len(store.metadata) == len(chunks)

    def test_build_raises_on_empty_chunks(self) -> None:
        """Building from zero chunks should raise a clear ValueError."""
        store = VectorStore()
        with pytest.raises(ValueError):
            store.build([])

    def test_search_returns_ranked_results_with_scores(self) -> None:
        """search() should return up to k results, each with a similarity score."""
        store = VectorStore()
        store.build(sample_chunks())

        results = store.search("retrieval augmented generation", k=2)
        assert len(results) == 2
        assert all("score" in r for r in results)
        # Results should be sorted by descending score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_k_clamping(self) -> None:
        """k should be clamped into [1, MAX_TOP_K], never crash on extreme values."""
        store = VectorStore()
        store.build(sample_chunks())

        results_zero = store.search("test query", k=0)
        assert len(results_zero) >= 1  # clamped to at least 1

        results_huge = store.search("test query", k=10_000)
        assert len(results_huge) <= len(sample_chunks())

    def test_search_before_load_raises(self) -> None:
        """Calling search() on an unbuilt/unloaded store must raise RuntimeError."""
        store = VectorStore()
        with pytest.raises(RuntimeError):
            store.search("anything")

    def test_save_before_build_raises(self, tmp_path) -> None:
        """Calling save() before build() must raise RuntimeError, not crash silently."""
        store = VectorStore()
        with pytest.raises(RuntimeError):
            store.save(
                index_path=str(tmp_path / "idx.faiss"),
                metadata_path=str(tmp_path / "meta.jsonl"),
            )

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """A saved index should reload with identical vector count and metadata."""
        store = VectorStore()
        store.build(sample_chunks())
        index_path = tmp_path / "idx.faiss"
        metadata_path = tmp_path / "meta.jsonl"
        store.save(index_path=str(index_path), metadata_path=str(metadata_path))

        loaded = VectorStore()
        loaded.load(index_path=str(index_path), metadata_path=str(metadata_path))
        assert loaded.index.ntotal == store.index.ntotal
        assert len(loaded.metadata) == len(store.metadata)

    def test_load_missing_files_raises(self, tmp_path) -> None:
        """Loading from nonexistent paths should raise FileNotFoundError."""
        store = VectorStore()
        with pytest.raises(FileNotFoundError):
            store.load(
                index_path=str(tmp_path / "nope.faiss"),
                metadata_path=str(tmp_path / "nope.jsonl"),
            )
