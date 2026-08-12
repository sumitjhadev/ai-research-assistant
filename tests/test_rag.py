"""Unit tests for backend/rag.py — citation verification and prompt formatting.

The Gemini API is never called in these tests; `_generate` is monkeypatched
so CI never needs a real GOOGLE_API_KEY or spends real API credits.
"""

from __future__ import annotations

import pytest

import backend.rag as rag_module
from backend.rag import _format_context, verify_citations


def sample_retrieved_chunks() -> list[dict]:
    """Build a small fixed set of retrieved chunks for citation tests."""
    return [
        {
            "paper_id": "2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
            "text": "RAG combines a retriever with a seq2seq generator.",
            "source_url": "http://arxiv.org/pdf/2005.11401",
            "score": 0.91,
        },
        {
            "paper_id": "2004.04906",
            "title": "Dense Passage Retrieval",
            "text": "DPR uses dual encoders trained with contrastive loss.",
            "source_url": "http://arxiv.org/pdf/2004.04906",
            "score": 0.83,
        },
    ]


class TestFormatContext:
    """Tests for the context-formatting helper used in all prompts."""

    def test_includes_paper_id_and_title_for_each_chunk(self) -> None:
        """Every chunk block must expose paper_id and title for citation."""
        chunks = sample_retrieved_chunks()
        context = _format_context(chunks)
        for c in chunks:
            assert c["paper_id"] in context
            assert c["title"] in context
            assert c["text"] in context

    def test_empty_chunks_returns_empty_string(self) -> None:
        """No chunks should format to an empty context block."""
        assert _format_context([]) == ""


class TestVerifyCitations:
    """Tests for the citation-grounding guardrail."""

    def test_all_valid_citations_are_grounded(self) -> None:
        """An answer citing only retrieved paper_ids should be marked grounded."""
        chunks = sample_retrieved_chunks()
        answer = "RAG improves factuality [2005.11401] via dense retrieval [2004.04906]."
        check = verify_citations(answer, chunks)

        assert check["is_grounded"] is True
        assert check["invalid_ids"] == []
        assert set(check["valid_ids"]) == {"2005.11401", "2004.04906"}
        assert check["declined"] is False

    def test_invalid_citation_is_flagged(self) -> None:
        """A citation to a paper_id NOT in the retrieved context must be flagged."""
        chunks = sample_retrieved_chunks()
        answer = "This claim is unsupported [9999.99999]."
        check = verify_citations(answer, chunks)

        assert check["is_grounded"] is False
        assert "9999.99999" in check["invalid_ids"]

    def test_declined_answer_is_detected(self) -> None:
        """The exact refusal phrase must be detected case-insensitively."""
        chunks = sample_retrieved_chunks()
        answer = "There is not enough information in the corpus to answer this."
        check = verify_citations(answer, chunks)

        assert check["declined"] is True

    def test_no_citations_at_all(self) -> None:
        """An answer with zero bracketed citations should report empty lists."""
        chunks = sample_retrieved_chunks()
        answer = "This answer has no citations at all."
        check = verify_citations(answer, chunks)

        assert check["cited_ids"] == []
        assert check["is_grounded"] is True  # vacuously true: no invalid citations either


class TestGenerateIsMockable:
    """Confirms _generate() is a single chokepoint that tests can monkeypatch."""

    def test_generate_can_be_monkeypatched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatching _generate must fully bypass the real Gemini SDK call."""

        def fake_generate(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
            return "mocked answer [2005.11401]"

        monkeypatch.setattr(rag_module, "_generate", fake_generate)
        result = rag_module._generate("sys", "user")
        assert result == "mocked answer [2005.11401]"
