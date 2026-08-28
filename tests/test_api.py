"""Unit tests for backend/api.py — FastAPI endpoints, validation, caching.

All calls into backend.rag are monkeypatched so these tests never call the
real Gemini API, download real PDFs, or require a built FAISS index. This
keeps CI fast, free, and deterministic.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.api as api_module


@pytest.fixture(autouse=True)
def reset_cache_and_rate_limit() -> None:
    """Ensure each test starts with a clean cache and rate-limit clock."""
    api_module._ask_cache.clear()
    api_module._last_request_time = 0.0
    yield
    api_module._ask_cache.clear()
    api_module._last_request_time = 0.0


@pytest.fixture()
def client() -> TestClient:
    """Build a TestClient for the FastAPI app."""
    return TestClient(api_module.app)


class TestHealth:
    """Tests for GET /health."""

    def test_health_ok(self, client: TestClient) -> None:
        """Health check should always return 200 with status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPapers:
    """Tests for GET /papers."""

    def test_list_papers(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """The endpoint should return whatever list_papers() provides."""
        fake_papers = [{"paper_id": "1234.5678", "title": "Fake Paper", "extracted": True}]
        monkeypatch.setattr(api_module, "list_papers", lambda: fake_papers)
        resp = client.get("/papers")
        assert resp.status_code == 200
        assert resp.json() == fake_papers


class TestAsk:
    """Tests for POST /ask: validation, caching, rate limiting."""

    def _fake_ask_result(self) -> dict:
        return {
            "answer": "RAG improves factuality [1234.5678].",
            "sources": [
                {
                    "paper_id": "1234.5678",
                    "title": "Fake Paper",
                    "source_url": "http://example.com",
                    "score": 0.9,
                }
            ],
            "citation_check": {
                "cited_ids": ["1234.5678"],
                "valid_ids": ["1234.5678"],
                "invalid_ids": [],
                "is_grounded": True,
                "declined": False,
            },
        }

    def test_ask_rejects_empty_question(self, client: TestClient) -> None:
        """A blank question must fail Pydantic validation with 422."""
        resp = client.post("/ask", json={"question": "", "k": 5})
        assert resp.status_code == 422

    def test_ask_rejects_too_short_question(self, client: TestClient) -> None:
        """A question below min_length must fail validation."""
        resp = client.post("/ask", json={"question": "hi", "k": 5})
        assert resp.status_code == 422

    def test_ask_rejects_k_above_max(self, client: TestClient) -> None:
        """k above MAX_TOP_K must fail validation."""
        resp = client.post("/ask", json={"question": "What datasets were used?", "k": 999})
        assert resp.status_code == 422

    def test_ask_rejects_k_below_min(self, client: TestClient) -> None:
        """k below 1 must fail validation."""
        resp = client.post("/ask", json={"question": "What datasets were used?", "k": 0})
        assert resp.status_code == 422

    def test_ask_success_returns_grounded_answer(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid question should return the mocked ask() result with cached=False."""
        monkeypatch.setattr(api_module, "ask", lambda question, k: self._fake_ask_result())
        resp = client.post("/ask", json={"question": "What datasets were used?", "k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False
        assert "[1234.5678]" in data["answer"]
        assert data["citation_check"]["is_grounded"] is True

    def test_ask_cache_hit_avoids_second_call(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asking the same question twice should hit the cache on the 2nd call."""
        call_count = {"n": 0}

        def fake_ask(question: str, k: int) -> dict:
            call_count["n"] += 1
            return self._fake_ask_result()

        monkeypatch.setattr(api_module, "ask", fake_ask)
        monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_REQUESTS", 0.0)

        payload = {"question": "What datasets were used?", "k": 5}
        resp1 = client.post("/ask", json=payload)
        resp2 = client.post("/ask", json=payload)

        assert resp1.json()["cached"] is False
        assert resp2.json()["cached"] is True
        assert call_count["n"] == 1  # second call served from cache, no real ask() call

    def test_ask_vector_store_not_built_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the vector store isn't built, the API should return a clean 503."""

        def raise_not_found(question: str, k: int):
            raise FileNotFoundError("no index")

        monkeypatch.setattr(api_module, "ask", raise_not_found)
        monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_REQUESTS", 0.0)
        resp = client.post("/ask", json={"question": "What datasets were used?", "k": 5})
        assert resp.status_code == 503

    def test_ask_rate_limit_returns_429(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two distinct questions in rapid succession should trigger the rate limiter."""
        monkeypatch.setattr(api_module, "ask", lambda question, k: self._fake_ask_result())
        monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_REQUESTS", 10.0)

        resp1 = client.post("/ask", json={"question": "First distinct question?", "k": 5})
        resp2 = client.post("/ask", json={"question": "Second distinct question?", "k": 5})

        assert resp1.status_code == 200
        assert resp2.status_code == 429


class TestSummarize:
    """Tests for POST /summarize."""

    def test_summarize_rejects_blank_paper_id(self, client: TestClient) -> None:
        """An empty paper_id must fail validation."""
        resp = client.post("/summarize", json={"paper_id": ""})
        assert resp.status_code == 422

    def test_summarize_success(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid paper_id should return the mocked summary."""
        fake_result = {
            "paper_id": "1234.5678",
            "title": "Fake Paper",
            "summary": "## Methodology\n...",
        }
        monkeypatch.setattr(api_module, "summarize_paper", lambda paper_id: fake_result)
        resp = client.post("/summarize", json={"paper_id": "1234.5678"})
        assert resp.status_code == 200
        assert resp.json() == fake_result

    def test_summarize_unknown_paper_returns_404(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown paper_id should surface as a 404, not a 500."""

        def raise_value_error(paper_id: str):
            raise ValueError(f"No ingested chunks found for paper_id={paper_id!r}")

        monkeypatch.setattr(api_module, "summarize_paper", raise_value_error)
        resp = client.post("/summarize", json={"paper_id": "nonexistent"})
        assert resp.status_code == 404


class TestCompare:
    """Tests for POST /compare."""

    def test_compare_rejects_single_paper(self, client: TestClient) -> None:
        """Fewer than 2 paper_ids must fail validation."""
        resp = client.post("/compare", json={"paper_ids": ["1234.5678"]})
        assert resp.status_code == 422

    def test_compare_rejects_more_than_five(self, client: TestClient) -> None:
        """More than 5 paper_ids must fail validation."""
        resp = client.post("/compare", json={"paper_ids": [f"id{i}" for i in range(6)]})
        assert resp.status_code == 422

    def test_compare_rejects_duplicate_ids(self, client: TestClient) -> None:
        """Duplicate paper_ids must fail validation."""
        resp = client.post("/compare", json={"paper_ids": ["a", "a"]})
        assert resp.status_code == 422

    def test_compare_success(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid 2-5 paper_id list should return the mocked comparison table."""
        fake_result = {
            "paper_ids": ["a", "b"],
            "titles": {"a": "Paper A", "b": "Paper B"},
            "comparison": "| Paper | Method |\n|---|---|\n| a | X |\n| b | Y |",
        }
        monkeypatch.setattr(api_module, "compare_papers", lambda paper_ids: fake_result)
        resp = client.post("/compare", json={"paper_ids": ["a", "b"]})
        assert resp.status_code == 200
        assert resp.json() == fake_result
