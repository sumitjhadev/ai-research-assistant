"""Evaluation harness: Recall@5 and retrieval latency over a labeled question set.

Loads data/research_questions.json, runs each question through the FAISS
retriever, and computes:
  - Recall@5: fraction of questions where at least one expected_sources
    paper_id appears in the top-5 retrieved chunks' paper_ids.
  - Average retrieval latency (seconds).

Questions with an empty "expected_sources" list are skipped for the Recall
computation (since there is no ground truth to check against yet) but are
still counted and reported separately, so this script is honest about how
much of the eval set currently has labels.

Results are saved to tests/eval_results.json.

Usage:
    python tests/evaluate.py
    python tests/evaluate.py --k 5 --questions data/research_questions.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from backend.config import DATA_DIR, DEFAULT_TOP_K, get_logger
from backend.vectorstore import VectorStore

logger = get_logger(__name__)

DEFAULT_QUESTIONS_PATH = DATA_DIR / "research_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load the labeled evaluation question set.

    Args:
        path: Path to the research_questions.json file.

    Returns:
        List of question dicts with keys: question, expected_sources, category.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def recall_at_k(expected_sources: list[str], retrieved_paper_ids: list[str]) -> bool:
    """Return True if any expected source paper_id is among the retrieved ones.

    Args:
        expected_sources: Ground-truth relevant paper_ids for this question.
        retrieved_paper_ids: paper_ids of the top-k retrieved chunks.

    Returns:
        True if at least one expected source was retrieved (hit), else False.
    """
    return any(pid in retrieved_paper_ids for pid in expected_sources)


def run_evaluation(questions: list[dict[str, Any]], k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Run retrieval for every question and compute Recall@k and latency stats.

    Args:
        questions: List of question dicts (question, expected_sources, category).
        k: Number of chunks to retrieve per question.

    Returns:
        A results dict summarizing recall, latency, and per-question detail.
    """
    store = VectorStore()
    store.load()

    latencies: list[float] = []
    per_question_results: list[dict[str, Any]] = []
    labeled_hits = 0
    labeled_total = 0
    unlabeled_count = 0

    for item in questions:
        question = item["question"]
        expected_sources = item.get("expected_sources") or []
        category = item.get("category", "uncategorized")

        start = time.perf_counter()
        retrieved = store.search(question, k=k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        retrieved_paper_ids = [r["paper_id"] for r in retrieved]

        hit: bool | None
        if expected_sources:
            labeled_total += 1
            hit = recall_at_k(expected_sources, retrieved_paper_ids)
            if hit:
                labeled_hits += 1
        else:
            unlabeled_count += 1
            hit = None  # no ground truth to score against yet

        per_question_results.append(
            {
                "question": question,
                "category": category,
                "expected_sources": expected_sources,
                "retrieved_paper_ids": retrieved_paper_ids,
                "hit": hit,
                "latency_seconds": round(elapsed, 4),
            }
        )

    recall = (labeled_hits / labeled_total) if labeled_total > 0 else None
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    results = {
        "k": k,
        "total_questions": len(questions),
        "labeled_questions": labeled_total,
        "labeled_hits": labeled_hits,
        "unlabeled_questions": unlabeled_count,
        f"recall_at_{k}": recall,
        "average_latency_seconds": round(avg_latency, 4),
        "p95_latency_seconds": round(
            statistics.quantiles(latencies, n=20)[18]
            if len(latencies) >= 20
            else max(latencies, default=0.0),
            4,
        ),
        "per_question": per_question_results,
    }
    return results


def main() -> None:
    """CLI entrypoint: python tests/evaluate.py [--k 5] [--questions PATH]."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval Recall@k and latency.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-k for retrieval")
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Path to eval question set",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("Question set not found at %s", questions_path)
        raise SystemExit(1)

    questions = load_questions(questions_path)
    logger.info("Loaded %d evaluation questions from %s", len(questions), questions_path)

    try:
        results = run_evaluation(questions, k=args.k)
    except FileNotFoundError:
        logger.error(
            "Vector store not built. Run `python backend/ingest.py` then "
            "`python backend/vectorstore.py build` before evaluating."
        )
        raise SystemExit(1) from None

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    recall_key = f"recall_at_{args.k}"
    if results[recall_key] is not None:
        logger.info(
            "Recall@%d = %.3f (%d/%d labeled questions), avg latency = %.4fs, "
            "%d questions still unlabeled",
            args.k,
            results[recall_key],
            results["labeled_hits"],
            results["labeled_questions"],
            results["average_latency_seconds"],
            results["unlabeled_questions"],
        )
    else:
        logger.warning(
            "No labeled questions (expected_sources) found — Recall@%d could not be computed. "
            "Fill in expected_sources in %s after verifying ground truth against the ingested "
            "corpus. Latency was still measured: avg=%.4fs",
            args.k,
            questions_path,
            results["average_latency_seconds"],
        )
    logger.info("Full results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
