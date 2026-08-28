"""Embedding generation and FAISS vector store for semantic retrieval.

Pipeline stage: Chunking -> Embedding Generation -> FAISS Vector Store
                 -> Semantic Retrieval

Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings and a FAISS
IndexFlatIP index over L2-normalized vectors, which is mathematically
equivalent to cosine-similarity search.

Independently runnable:

    python backend/vectorstore.py build
    python backend/vectorstore.py search "what datasets were used" --k 5
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    MAX_TOP_K,
    get_logger,
)
from backend.ingest import load_chunks

logger = get_logger(__name__)

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load (and cache) the sentence-transformers embedding model.

    Returns:
        A shared SentenceTransformer instance, loaded once per process.
    """
    global _model
    with _model_lock:
        if _model is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and L2-normalize the resulting vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        A (len(texts), dim) float32 numpy array, L2-normalized row-wise so
        that inner product == cosine similarity.
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain only non-empty strings")

    model = get_embedding_model()
    embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
    ).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("embedding model returned an invalid shape")
    faiss.normalize_L2(embeddings)
    return embeddings


class VectorStore:
    """A FAISS IndexFlatIP-backed store over chunk embeddings with metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build the FAISS index from scratch given a list of chunk dicts.

        Args:
            chunks: List of chunk dicts (from ingest.load_chunks()), each
                containing at least a "text" field plus citation metadata.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks. Run ingest.py first.")

        logger.info("Embedding %d chunks with %s", len(chunks), EMBEDDING_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.metadata = chunks
        logger.info("Built FAISS IndexFlatIP with %d vectors of dim %d", index.ntotal, dim)

    def save(
        self, index_path: str = str(FAISS_INDEX_PATH), metadata_path: str = str(FAISS_METADATA_PATH)
    ) -> None:
        """Persist the FAISS index and its metadata sidecar to disk.

        Args:
            index_path: File path to write the FAISS index to.
            metadata_path: File path to write the JSONL metadata sidecar to.

        Raises:
            RuntimeError: If build() has not been called yet.
        """
        if self.index is None:
            raise RuntimeError("Cannot save an unbuilt vector store. Call build() first.")
        from pathlib import Path

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info("Saved FAISS index to %s and metadata to %s", index_path, metadata_path)

    def load(
        self, index_path: str = str(FAISS_INDEX_PATH), metadata_path: str = str(FAISS_METADATA_PATH)
    ) -> None:
        """Load a previously saved FAISS index and its metadata sidecar.

        Args:
            index_path: Path to the saved FAISS index file.
            metadata_path: Path to the saved JSONL metadata sidecar.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        import os

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Vector store not found at {index_path} / {metadata_path}. "
                "Run `python backend/vectorstore.py build` first."
            )
        loaded_index = faiss.read_index(index_path)
        self.metadata = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        if loaded_index.ntotal != len(self.metadata):
            raise ValueError(
                "Vector index and metadata are out of sync: "
                f"{loaded_index.ntotal} vectors but {len(self.metadata)} metadata records"
            )
        self.index = loaded_index
        logger.info("Loaded FAISS index with %d vectors from %s", self.index.ntotal, index_path)

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Run semantic search for a query and return the top-k chunks.

        Args:
            query: Natural-language query string.
            k: Number of results to return (clamped to [1, MAX_TOP_K]).

        Returns:
            List of result dicts, each containing the original chunk
            metadata plus a "score" field (cosine similarity, higher is
            better).

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self.index is None:
            raise RuntimeError("Vector store is not loaded. Call build() or load() first.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = max(1, min(k, MAX_TOP_K, self.index.ntotal))

        query_vec = embed_texts([query])
        scores, indices = self.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results


_store_lock = threading.Lock()
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Lazily load (and cache) the singleton VectorStore for the process.

    Returns:
        A VectorStore loaded from disk artifacts.

    Raises:
        FileNotFoundError: If no vector store has been built yet.
    """
    global _store
    with _store_lock:
        if _store is None:
            store = VectorStore()
            store.load()
            _store = store
        return _store


def build_command() -> None:
    """CLI subcommand: build and save the vector store from ingested chunks."""
    chunks = load_chunks()
    if not chunks:
        logger.error(
            'No chunks found. Run `python backend/ingest.py --query "..."` before building.'
        )
        raise SystemExit(1)
    store = VectorStore()
    store.build(chunks)
    store.save()


def search_command(query: str, k: int) -> None:
    """CLI subcommand: load the vector store and print top-k search results.

    Args:
        query: Query string to search for.
        k: Number of results to print.
    """
    store = VectorStore()
    store.load()
    results = store.search(query, k=k)
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] score={r['score']:.4f} paper_id={r['paper_id']} title={r['title'][:80]}")
        print(f"    {r['text'][:200]}...")


def main() -> None:
    """CLI entrypoint supporting `build` and `search` subcommands."""
    parser = argparse.ArgumentParser(description="Build or query the FAISS vector store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build the vector store from ingested chunks.")

    search_parser = subparsers.add_parser("search", help="Search the vector store.")
    search_parser.add_argument("query", type=str, help="Query text")
    search_parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of results")

    args = parser.parse_args()

    if args.command == "build":
        build_command()
    elif args.command == "search":
        search_command(args.query, args.k)


if __name__ == "__main__":
    main()
