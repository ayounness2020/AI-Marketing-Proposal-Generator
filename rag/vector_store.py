"""
Vector Store
============
Wraps FAISS to provide a simple add / search / persist API.

Stores:
- FAISS index (vectors)               → data/faiss/index.faiss
- Chunk texts + metadata (JSON)       → data/faiss/metadata.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

import config
from rag.section_chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with disk persistence."""

    def __init__(
        self,
        dimension: int = config.EMBEDDING_DIMENSION,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.FAISS_METADATA_PATH,
    ) -> None:
        self._dimension = dimension
        self._index_path = index_path
        self._metadata_path = metadata_path

        # Parallel list: entry i corresponds to FAISS vector at position i
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)  # cosine sim

        if self._index_path.exists() and self._metadata_path.exists():
            self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: Chunk objects (text + metadata).
            embeddings: One embedding vector per chunk.
        """
        if not chunks:
            return

        matrix = np.array(embeddings, dtype=np.float32)
        # L2-normalise so IndexFlatIP gives cosine similarity scores
        faiss.normalize_L2(matrix)

        self._index.add(matrix)
        self._texts.extend(c.text for c in chunks)
        self._metadatas.extend(c.metadata for c in chunks)

        logger.info("Added %d chunks. Total: %d.", len(chunks), self.total_chunks)
        self._persist()

    def search(
        self, query_embedding: list[float], top_k: int = config.TOP_K
    ) -> list[dict]:
        """
        Retrieve the top-k most similar chunks.

        Args:
            query_embedding: Embedding of the query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata, score.
        """
        if self.total_chunks == 0:
            return []

        k = min(top_k, self.total_chunks)
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        scores, indices = self._index.search(query, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "text": self._texts[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(score),
                }
            )

        return results

    def clear(self) -> None:
        """Wipe the store and delete persisted files."""
        self._index = faiss.IndexFlatIP(self._dimension)
        self._texts = []
        self._metadatas = []

        for path in (self._index_path, self._metadata_path):
            if path.exists():
                path.unlink()

        logger.info("Vector store cleared.")

    def get_document_stats(self) -> dict:
        """Return summary statistics about the stored documents."""
        if not self._metadatas:
            return {"total_chunks": 0, "documents": {}, "document_types": {}}

        documents: dict[str, int] = {}
        doc_types: dict[str, int] = {}
        for meta in self._metadatas:
            src = meta.get("source_file", "unknown")
            dt = meta.get("document_type", "unknown")
            documents[src] = documents.get(src, 0) + 1
            doc_types[dt] = doc_types.get(dt, 0) + 1

        return {
            "total_chunks": self.total_chunks,
            "documents": documents,
            "document_types": doc_types,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        payload = {"texts": self._texts, "metadatas": self._metadatas}
        with open(self._metadata_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.debug("Vector store persisted.")

    def _load(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            self._index = faiss.read_index(str(self._index_path))
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._texts = payload["texts"]
            self._metadatas = payload["metadatas"]
            logger.info("Loaded %d chunks from disk.", self.total_chunks)
        except Exception as exc:
            logger.error("Failed to load vector store: %s. Starting fresh.", exc)
            self._index = faiss.IndexFlatIP(self._dimension)
            self._texts = []
            self._metadatas = []
