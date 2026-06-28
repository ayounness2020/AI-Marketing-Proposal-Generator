"""
Retriever
=========
Semantic retrieval layer that sits between the vector store
and the generation layer. Responsible for:

- Embedding the query.
- Calling the vector store search.
- Formatting results for consumption by the LLM and the UI.
"""

from __future__ import annotations

import logging

import config
from rag.vector_store import VectorStore
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks for a given natural-language query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = config.TOP_K,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_service
        self._top_k = top_k

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: Natural-language question or topic.

        Returns:
            List of result dicts:
            {
                "text": str,
                "metadata": { source_file, section_name, document_type, chunk_id },
                "score": float   # cosine similarity 0-1
            }
        """
        if self._store.total_chunks == 0:
            logger.warning("Knowledge base is empty. Upload documents first.")
            return []

        query_embedding = self._embedder.embed_text(query)
        results = self._store.search(query_embedding, top_k=self._top_k)

        logger.info("Retrieved %d chunks for query: '%s'.", len(results), query[:80])
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.

        Each chunk is prefixed with its source and section name so the
        model can refer to them accurately.

        Args:
            results: Output of retrieve().

        Returns:
            Formatted multi-line context string.
        """
        if not results:
            return ""

        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            meta = r["metadata"]
            header = (
                f"[{i}] Source: {meta.get('source_file', 'unknown')} | "
                f"Section: {meta.get('section_name', 'unknown')} | "
                f"Type: {meta.get('document_type', 'unknown')}"
            )
            parts.append(f"{header}\n{r['text']}")

        return "\n\n---\n\n".join(parts)
