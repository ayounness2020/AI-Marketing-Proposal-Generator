"""
Embedding Service
=================
Uses TF-IDF vectorization — no API needed, works on any platform.
Dimension: 768 (to match existing FAISS index config)
"""

from __future__ import annotations

import logging
import hashlib
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

DIMENSION = 768


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def _hash_embed(text: str, dim: int = DIMENSION) -> list[float]:
    """Fast hash-based embedding — deterministic, no API needed."""
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim

    vec = [0.0] * dim
    counts = Counter(tokens)
    total = sum(counts.values())

    for token, count in counts.items():
        # Use multiple hash functions for better distribution
        for seed in range(3):
            h = hashlib.md5(f"{seed}:{token}".encode()).hexdigest()
            idx = int(h[:8], 16) % dim
            sign = 1 if int(h[8], 16) % 2 == 0 else -1
            tfidf = (count / total) * math.log(1 + 1 / (count / total))
            vec[idx] += sign * tfidf

    # Normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingService:

    def __init__(self, model_name: str = "tfidf-hash-768") -> None:
        self._model_name = model_name
        logger.info("Embedding service ready: %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        return _hash_embed(text)

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding %d chunks...", len(texts))
        return [_hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_embed(text)
