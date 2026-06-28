"""
Embedding Service
=================
Uses Google Gemini Embedding API — no local models needed.
Model: models/text-embedding-004 (768 dimensions)
"""

from __future__ import annotations

import logging
import os
import google.generativeai as genai
import config

logger = logging.getLogger(__name__)

_MODEL_NAME = "models/text-embedding-004"


class EmbeddingService:

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY", config.GEMINI_API_KEY)
        genai.configure(api_key=api_key)
        logger.info("Gemini embedding service ready: %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self._model_name,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self._model_name,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self._model_name,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]
