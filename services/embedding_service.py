"""
Embedding Service
=================
Uses Google Gemini Embedding API with retry and batching.
Model: models/text-embedding-004 (768 dimensions)
"""

from __future__ import annotations

import logging
import os
import time
import google.generativeai as genai
import google.generativeai.client as genai_client
import streamlit as st
import config

logger = logging.getLogger(__name__)

_MODEL_NAME = "models/text-embedding-004"


class EmbeddingService:

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
        genai.configure(api_key=api_key, transport="rest")
        logger.info("Gemini embedding service ready: %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_single(self, text: str, task_type: str = "retrieval_document", retries: int = 3) -> list[float]:
        # Truncate text to avoid timeouts on very long chunks
        text = text[:2000]
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=self._model_name,
                    content=text,
                    task_type=task_type,
                )
                return result["embedding"]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                    logger.warning("Embedding retry %d: %s", attempt + 1, e)
                else:
                    logger.error("Embedding failed after %d retries: %s", retries, e)
                    raise

    def embed_text(self, text: str) -> list[float]:
        return self._embed_single(text, task_type="retrieval_document")

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self._embed_single(text, task_type="retrieval_document")
            embeddings.append(embedding)
            # Small delay every 5 chunks to avoid rate limits
            if (i + 1) % 5 == 0:
                time.sleep(0.5)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text, task_type="retrieval_query")
