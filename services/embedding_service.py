"""
Embedding Service
=================
Uses Gemini REST API directly via requests — no gRPC, no Google auth issues.
"""

from __future__ import annotations

import logging
import os
import time
import requests
import streamlit as st
import config

logger = logging.getLogger(__name__)

_MODEL_NAME = "models/text-embedding-004"
_API_URL = "https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"


class EmbeddingService:

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._api_key = (
            st.secrets.get("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or config.GEMINI_API_KEY
        )
        logger.info("Embedding service ready (REST): %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_single(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        text = text[:1500]
        url = _API_URL.format(model=self._model_name)
        payload = {
            "model": self._model_name,
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    params={"key": self._api_key},
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding failed: {e}")

    def embed_text(self, text: str) -> list[float]:
        return self._embed_single(text, "RETRIEVAL_DOCUMENT")

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self._embed_single(text, "RETRIEVAL_DOCUMENT"))
            if (i + 1) % 5 == 0:
                time.sleep(0.3)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text, "RETRIEVAL_QUERY")
