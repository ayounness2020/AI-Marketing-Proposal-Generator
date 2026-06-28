"""
LLM Service
===========
Uses OpenRouter API with google/gemma-4-31b-it:free — supports Arabic & English.
"""

from __future__ import annotations

import logging
import os
import requests
import streamlit as st
import config

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "google/gemma-4-31b-it:free"

_PROPOSAL_SYSTEM = """You are an expert marketing consultant with 20+ years of experience.
Generate a professional, customized marketing proposal based ONLY on the context provided.
Never fabricate pricing, timelines, or case study results.
Structure with clear markdown headings. You can respond in Arabic or English based on the client language."""

_CHAT_SYSTEM = """You are a helpful marketing assistant. Answer using only the provided document excerpts.
You can respond in Arabic or English."""


class GeminiService:

    def __init__(self) -> None:
        self._api_key = (
            st.secrets.get("OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or config.OPENROUTER_API_KEY
        )
        self._model = _MODEL
        logger.info("LLM service ready: %s", self._model)

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate_response(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                _API_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": self._model, "messages": messages},
                timeout=55,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            return f"⚠️ Error: {exc}"

    def generate_proposal(self, context, client_name, industry, budget, goals, services):
        prompt = f"""
## CLIENT BRIEF
- Client: {client_name}
- Industry: {industry}
- Budget: {budget}
- Goals: {goals}
- Services: {services}

## CONTEXT FROM KNOWLEDGE BASE
{context}

## TASK
Generate a complete professional marketing proposal for {client_name} with these sections:
1. Executive Summary
2. Business Objectives
3. Recommended Services
4. Scope of Work
5. Timeline
6. Pricing Recommendation
7. KPIs
8. Next Steps
"""
        return self.generate_response(prompt, system=_PROPOSAL_SYSTEM)

    def chat_with_documents(self, question: str, context: str) -> str:
        if not context.strip():
            return "The requested information could not be found in the uploaded documents."
        prompt = f"## DOCUMENTS\n{context}\n\n## QUESTION\n{question}\n\nAnswer using only the documents above."
        return self.generate_response(prompt, system=_CHAT_SYSTEM)
