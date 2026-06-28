"""
Gemini Service
==============
Uses Gemini REST API directly — no gRPC, no Google auth issues.
"""

from __future__ import annotations

import logging
import os
import requests
import streamlit as st
import config

logger = logging.getLogger(__name__)

_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_PROPOSAL_SYSTEM_PROMPT = """You are an expert marketing consultant and proposal writer with 20+ years of experience.
Generate a professional, customized marketing proposal based ONLY on the context provided.
Never fabricate pricing, timelines, or case study results.
Structure the proposal with clear sections using markdown headings.
Write in a confident, professional tone. You can respond in Arabic or English.
OUTPUT: Return a complete proposal in markdown format."""

_CHAT_SYSTEM_PROMPT = """You are a helpful marketing assistant. Use the provided document excerpts to answer questions.
Be helpful and informative. You can respond in Arabic or English."""


class GeminiService:

    def __init__(self) -> None:
        self._api_key = (
            st.secrets.get("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or config.GEMINI_API_KEY
        )
        self._model_name = config.GEMINI_MODEL
        self._url = _API_URL.format(model=self._model_name)
        logger.info("Gemini service ready (REST): %s", self._model_name)

    @property
    def model(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate_response(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
        }
        try:
            resp = requests.post(
                self._url,
                json=payload,
                params={"key": self._api_key},
                timeout=55,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            return f"⚠️ Error: {exc}"

    def generate_proposal(self, context, client_name, industry, budget, goals, services):
        prompt = f"""
## CLIENT BRIEF
- Client: {client_name}
- Industry: {industry}
- Budget: {budget}
- Goals: {goals}
- Services: {services}

## CONTEXT
{context}

## TASK
Generate a complete professional marketing proposal for {client_name} with sections:
1. Executive Summary
2. Business Objectives
3. Recommended Services
4. Scope of Work
5. Timeline
6. Pricing Recommendation
7. KPIs
8. Next Steps
"""
        return self.generate_response(prompt, system=_PROPOSAL_SYSTEM_PROMPT)

    def chat_with_documents(self, question: str, context: str) -> str:
        if not context.strip():
            return "The requested information could not be found in the uploaded documents."
        prompt = f"## DOCUMENTS\n{context}\n\n## QUESTION\n{question}\n\nAnswer using only the documents above."
        return self.generate_response(prompt, system=_CHAT_SYSTEM_PROMPT)
