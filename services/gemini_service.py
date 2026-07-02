"""
LLM Service
===========
Uses OpenRouter API with google/gemma-4-31b-it:free — supports Arabic & English.
"""

from __future__ import annotations

import logging
import os
import time
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

    def generate_response(self, prompt: str, system: str = "", max_retries: int = 4) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    _API_URL,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json={"model": self._model, "messages": messages},
                    timeout=55,
                )
                if resp.status_code == 429:
                    wait = (2 ** attempt) + 1
                    logger.warning("Rate limited (429), retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    last_error = "rate_limit"
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if not content:
                    last_error = "empty_response"
                    time.sleep(2)
                    continue
                return content
            except requests.exceptions.Timeout:
                last_error = "timeout"
                logger.warning("Request timed out, attempt %d/%d", attempt + 1, max_retries)
                continue
            except Exception as exc:
                last_error = str(exc)
                logger.error("LLM error: %s", exc)
                time.sleep(1.5)
                continue

        # All retries exhausted — friendly message
        if last_error == "rate_limit":
            return "⚠️ The AI service is currently busy (high demand on the free tier). Please wait a moment and try again."
        elif last_error == "timeout":
            return "⚠️ The request took too long to process. Please try again with a shorter question."
        else:
            return "⚠️ Something went wrong generating a response. Please try again in a moment."

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
        prompt = f"""## DOCUMENT EXCERPTS FROM KNOWLEDGE BASE
{context}

## USER QUESTION
{question}

## INSTRUCTIONS
- Answer the question directly and completely based on the document excerpts above
- Do NOT repeat the question back as your answer
- Summarize and explain the relevant information clearly
- If the documents do not contain enough information, say so honestly
- Respond in the same language as the question (Arabic or English)
- Provide a helpful, detailed answer"""
        return self.generate_response(prompt, system=_CHAT_SYSTEM)
