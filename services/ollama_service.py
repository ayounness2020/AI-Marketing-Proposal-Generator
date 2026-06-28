"""
Ollama Service
==============
Isolated communication layer for the local Ollama server.
"""

from __future__ import annotations

import logging
from typing import Generator

import requests

import config

logger = logging.getLogger(__name__)

_PROPOSAL_SYSTEM_PROMPT = """You are an expert marketing consultant and proposal writer with 20+ years of experience.
Your task is to generate a professional, customized marketing proposal based ONLY on the context provided below.

RULES:
- Use ONLY information from the provided context documents.
- If specific information is not in the context, acknowledge the gap and suggest the client provide it.
- Never fabricate pricing, timelines, or case study results.
- Structure the proposal with clear sections using markdown headings.
- Write in a confident, professional tone suitable for senior business stakeholders.
- Tailor all recommendations to the client's industry and goals.

OUTPUT: Return a complete proposal in markdown format with all requested sections.
"""

_CHAT_SYSTEM_PROMPT = """You are a helpful marketing assistant. Use the provided document excerpts to answer questions. Summarize what you find. Be helpful and informative."""


class OllamaService:

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._generate_url = f"{self._base_url}/api/generate"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.error("Could not list Ollama models: %s", exc)
            return []

    def generate_response(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096},
        }
        try:
            resp = requests.post(self._generate_url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.Timeout:
            return "⚠️ Request timed out. Please try again."
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            return f"⚠️ Error communicating with Ollama: {exc}"

    def generate_proposal(self, context: str, client_name: str, industry: str, budget: str, goals: str, services: str) -> str:
        prompt = f"""
## CLIENT BRIEF
- **Client Name:** {client_name}
- **Industry:** {industry}
- **Budget:** {budget}
- **Business Goals:** {goals}
- **Requested Services:** {services}

## CONTEXT FROM KNOWLEDGE BASE
{context}

## TASK
Generate a complete professional marketing proposal for {client_name}.
Structure with these sections (use ## for each):
1. Executive Summary
2. Business Objectives
3. Recommended Services
4. Scope of Work
5. Deliverables
6. Timeline
7. Pricing Recommendation
8. Key Performance Indicators (KPIs)
9. Next Steps
"""
        return self.generate_response(prompt, system=_PROPOSAL_SYSTEM_PROMPT)

    def chat_with_documents(self, question: str, context: str) -> str:
        if not context.strip():
            return "The requested information could not be found in the uploaded documents."
        prompt = f"""
## DOCUMENT EXCERPTS
{context}

## QUESTION
{question}

Answer using only the document excerpts above.
"""
        return self.generate_response(prompt, system=_CHAT_SYSTEM_PROMPT)
