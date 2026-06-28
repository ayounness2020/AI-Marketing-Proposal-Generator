"""
Gemini Service
==============
Replaces OllamaService — uses Google Gemini API for LLM generation.
"""

from __future__ import annotations

import logging
import os
import google.generativeai as genai
import streamlit as st
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
- You can respond in Arabic or English depending on the client's language.

OUTPUT: Return a complete proposal in markdown format with all requested sections.
"""

_CHAT_SYSTEM_PROMPT = """You are a helpful marketing assistant. Use the provided document excerpts to answer questions.
Summarize what you find. Be helpful and informative. You can respond in Arabic or English."""


class GeminiService:

    def __init__(self) -> None:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
        genai.configure(api_key=api_key, transport="rest")
        self._model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4096,
            }
        )
        self._model_name = config.GEMINI_MODEL
        logger.info("Gemini service ready: %s", self._model_name)

    @property
    def model(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        try:
            self._model.generate_content("ping")
            return True
        except Exception:
            return False

    def generate_response(self, prompt: str, system: str = "") -> str:
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = self._model.generate_content(full_prompt)
            return response.text
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            return f"⚠️ Error communicating with Gemini: {exc}"

    def generate_proposal(
        self,
        context: str,
        client_name: str,
        industry: str,
        budget: str,
        goals: str,
        services: str,
    ) -> str:
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
