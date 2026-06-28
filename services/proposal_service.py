"""
Proposal Service
================
Orchestrates RAG pipeline: ingest → embed → retrieve → generate.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import streamlit as st

import config
from rag.document_loader import DocumentLoader
from rag.section_chunker import SectionChunker
from rag.text_cleaner import TextCleaner
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from services.embedding_service import EmbeddingService
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class ProposalService:

    def __init__(self) -> None:
        self._loader = DocumentLoader()
        self._chunker = SectionChunker()
        self._cleaner = TextCleaner()
        self._embedder = EmbeddingService()
        self._store = VectorStore()
        self._retriever = Retriever(self._store, self._embedder)
        self._llm = GeminiService()
        logger.info("ProposalService ready.")

    # ── Availability ──────────────────────────────────────────────────────────

    def gemini_available(self) -> bool:
        return True  # Gemini API is always available if key is set

    # kept for backward compat with any UI references
    def ollama_available(self) -> bool:
        return self.gemini_available()

    # ── Ingest ────────────────────────────────────────────────────────────────

    def save_uploaded_file(self, uploaded_file) -> Path:
        dest = config.DOCUMENTS_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return dest

    def ingest_document(self, file_path: Path, document_type: str) -> dict:
        raw_text = self._loader.load(file_path)
        cleaned = self._cleaner.clean(raw_text)
        chunks = self._chunker.chunk(
            cleaned,
            source_file=file_path.name,
            document_type=document_type,
        )
        if not chunks:
            return {"num_chunks": 0}

        if len(chunks) > 10:
            chunks = chunks[:10]

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_chunks(texts)
        self._store.add_chunks(chunks, embeddings)
        return {"num_chunks": len(chunks)}

    def rebuild_index(self) -> dict:
        self._store.clear()
        total_files = 0
        total_chunks = 0
        for file_path in config.DOCUMENTS_DIR.glob("*"):
            if file_path.suffix.lower() in {".pdf", ".docx"}:
                try:
                    result = self.ingest_document(file_path, "Unknown")
                    total_files += 1
                    total_chunks += result["num_chunks"]
                except Exception as exc:
                    logger.error("Failed to ingest %s: %s", file_path.name, exc)
        return {"total_files": total_files, "total_chunks": total_chunks}

    def clear_knowledge_base(self) -> None:
        self._store.clear()
        if config.DOCUMENTS_DIR.exists():
            shutil.rmtree(config.DOCUMENTS_DIR)
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate_proposal(
        self,
        client_name: str,
        industry: str,
        budget: str,
        goals: str,
        services: str,
    ) -> tuple[str, list[dict]]:
        query = f"{industry} marketing proposal {goals} {services}"
        retrieved = self._retriever.retrieve(query)
        context = self._retriever.format_context(retrieved)

        proposal = self._llm.generate_proposal(
            context=context,
            client_name=client_name,
            industry=industry,
            budget=budget,
            goals=goals,
            services=services,
        )
        return proposal, retrieved

    def chat(self, question: str) -> tuple[str, list[dict]]:
        retrieved = self._retriever.retrieve(question)
        context = self._retriever.format_context(retrieved)
        answer = self._llm.chat_with_documents(question, context)
        return answer, retrieved

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        doc_stats = self._store.get_document_stats()
        return {
            "total_documents": len(doc_stats.get("documents", {})),
            "total_chunks": doc_stats.get("total_chunks", 0),
            "documents": doc_stats.get("documents", {}),
            "document_types": doc_stats.get("document_types", {}),
            "embedding_model": self._embedder.model_name,
            "ollama_model": self._llm.model,
            "ollama_available": True,
            "faiss_index_path": str(config.FAISS_INDEX_PATH),
        }
