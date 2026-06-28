"""
AI Marketing Proposal Generator
================================
Streamlit frontend — thin UI layer that delegates all logic
to ProposalService.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st
import traceback

# ── Bootstrap path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from services.proposal_service import ProposalService

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Marketing Proposal Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .chunk-card {
        background: #f1f5f9;
        border-left: 4px solid #2d6a9f;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .score-high { color: #16a34a; font-weight: bold; }
    .score-mid  { color: #d97706; font-weight: bold; }
    .score-low  { color: #dc2626; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Service singleton (cached) ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising AI services…")
def get_service() -> ProposalService:
    return ProposalService()


service = get_service()

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks: list[dict] = []

if "last_context_sent" not in st.session_state:
    st.session_state.last_context_sent: str = ""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📁 Document Management")

    # -- Upload --
    st.markdown("### Upload Documents")
    doc_type = st.selectbox("Document Type", config.DOCUMENT_TYPES)
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload previous proposals, case studies, pricing sheets, etc.",
    )

    if st.button("➕ Add to Knowledge Base", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file.")
        else:
            progress = st.progress(0)
            for i, uf in enumerate(uploaded_files):
                with st.spinner(f"Processing {uf.name}…"):
                    try:
                        saved_path = service.save_uploaded_file(uf)
                        result = service.ingest_document(saved_path, doc_type)
                        st.success(
                            f"✅ {uf.name} → {result['num_chunks']} chunks added"
                        )
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")
                progress.progress((i + 1) / len(uploaded_files))

    st.divider()

    # -- Actions --
    st.markdown("### Knowledge Base Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding..."):
                st.cache_resource.clear()
                res = service.rebuild_index()
                st.success(f"Rebuilt: {res['total_files']} files, {res['total_chunks']} chunks")
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            if st.session_state.get("confirm_clear"):
                service.clear_knowledge_base()
                st.success("Knowledge base cleared.")
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm.")

    st.divider()

    # -- Quick stats --
    stats = service.get_stats()
    st.markdown("### 📊 System Status")
    st.metric("Documents", stats["total_documents"])
    st.metric("Chunks", stats["total_chunks"])

    ollama_ok = stats["ollama_available"]
    st.markdown(
        f"**Ollama:** {'🟢 Online' if ollama_ok else '🔴 Offline'}"
    )
    st.markdown(f"**Model:** `{stats['ollama_model']}`")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="main-header">
        <h1>📊 AI Marketing Proposal Generator</h1>
        <p>RAG-powered proposals from your agency's knowledge base</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Proposal Generator", "💬 Chat With Documents", "📚 Knowledge Base", "🔍 Debug Panel"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PROPOSAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Generate a Customised Marketing Proposal")
    st.info(
        "Fill in the client details below. The system will search your knowledge base "
        "and generate a tailored proposal using your previous work."
    )

    with st.form("proposal_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            client_name = st.text_input("🏢 Client Name", placeholder="e.g. Al-Nour Real Estate")
            industry = st.text_input("🏭 Industry", placeholder="e.g. Real Estate, Healthcare, Retail")
            budget = st.text_input("💰 Budget Range", placeholder="e.g. EGP 50,000/month")
        with col_b:
            goals = st.text_area(
                "🎯 Business Goals",
                placeholder="e.g. Increase brand awareness, generate 200 qualified leads/month, expand to Alexandria",
                height=120,
            )
            services = st.text_area(
                "🛠️ Requested Services",
                placeholder="e.g. SEO, Paid Ads (Google + Meta), Content Marketing, Monthly Reporting",
                height=120,
            )

        submitted = st.form_submit_button("🚀 Generate Proposal", use_container_width=True, type="primary")

    if submitted:
        if not client_name or not industry:
            st.error("Please provide at least a Client Name and Industry.")
        elif stats["total_chunks"] == 0:
            st.warning(
                "⚠️ Your knowledge base is empty. "
                "Upload previous proposals first so the AI can generate relevant content."
            )
        elif not service.ollama_available():
            st.error("❌ Ollama is not running. Start Ollama with `ollama serve` and try again.")
        else:
            with st.spinner("🔍 Searching knowledge base and crafting your proposal…"):
                try:
                    proposal, retrieved = service.generate_proposal(
                        client_name=client_name,
                        industry=industry,
                        budget=budget,
                        goals=goals,
                        services=services,
                    )
                    # Store for debug panel
                    st.session_state.last_retrieved_chunks = retrieved
                    st.session_state.last_context_sent = (
                        f"Proposal query for: {client_name} / {industry}"
                    )

                    st.success(f"✅ Proposal generated using {len(retrieved)} knowledge base chunks.")
                    st.divider()

                    # Download button
                    import io
                    from docx import Document as DocxDocument
                    def md_to_docx(t, n):
                        doc = DocxDocument()
                        doc.add_heading("Marketing Proposal for " + client_name, 0)
                        for ln in t.split(chr(10)):
                            ln = ln.strip()
                            if not ln: doc.add_paragraph("")
                            elif ln.startswith("## "): doc.add_heading(ln[3:], level=1)
                            elif ln.startswith("- "): doc.add_paragraph(ln[2:], style="List Bullet")
                            else: doc.add_paragraph(ln)
                        buf = io.BytesIO()
                        doc.save(buf)
                        buf.seek(0)
                        return buf.getvalue()
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button("Word", data=md_to_docx(proposal, client_name), file_name="proposal.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
                    with col_dl2:
                        st.download_button("Markdown", data=proposal, file_name="proposal.md", mime="text/markdown", use_container_width=True)
                    st.markdown(proposal)

                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")
                    st.code(traceback.format_exc())
                    logger.exception("Proposal generation error")
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHAT WITH DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 💬 Chat With Your Document Library")
    st.caption(
        "Ask questions about your previous proposals, case studies, and pricing documents. "
        "The AI answers strictly from uploaded content — no hallucinations."
    )

    # Example prompts
    example_questions = [
        "Show me similar proposals for real estate clients.",
        "What pricing models have we used before?",
        "What services are included in our SEO packages?",
        "What KPIs were proposed for healthcare clients?",
        "What was the timeline in our last e-commerce proposal?",
    ]

    st.markdown("**💡 Example questions:**")
    cols = st.columns(len(example_questions))
    for col, q in zip(cols, example_questions):
        if col.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Searching documents…"):
                answer, retrieved = service.chat(q)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_retrieved_chunks = retrieved
                st.session_state.last_context_sent = f"Chat question: {q}"

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask anything about your documents…"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if stats["total_chunks"] == 0:
            response = "⚠️ Knowledge base is empty. Please upload documents first."
            retrieved = []
        elif not service.ollama_available():
            response = "❌ Ollama is offline. Start it with `ollama serve`."
            retrieved = []
        else:
            with st.spinner("Searching knowledge base…"):
                response, retrieved = service.chat(user_input)
                st.session_state.last_retrieved_chunks = retrieved
                st.session_state.last_context_sent = f"Chat question: {user_input}"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — KNOWLEDGE BASE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📚 Knowledge Base Overview")

    # Refresh
    if st.button("🔄 Refresh Stats"):
        st.rerun()

    stats = service.get_stats()  # re-fetch for latest values

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", stats["total_documents"])
    c2.metric("🧩 Chunks", stats["total_chunks"])
    c3.metric("🤖 Embedding Model", "all-MiniLM-L6-v2")
    c4.metric("💾 FAISS", "Active" if stats["total_chunks"] > 0 else "Empty")

    st.divider()

    if stats["total_chunks"] == 0:
        st.info("No documents in the knowledge base yet. Upload files from the sidebar.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📁 Documents & Chunk Counts")
            for doc_name, chunk_count in stats["documents"].items():
                st.markdown(f"- **{doc_name}** — {chunk_count} chunks")

        with col_right:
            st.markdown("#### 🏷️ Document Types")
            for dtype, count in stats["document_types"].items():
                st.markdown(f"- **{dtype}** — {count} chunks")

        st.divider()
        st.markdown("#### ⚙️ System Configuration")
        cfg_data = {
            "Embedding Model": stats["embedding_model"],
            "LLM Model": stats["ollama_model"],
            "Ollama Status": "🟢 Online" if stats["ollama_available"] else "🔴 Offline",
            "FAISS Index": stats["faiss_index_path"],
            "Documents Dir": str(config.DOCUMENTS_DIR),
            "Top K Retrieval": str(config.TOP_K),
            "Max Chunk Words": str(config.MAX_CHUNK_TOKENS),
        }
        for key, val in cfg_data.items():
            st.markdown(f"**{key}:** `{val}`")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DEVELOPER DEBUG PANEL
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### 🔍 Developer Debug Panel")
    st.caption(
        "Inspect the RAG pipeline: what was retrieved, similarity scores, "
        "source files, and the exact context sent to the LLM."
    )

    retrieved = st.session_state.last_retrieved_chunks

    if not retrieved:
        st.info("No retrieval data yet. Generate a proposal or ask a chat question first.")
    else:
        st.markdown(f"**Last query:** `{st.session_state.last_context_sent}`")
        st.markdown(f"**Chunks retrieved:** {len(retrieved)}")
        st.divider()

        st.markdown("#### Retrieved Chunks")
        for i, chunk in enumerate(retrieved, start=1):
            score = chunk["score"]
            meta = chunk["metadata"]

            # Color-coded score
            if score >= 0.7:
                score_class = "score-high"
            elif score >= 0.4:
                score_class = "score-mid"
            else:
                score_class = "score-low"

            with st.expander(
                f"Chunk {i} | {meta.get('source_file', '?')} › {meta.get('section_name', '?')} | Score: {score:.4f}",
                expanded=(i == 1),
            ):
                col_meta, col_score = st.columns([3, 1])
                with col_meta:
                    st.markdown(f"**Source File:** `{meta.get('source_file', 'unknown')}`")
                    st.markdown(f"**Section:** `{meta.get('section_name', 'unknown')}`")
                    st.markdown(f"**Document Type:** `{meta.get('document_type', 'unknown')}`")
                    st.markdown(f"**Chunk ID:** `{meta.get('chunk_id', 'unknown')}`")
                with col_score:
                    st.markdown(
                        f"<p class='{score_class}'>Score: {score:.4f}</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("**Text:**")
                st.text(chunk["text"])

        st.divider()
        st.markdown("#### Context Sent to LLM")
        if retrieved:
            from rag.retriever import Retriever
            from rag.vector_store import VectorStore
            from services.embedding_service import EmbeddingService
            # Format the same way the service does
            retriever = Retriever(VectorStore(), EmbeddingService())
            formatted = retriever.format_context(retrieved)
