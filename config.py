import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_DIR = DATA_DIR / "faiss"

FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_DIR / "metadata.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "models/text-embedding-004"
EMBEDDING_DIMENSION: int = 768  # Gemini text-embedding-004 output dimension

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))

# ── Section headers ───────────────────────────────────────────────────────────
SECTION_HEADERS: list[str] = [
    "executive summary", "client background", "business objectives",
    "services included", "scope of work", "deliverables", "timeline",
    "pricing", "investment", "budget", "kpis", "key performance indicators",
    "marketing strategy", "seo strategy", "paid ads strategy",
    "social media strategy", "content strategy", "case study", "results",
    "recommendations", "next steps", "about us", "our approach",
    "target audience", "competitive analysis", "introduction",
    "overview", "conclusion", "contact", "appendix",
]

DOCUMENT_TYPES: list[str] = [
    "Marketing Proposal", "Case Study", "Pricing Document",
    "Service Description", "Marketing Plan", "Other",
]

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
