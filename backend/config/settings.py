"""Central configuration — the single source of truth."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # used by the extractor (Phase 1)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")        # fallback LLM (Phase 2+; unused now)

# --- Models ---
GEMINI_MODEL = "gemini-2.5-flash"               # structured extraction
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # local, free

# --- Chunking (section-aware) ---
# Docs are chunked by SECTION (heading), not fixed size. A section longer than
# CHUNK_MAX_CHARS is sub-split with the window below; a doc with NO headings
# falls back entirely to the window. (This is the one knob MDIS didn't need.)
CHUNK_MAX_CHARS = 1200
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Vector store ---
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "api_docs"
DISTANCE_METRIC = "cosine"      # matches normalized BGE embeddings

# --- Retrieval ---
TOP_K = 3                       # per targeted query (auth / endpoint / pagination / ...)

# --- Generation ---
MAX_OUTPUT_TOKENS = 1024        # schema JSON is small; cap guards against runaway output