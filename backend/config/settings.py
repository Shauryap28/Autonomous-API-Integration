"""Central configuration — the single source of truth."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # primary LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")        # fallback LLM (now active)

# --- Models ---
GEMINI_MODEL = "gemini-2.5-flash"               # primary: extraction + codegen
GROQ_MODEL = "openai/gpt-oss-120b"              # fallback: codegen/reasoning + json
GROQ_FAST_MODEL = "openai/gpt-oss-20b"          # fast/cheap lane (unused yet)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # local, free
# Note: llama-3.3-70b-versatile is being deprecated on Groq; gpt-oss is the path.

# --- Chunking (section-aware) ---
CHUNK_MAX_CHARS = 1200
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Vector store ---
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "api_docs"
DISTANCE_METRIC = "cosine"

# --- Retrieval ---
TOP_K = 3

# --- Generation (separate budgets — these outputs are different sizes) ---
EXTRACT_MAX_OUTPUT_TOKENS = 2048   # schema JSON with a full parameter list
CODEGEN_MAX_OUTPUT_TOKENS = 2048   # generated fetch scripts

# --- Execution (Phase 2) ---
EXEC_TIMEOUT = 30                  # seconds; hard cap on a generated script