"""Central configuration — the single source of truth."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # primary LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")        # fallback LLM

# --- Models ---
GEMINI_MODEL = "gemini-2.5-flash"               # primary: extraction + codegen
GROQ_MODEL = "openai/gpt-oss-120b"              # fallback: codegen/reasoning + json
GROQ_FAST_MODEL = "openai/gpt-oss-20b"          # fast/cheap lane (unused yet)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # local, free

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

# --- Generation ---
EXTRACT_MAX_OUTPUT_TOKENS = 8192
CODEGEN_MAX_OUTPUT_TOKENS = 4096

# --- Execution ---
EXEC_TIMEOUT = 30                  # seconds; local runner cap

# --- Sandbox (Phase 3) ---
USE_SANDBOX = True                 # True = Docker sandbox; False = local_runner fallback
SANDBOX_IMAGE = "aaie-sandbox"
SANDBOX_MEM_LIMIT = "256m"
SANDBOX_CPUS = 0.5
SANDBOX_TIMEOUT = 30
SANDBOX_READONLY = True

# --- Agent loop (Phase 4) ---
# Bounded retries: a stuck agent must never loop forever burning API quota.
# 5 is enough to correct genuine mistakes, small enough to fail fast when hopeless.
MAX_RETRIES = 5

# Demo lever: our GitHub call succeeds first try, so nothing would naturally
# exercise the self-healing loop. FORCE_FAILURE deliberately corrupts the FIRST
# generated script (breaks the endpoint path) so the API returns a real 404 and the
# agent must diagnose and repair it. Set False for normal runs.
FORCE_FAILURE = False

# --- Endpoint selection ---
# The LLM proposes the top-3 doc sections matching the goal; the human confirms.
# Set False to auto-accept the top pick (scripted runs / demos without a prompt).
# The plan's warning applies: too eager to ask = annoying; too reluctant = hallucinated
# guesses. A toggle resolves that rather than picking one extreme.
CONFIRM_ENDPOINT = True
SELECT_MAX_OUTPUT_TOKENS = 1024   # section names + 3 candidates — a small response