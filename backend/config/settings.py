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
EXTRACT_MAX_OUTPUT_TOKENS = 4096   # schema + the model's verbose field descriptions
CODEGEN_MAX_OUTPUT_TOKENS = 2048

# --- Execution ---
EXEC_TIMEOUT = 30                  # seconds; local runner cap

# --- Sandbox (Phase 3) ---
USE_SANDBOX = True                 # True = Docker sandbox; False = local_runner fallback
SANDBOX_IMAGE = "aaie-sandbox"     # built from sandbox_image/Dockerfile
SANDBOX_MEM_LIMIT = "256m"         # hard memory cap
SANDBOX_CPUS = 0.5                 # half a CPU
SANDBOX_TIMEOUT = 30               # seconds; container is killed if it overruns
SANDBOX_READONLY = True            # read-only root filesystem inside the container