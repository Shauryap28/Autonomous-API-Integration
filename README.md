# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation + a natural-language goal,
writes and runs code in a sandbox to fetch the data, self-corrects from real HTTP
errors, and asks a human when genuinely stuck. Built on a LangGraph cyclic state
machine. 



## Phase 1 — Docs → `api_schema`  (current)

**What we're building:** a RAG pipeline over ONE API's documentation that produces
a clean, Pydantic-validated `api_schema` describing how to talk to that API.

**Definition of done:** feed GitHub's docs in, get back a valid `ApiSchema` with
the right `{auth_method, base_url, endpoint, params, pagination, rate_limit}`.

Nothing else yet — no LangGraph graph, no codegen, no sandbox, no Postgres, no
retry loop. Those arrive in later phases. Phase 1 is about deeply understanding
doc comprehension → structured schema.

### Pipeline
```
doc URL/PDF → fetcher → chunker → embeddings (BGE-small) → ChromaDB
            → targeted retrieval (auth / endpoint / params / pagination)
            → extractor (Gemini) → ApiSchema (Pydantic, validated)
```

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GEMINI_API_KEY
```

### Run the Phase 1 milestone
```bash
pytest tests/test_rag.py
```

---

## Project layout (Phase 1 slice)
```
backend/
  agent/      state.py (AgentState), schemas.py (ApiSchema)   ← contracts
  rag/        fetcher, chunker, embeddings, vectorstore, extractor
  config/     settings.py
tests/        test_rag.py
data/         chroma/ (local vector store, gitignored)
```
Later phases add `agent/graph.py`, `agent/nodes/`, `agent/edges.py`,
`sandbox/`, the FastAPI backend, the Streamlit UI, and the AWS infra (Phase 7).
