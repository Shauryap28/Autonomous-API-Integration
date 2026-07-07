# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
generates code to fetch the requested data, runs that code inside an isolated sandbox,
and (in later phases) self-corrects from real HTTP errors and escalates to a human when
stuck. Built toward a LangGraph cyclic state machine.

This README describes the **current state**, not a change history.

---

## Current capability (Phases 1-3 complete)

Given an API's docs (URL or PDF) and a goal, the system:
1. **comprehends the docs** into a validated `ApiSchema` (Phase 1),
2. **generates a Python script** from that schema and goal (Phase 2), and
3. **runs that script inside an isolated Docker sandbox** and returns the fetched
   data (Phase 3).

It does not yet self-correct from errors, loop, or sync to a database — those are
Phase 4 and beyond.

### What works today
- Fetch docs from a **URL** or local **PDF/HTML**, normalized to clean text with
  markdown headings.
- **Section-aware chunking**, each chunk tagged with its enclosing endpoint.
- Local **BGE-small** embeddings + **ChromaDB**.
- **Endpoint-scoped retrieval** (metadata filter) — no cross-endpoint contamination.
- **Structured extraction** into a Pydantic-validated `ApiSchema`.
- **Code generation** from schema + goal into a runnable `requests` script.
- **Sandboxed execution** in a Docker container: no host filesystem, capped
  memory/CPU, read-only filesystem, non-root user, hard timeout.
- **LLM resilience:** Gemini primary with 503 retry/backoff; automatic **Groq
  fallback** (`openai/gpt-oss-120b`) when Gemini is unavailable.

### Validated on
- **GitHub REST API**, *"List organization repositories"* — extracts the schema,
  generates a script, and fetches 50 real repos end-to-end, executed inside the
  container. Safety proven: infinite loops are killed at the timeout; file writes
  are blocked by the read-only filesystem; code runs as a non-root user.

---

## Architecture (current pipeline)

```
docs URL / PDF  +  natural-language goal
        |
   [ Phase 1: comprehension ]
  fetcher -> chunker -> embeddings -> vectorstore -> retrieval (endpoint filter)
                                                          |
                                                          v
                                              extractor -> ApiSchema (validated)
        |
   [ Phase 2: code generation ]
  codegen (schema + goal -> Python fetch script)
        |
   [ Phase 3: sandboxed execution ]
  runner (USE_SANDBOX?) -> docker_runner -> [ container: capped, read-only,
        |                                      non-root, no host mount ] -> JSON
        |                     \--> local_runner (fallback if USE_SANDBOX=False)
        v
  fetched JSON  (exit_code / stdout / stderr)

  backend/llm.py underlies extraction + codegen: Gemini primary, Groq fallback.
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                  # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── schemas.py          # ApiSchema — the comprehension contract
│   │   ├── codegen.py          # schema + goal -> Python fetch script
│   │   └── state.py            # AgentState skeleton (later phases)
│   ├── rag/
│   │   ├── fetcher.py          # URL/PDF -> clean text + fetchability report
│   │   ├── chunker.py          # section-aware -> LangChain Documents
│   │   ├── embeddings.py       # BGE-small
│   │   ├── vectorstore.py      # ChromaDB + get_endpoint_chunks
│   │   ├── retriever.py        # endpoint-scoped similarity retrieval
│   │   └── extractor.py        # retrieved chunks -> ApiSchema
│   ├── sandbox/
│   │   ├── local_runner.py     # run script as a local subprocess (fallback)
│   │   ├── docker_runner.py    # run script in an isolated Docker container
│   │   └── runner.py           # selects runner via settings.USE_SANDBOX
│   └── config/settings.py      # single source of truth
├── sandbox_image/Dockerfile    # minimal python:3.11-slim + requests, non-root
├── data/chroma_db/             # local vector store (gitignored — generated)
├── tests/                      # test_rag.py
├── main.py                     # end-to-end runner
├── discover.py                 # generality probe: list a doc's endpoints (no LLM)
├── test_sandbox_isolation.py   # sandbox safe-failure checks
├── requirements.txt
└── .env                        # GEMINI_API_KEY, GROQ_API_KEY
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# put GEMINI_API_KEY and GROQ_API_KEY in .env
```

Phase 3 also requires **Docker Desktop** installed and running, and the sandbox
image built once:

```bash
docker build -t aaie-sandbox ./sandbox_image
```

## Run

```bash
python main.py                        # full pipeline (docs -> schema -> code -> sandboxed fetch)
python discover.py <url-or-file>      # probe any doc's endpoint structure (no LLM)
python test_sandbox_isolation.py      # prove the sandbox's safety guarantees
```

First run downloads the BGE-small model (~130MB). To run without Docker, set
`USE_SANDBOX = False` in `settings.py` (falls back to local execution).

---

## Configuration (settings.py highlights)
| Setting | Purpose |
|---|---|
| `USE_SANDBOX` | `True` = Docker sandbox; `False` = local execution fallback |
| `SANDBOX_IMAGE` | image tag built from `sandbox_image/Dockerfile` |
| `SANDBOX_MEM_LIMIT` / `SANDBOX_CPUS` | container memory / CPU caps |
| `SANDBOX_TIMEOUT` | seconds before a runaway container is killed |
| `SANDBOX_READONLY` | read-only container filesystem |
| `EXTRACT_MAX_OUTPUT_TOKENS` / `CODEGEN_MAX_OUTPUT_TOKENS` | LLM output budgets |
| `TOP_K`, `CHUNK_*` | retrieval / chunking knobs |

---

## Development line (roadmap)

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs -> validated `ApiSchema` | done |
| 2 | Schema -> generated script -> real fetch | done |
| 3 | Docker sandbox (contained execution) | done |
| 4 | Self-healing loop (LangGraph): diagnose errors, retry, error-history | next |
| 4.5 | Comprehension hardening: table parsing + PDF heading detection (measure-driven) | planned |
| 5 | Human-in-the-loop + checkpointing | planned |
| 6 | Demo surface + eval harness | planned |
| 7 | AWS deployment & hardening (RDS, Parameter Store, CloudWatch); egress lockdown | planned |
| 8 | Safe write operations (POST/PUT/DELETE, idempotency, HITL-before-mutate) | future |