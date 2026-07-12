# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
generates code to fetch the requested data, runs that code inside an isolated sandbox,
and self-corrects from real HTTP errors — retrying until it succeeds or exhausts a
bounded retry budget. Built on a LangGraph cyclic state machine.

This README describes the **current state**, not a change history.

---

## Current capability (Phases 1-4 complete)

Given an API's docs (URL or PDF) and a natural-language goal, the system:
1. **comprehends the docs** into a validated `ApiSchema` (auth, endpoint, params,
   pagination, response shape),
2. **generates a Python script** from that schema and goal,
3. **runs it inside an isolated Docker sandbox**, and
4. **on failure, reads the real error, re-queries the docs for the section that
   explains it, rewrites the script, and retries** — bounded, with memory of every
   past attempt.

Not yet implemented: human-in-the-loop escalation, checkpoint/resume, database sync.

### What works today
- Fetch docs from a **URL** or local **PDF/HTML**; headings normalized to markdown.
- **Section-aware chunking**, each chunk tagged with its enclosing endpoint section.
- Local **BGE-small** embeddings + **ChromaDB**; **endpoint-scoped retrieval**.
- **Structured extraction** into a Pydantic-validated `ApiSchema`.
- **Code generation** from schema + goal into a runnable `requests` script.
- **Sandboxed execution**: no host filesystem, capped memory/CPU, read-only
  filesystem, non-root user, hard timeout.
- **Self-healing loop** (LangGraph): conditional routing on the execution result,
  error-driven doc retrieval, full-script regeneration, bounded retries, and an
  append-only `error_history` so a failed fix is never repeated.
- **LLM resilience:** Gemini primary with 503 retry/backoff; automatic **Groq
  fallback** (`openai/gpt-oss-120b`).

### Validated on
- **GitHub REST API** (`/orgs/{org}/repos`) — page-based pagination, top-level array
  response. Fetches 50 repos. With a deliberately corrupted endpoint path, the agent
  receives a real **404**, retrieves the relevant doc section, corrects the path, and
  succeeds on attempt 2.
- **PokéAPI** (`/pokemon`) — **offset/limit** pagination and records **nested under
  `results`**, both structurally different from GitHub. Succeeded first try, with no
  code changes to the pipeline.

---

## Architecture

```
docs URL / PDF  +  natural-language goal
        |
   [ SETUP — runs once: comprehension ]
  fetcher -> chunker -> embeddings -> vectorstore -> endpoint-scoped retrieval
                                                          |
                                              extractor -> ApiSchema (validated)
        |
   [ AGENT GRAPH — LangGraph, cyclic ]
        |
   START -> generate_code -> execute -> <route_result>
                                ^             |
                                |             +-- success            -> END
                                |             +-- retries exhausted  -> END (failed)
                                |             +-- retry -> diagnose_and_fix --+
                                +----------------------------------------------+

   execute  = the Docker sandbox (capped, read-only, non-root, no host mount)
   diagnose = reads the REAL error + the doc section retrieved USING that error
              + error_history -> regenerates the full script

  backend/llm.py underlies extraction / codegen / diagnosis: Gemini -> Groq fallback.
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                  # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── state.py            # AgentState (TypedDict; error_history has an append reducer)
│   │   ├── graph.py            # the cyclic StateGraph
│   │   ├── nodes.py            # generate_code / execute / diagnose / route_result
│   │   ├── codegen.py          # schema + goal -> Python fetch script
│   │   ├── diagnose.py         # error + relevant docs + history -> corrected script
│   │   └── schemas.py          # ApiSchema — the comprehension contract
│   ├── rag/                    # fetcher, chunker, embeddings, vectorstore,
│   │                           # retriever, extractor
│   ├── sandbox/
│   │   ├── docker_runner.py    # isolated container execution
│   │   ├── local_runner.py     # local subprocess (fallback)
│   │   └── runner.py           # selects backend via settings.USE_SANDBOX
│   └── config/settings.py      # single source of truth
├── sandbox_image/Dockerfile    # python:3.11-slim + requests, non-root
├── data/chroma_db/             # local vector store (gitignored — generated)
├── main.py                     # setup + agent graph
├── discover.py                 # probe any doc's endpoint structure (no LLM)
├── test_sandbox_isolation.py   # sandbox safety checks
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

Requires **Docker Desktop** running, and the sandbox image built once:
```bash
docker build -t aaie-sandbox ./sandbox_image
```

## Run

```bash
python main.py                        # full agent run
python discover.py <url-or-file>      # list a doc's endpoint sections (no LLM)
python test_sandbox_isolation.py      # prove the sandbox's safety guarantees
```

To watch the self-healing loop, set `FORCE_FAILURE = True` in `settings.py`: the
first generated script is deliberately corrupted, producing a genuine HTTP 404 that
the agent must diagnose and repair.

---

## Configuration (settings.py)

| Setting | Purpose |
|---|---|
| `MAX_RETRIES` | bounded retry budget for the self-healing loop (default 5) |
| `FORCE_FAILURE` | demo lever: corrupt the first script to force a real error |
| `USE_SANDBOX` | `True` = Docker sandbox; `False` = local execution fallback |
| `SANDBOX_*` | image, memory/CPU caps, timeout, read-only filesystem |
| `EXTRACT_MAX_OUTPUT_TOKENS` / `CODEGEN_MAX_OUTPUT_TOKENS` | LLM output budgets |
| `TOP_K`, `CHUNK_*` | retrieval / chunking knobs |

---

## Development line

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs -> validated `ApiSchema` | done |
| 2 | Schema -> generated script -> real fetch | done |
| 3 | Docker sandbox (contained execution) | done |
| 4 | Self-healing loop (LangGraph cyclic graph) | done |
| 4.5 | Comprehension hardening: table parsing + PDF heading detection | planned |
| 5 | Human-in-the-loop + checkpoint/resume | next |
| 6 | Demo surface + eval harness | planned |
| 7 | AWS deployment & hardening; egress lockdown | planned |
| 8 | Safe write operations (idempotency, HITL-before-mutate) | future |