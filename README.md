# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
works out which endpoint fits, generates code to fetch the data, runs that code inside an
isolated sandbox, self-corrects from real HTTP errors, and syncs the result into Postgres.
Built on a LangGraph cyclic state machine.

This README describes the **current state**, not a change history.

---

## Current capability

Give it a docs URL and a plain-English goal. It then:

1. **Reads the docs** — fetch, section-aware chunk, embed (BGE-small), index (ChromaDB).
2. **Picks the endpoint** — an LLM matches the goal against the doc's section list,
   proposes a ranked top-3 for you to confirm, or **denies the goal** outright if no
   endpoint can satisfy it (before spending anything on embedding or codegen).
3. **Comprehends it** — endpoint-scoped retrieval, plus any **global sections**
   (pagination / auth documented separately) → a Pydantic-validated `ApiSchema`.
4. **Writes the code** — a runnable `requests` script from the schema + goal.
5. **Runs it safely** — inside a Docker container: no host filesystem, capped memory/CPU,
   read-only filesystem, non-root user, hard timeout, **no database credentials**.
6. **Fixes itself** — on failure, reads the real error, re-queries the docs *using that
   error*, rewrites the script, retries (bounded, with memory of every past attempt).
7. **Persists the data** — a **trusted backend** (outside the sandbox) structurally
   validates the fetched JSON and upserts it into Postgres as JSONB, then verifies the
   row count. Re-running never duplicates.

Not yet implemented: human-in-the-loop escalation, checkpoint/resume, doc caching.

### Validated on
- **GitHub REST API** (`/orgs/{org}/repos`) — page-based pagination, top-level array.
  50 repos fetched and persisted. With the endpoint path deliberately corrupted, the agent
  gets a real **404**, retrieves the relevant docs, corrects the path, succeeds on attempt 2.
- **PokéAPI** (`/pokemon`) — offset/limit pagination (documented in a *separate* global
  section), records nested under `results`. 50 Pokémon fetched and persisted.
- **Goal denial** — asked to "delete all repositories and drop the database", the selector
  correctly refuses rather than forcing a match onto the unrelated "Delete a repository".
- **Idempotency** — running the same goal twice leaves the row count unchanged (50, not 100).

---

## Architecture

```
docs URL  +  natural-language goal
        |
   [ SETUP — runs once ]
   fetcher -> chunker -> section list
                            |
                    select_endpoint (LLM: top-3 or DENY)  <-- human confirms
                    identify_global_sections (pagination/auth documented separately?)
                            |
             embeddings -> vectorstore -> endpoint-scoped + global retrieval
                            |
                     extractor -> ApiSchema (validated)
        |
   [ AGENT GRAPH — LangGraph, cyclic ]
   START -> generate_code -> execute -> <route_result>
                                ^            |
                                |            +-- success           -> persist_and_verify -> END
                                |            +-- retries exhausted -> END (failed)
                                |            +-- retry -> diagnose_and_fix --+
                                +----------------------------------------------+

   execute            = Docker sandbox — fetches; holds the API key, NO DB credentials
   diagnose_and_fix   = real error + docs retrieved USING that error + error_history
   persist_and_verify = TRUSTED host process — holds DB_URL, validates + upserts to Postgres

   backend/llm.py underlies every LLM step: Gemini primary, Groq fallback on 429/503.
```

---

## Data model

One row per fetched record, stored whole in a `JSONB` column — so any API's shape is
accepted with no per-API table design.

```sql
fetched_records (
    id, source, endpoint, record_key, goal, record JSONB, fetched_at,
    UNIQUE (source, endpoint, record_key)
)
```

Identity is `(source, endpoint, record_key)` — deliberately **not** the goal, so the same
record fetched by "first 50" and "first 100" updates in place rather than duplicating.
`record_key` is derived from the record (`id` → `name` → … → content hash).

Query it like any relational data:
```sql
SELECT record->>'name', record->>'language' FROM fetched_records WHERE source = 'repos';
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                  # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── selector.py         # goal -> endpoint section (top-3 / deny)
│   │   ├── global_sections.py  # which sections document pagination/auth globally
│   │   ├── state.py            # AgentState (error_history has an append reducer)
│   │   ├── graph.py            # the cyclic StateGraph
│   │   ├── nodes.py            # generate / execute / diagnose / persist / route
│   │   ├── codegen.py          # schema + goal -> Python fetch script
│   │   ├── diagnose.py         # error + relevant docs + history -> corrected script
│   │   └── schemas.py          # ApiSchema — the comprehension contract
│   ├── rag/                    # fetcher, chunker, embeddings, vectorstore,
│   │                           # retriever, extractor
│   ├── db/persist.py           # TRUSTED: validate -> upsert -> verify (holds DB_URL)
│   ├── sandbox/                # docker_runner, local_runner, runner (USE_SANDBOX)
│   └── config/settings.py      # single source of truth
├── sandbox_image/Dockerfile    # python:3.11-slim + requests, non-root
├── data/chroma_db/             # vector store (gitignored — generated)
├── main.py                     # setup + agent graph
├── discover.py                 # list a doc's endpoint sections (no LLM)
├── test_sandbox_isolation.py   # sandbox safety checks
├── requirements.txt
└── .env                        # GEMINI_API_KEY, GROQ_API_KEY, DB_URL
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

`.env`:
```
GEMINI_API_KEY=...
GROQ_API_KEY=...
DB_URL=postgresql://USER:PASSWORD@localhost:5432/YOUR_DB
```

Also required:
- **Docker Desktop** running, with the sandbox image built once:
  `docker build -t aaie-sandbox ./sandbox_image`
- **PostgreSQL** reachable at `DB_URL` (the table is created automatically on first run).

## Run

```bash
python main.py                                     # defaults (GitHub repos)
python main.py <docs-url> "your goal in English"   # any docs, any goal
python discover.py <docs-url>                      # list a doc's sections (no LLM)
python test_sandbox_isolation.py                   # prove the sandbox guarantees
```

Set `FORCE_FAILURE = True` in `settings.py` to watch the self-healing loop: the first
generated script is deliberately corrupted, producing a genuine 404 the agent must repair.

---

## Configuration (settings.py)

| Setting | Purpose |
|---|---|
| `KEY_STRATEGY` | `"derived"` (default) or `"llm"` — how each record's identity key is chosen |
| `CONFIRM_ENDPOINT` | ask the human to confirm the selected endpoint (False = auto-accept) |
| `MAX_RETRIES` | bounded retry budget for the self-healing loop (default 5) |
| `FORCE_FAILURE` | demo lever: corrupt the first script to force a real error |
| `USE_SANDBOX` | `True` = Docker sandbox; `False` = local execution fallback |
| `SANDBOX_*` | image, memory/CPU caps, timeout, read-only filesystem |
| `GLOBAL_CONTEXT_MAX_CHUNKS` | cap on global-section chunks added to extraction context |
| `*_MAX_OUTPUT_TOKENS` | LLM output budgets (selection / extraction / codegen) |

---

## Known limitations

- **No DB-level schema enforcement.** JSONB accepts any valid JSON; we validate the
  *structure* of the batch (a non-empty list of JSON objects), not per-record fields —
  a deliberate trade of strictness for generality.
- **The Groq fallback can't serve large prompts.** Its free-tier limit (8000 TPM) is below
  what a large extraction context needs (~9700 tokens), so when Gemini is rate-limited the
  fallback fails on exactly the heaviest requests.
- **Provider fallback is not deterministic** — Gemini and Groq can make different endpoint
  selections on identical input.
- **Egress is not restricted.** The sandbox blocks host filesystem access and caps
  resources, but default bridge networking lets the container reach any host. Phase 7.
- **Tables are flattened, not parsed**; the **PDF input path is untested**. Phase 4.5.
- **Self-correction is proven against an injected error**, not yet a spontaneous one.
- **Single-page pagination only** — "first N" fits one page; multi-page "fetch all" with
  backoff is deferred.
- **Docs are re-embedded on every run** (the store is cleared each time) — doc caching is
  the next phase.

---

## Development line

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs -> validated `ApiSchema` | done |
| 2 | Schema -> generated script -> real fetch | done |
| 3 | Docker sandbox (contained execution) | done |
| 4 | Self-healing loop (LangGraph cyclic graph) | done |
| 4.1 | Endpoint selection from the goal (top-3 / deny / confirm) | done |
| 4.2 | `persist_and_verify` — Postgres JSONB sync, idempotent upserts | done |
| 4.3 | Doc caching (content hash + freshness window; stop re-embedding) | next |
| 4.5 | Comprehension hardening: table parsing, PDF heading detection | planned |
| 5 | Human-in-the-loop (`interrupt`) + checkpoint/resume | planned |
| 6 | Streamlit surface (browse endpoints, live trace) + eval harness | planned |
| 7 | AWS deployment & hardening; egress lockdown | planned |
| 8 | Safe write operations (idempotency, HITL-before-mutate) | future |