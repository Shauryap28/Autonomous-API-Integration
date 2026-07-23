# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
works out which endpoint fits, generates code to fetch the data, runs that code inside an
isolated Docker sandbox, self-corrects from real HTTP errors, and syncs the result into
PostgreSQL. Built on a LangGraph cyclic state machine, with a Streamlit interface.

This README describes the **current state**, not a change history.
Design rationale and the full problem log live in [`DESIGN.md`](DESIGN.md).

---

## What it does

Give it a documentation URL and a plain-English goal:

```
python main.py https://docs.github.com/en/rest/repos/repos \
  "List the first 50 public repositories of the 'github' organization."
```

1. **Reads the docs** — fetch → section-aware chunk → BGE-small embeddings → ChromaDB.
   Already-indexed docs are reused instantly unless they changed or went stale.
2. **Picks the endpoint** — an LLM matches the goal against the doc's sections, proposes a
   ranked top-3 to confirm, or **denies the goal** if nothing fits.
3. **Comprehends it** — endpoint-scoped retrieval plus any **global sections** (pagination
   or auth documented separately) → a Pydantic-validated `ApiSchema`, then a sanity check
   that the schema is based on real endpoint documentation.
4. **Writes the code** — a runnable `requests` script from the schema + goal.
5. **Runs it safely** — in a container with no host filesystem, capped memory/CPU, a
   read-only filesystem, a non-root user, a hard timeout, and **no database credentials**.
6. **Fixes itself** — on failure it reads the real error, re-queries the docs *using that
   error*, rewrites the script, and retries (bounded, remembering every past attempt).
7. **Persists the data** — a **trusted process outside the sandbox** validates the JSON
   structurally and upserts it into Postgres as JSONB, then verifies the row count.
   Re-running never duplicates.

Not yet implemented: human-in-the-loop escalation, checkpoint/resume, multi-page
pagination, write operations.

---

## Validated on

| API | What it exercises | Result |
|---|---|---|
| **GitHub REST** `/orgs/{org}/repos` | page pagination, top-level array | 50 repos fetched + persisted |
| **PokéAPI** `/pokemon` | offset/limit pagination documented in a *separate* section; records nested under `results` | 50 Pokémon fetched + persisted |
| **Self-healing** | endpoint path deliberately corrupted | real 404 → diagnosed → corrected → success on attempt 2 |
| **Goal denial** | "delete all repositories and drop the database" | refused, with reasoning; no wasted calls |
| **Idempotency** | same goal run twice | row count unchanged (50, not 100) |
| **Hallucination guard** | Open Library's navigation page | schema flagged as not-real before any run |

---

## The interface

```bash
streamlit run frontend/streamlit_app.py
```

**Explore the docs**
- **Endpoint browser** — every section with chunk counts and previews. Free: no LLM, no
  embedding, just the chunker's output.
- **Ask the documentation** — RAG Q&A so you can understand an API *before* writing a
  goal. Scope a question to one section (which scopes both the retrieval and the prompt),
  and expand **Sources** to see which chunks produced the answer.

**Run a goal**
- Goal → ranked candidates with confidence and reasoning → click one → **what the agent
  understood** (method, URL, auth, pagination, where records live, full schema JSON, plus
  a warning if the schema looks invented).
- **Run the agent** → the trace streams node by node → results in four tabs: **Trace**,
  **Generated code** (downloadable), **Error history**, **Data**.

**Vector store panel** — every indexed document with its age, click to load instantly,
delete one, or clear the store.

**Settings** — Docker sandbox on/off, **Force failure** to demo the self-healing loop,
max retries.

---

## Architecture

```
docs URL  +  natural-language goal
        |
   [ SETUP — runs once per document ]
   fetcher ──> verdict: usable / flat / not usable / unreachable
        |
   doc_cache ──> hash + age → reuse the index, or re-chunk and re-embed
        |
   chunker ──> embeddings ──> vectorstore (a persistent library of many docs)
        |
   select_endpoint (LLM: top-3 or DENY)          <── human confirms
   identify_global_sections (pagination/auth documented separately?)
        |
   extractor ──> ApiSchema ──> schema_check (is this real documentation?)
        |
   [ AGENT GRAPH — LangGraph, cyclic ]
   START ─> generate_code ─> execute ─> <route_result>
                                ^            ├── success            ─> persist_and_verify ─> END
                                │            ├── retries exhausted  ─> END (failed)
                                │            └── retry ─> diagnose_and_fix ──┐
                                └────────────────────────────────────────────┘

   execute            = Docker sandbox — holds the API key, NO DB credentials
   diagnose_and_fix   = real error + docs retrieved USING that error + error_history
   persist_and_verify = trusted host process — holds DB_URL, validates + upserts

   backend/llm.py underlies every LLM step: Gemini primary, Groq fallback on 429/503.
```

---

## Data model

One row per fetched record, stored whole in a `JSONB` column, so any API's shape is
accepted with no per-API table design.

```sql
fetched_records (
    id BIGSERIAL PRIMARY KEY,
    source      TEXT,        -- which API
    endpoint    TEXT,        -- which endpoint produced it
    record_key  TEXT,        -- which record (id -> name -> ... -> content hash)
    goal        TEXT,        -- provenance (NOT part of identity)
    record      JSONB,       -- the record, unmodified
    fetched_at  TIMESTAMPTZ,
    UNIQUE (source, endpoint, record_key)
)
```

Identity is `(source, endpoint, record_key)` — deliberately **not** the goal, so the same
record fetched by "first 50" and "first 100" updates in place instead of duplicating.
Idempotency is enforced by the database (`ON CONFLICT DO UPDATE`), not by application code.

```sql
SELECT record->>'name', record->>'language' FROM fetched_records WHERE source = 'docs.github.com/repos';
```

---

## Setup

**Prerequisites:** Python 3.11+, Docker Desktop, PostgreSQL.

```bash
git clone <repo> && cd autonomous-api-integration-engine
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**1. Keys and database** — create `.env` in the repo root:
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
DB_URL=postgresql://USER:PASSWORD@localhost:5432/YOUR_DB
```
`.env` is gitignored — secrets never enter the code, and never enter the sandbox.

**2. Create the database** (the table is created automatically on first run):
```sql
CREATE DATABASE "Autonomous_Api_db";
```

**3. Build the sandbox image** (once):
```bash
docker build -t aaie-sandbox ./sandbox_image
```

**4. Verify:**
```bash
python test_sandbox_isolation.py     # proves the sandbox's safety guarantees
python discover.py https://docs.github.com/en/rest/repos/repos
```

The first run downloads the BGE-small embedding model (~130 MB).

---

## Usage

```bash
streamlit run frontend/streamlit_app.py            # the UI

python main.py                                     # defaults (GitHub repos)
python main.py <docs-url> "your goal in English"
python main.py <docs-url> "your goal" --reindex    # ignore the doc cache

python discover.py <docs-url>                      # can this project use this URL?
python test_sandbox_isolation.py                   # sandbox safety checks
```

**To see the self-healing loop**, set `FORCE_FAILURE = True` in `settings.py` (or toggle
it in the UI sidebar): the first generated script is deliberately corrupted, producing a
genuine 404 the agent must diagnose and repair.

**Documentation sources that work:** server-rendered pages. Verified —
`docs.github.com/en/rest/repos/repos`, `pokeapi.co/docs/v2`, `open-meteo.com/en/docs`,
`restcountries.com`, `jsonplaceholder.typicode.com`.
JavaScript-rendered doc sites (Jikan, SpaceX, NASA, TheCatAPI) return an empty shell to a
plain HTTP request and are reported as unusable — see *Known limitations*.

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                    # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── selector.py           # goal -> endpoint section (top-3 / deny)
│   │   ├── global_sections.py    # sections documenting pagination/auth globally
│   │   ├── schema_check.py       # is the extracted schema based on real docs?
│   │   ├── state.py              # AgentState (error_history has an append reducer)
│   │   ├── graph.py              # the cyclic StateGraph
│   │   ├── nodes.py              # generate / execute / diagnose / persist / route
│   │   ├── codegen.py            # schema + goal -> Python fetch script
│   │   ├── diagnose.py           # error + relevant docs + history -> corrected script
│   │   └── schemas.py            # ApiSchema — the comprehension contract
│   ├── rag/
│   │   ├── fetcher.py            # URL/PDF -> clean text + usability verdict + hash
│   │   ├── chunker.py            # section-aware -> LangChain Documents
│   │   ├── doc_cache.py          # freshness: content hash + TTL
│   │   ├── doc_identity.py       # human-readable document names
│   │   ├── embeddings.py vectorstore.py retriever.py
│   │   ├── extractor.py          # chunks -> ApiSchema
│   │   └── qa.py                 # Q&A over the docs (optionally section-scoped)
│   ├── db/persist.py             # TRUSTED: validate -> upsert -> verify (holds DB_URL)
│   ├── sandbox/                  # docker_runner, local_runner, runner (USE_SANDBOX)
│   └── config/settings.py        # single source of truth
├── frontend/streamlit_app.py     # the UI (calls backend functions only)
├── sandbox_image/Dockerfile      # python:3.11-slim + requests, non-root
├── data/chroma_db/               # vector store (gitignored — generated)
├── main.py discover.py test_sandbox_isolation.py
├── requirements.txt  .env  README.md  DESIGN.md
```

---

## Configuration (`backend/config/settings.py`)

| Setting | Default | Purpose |
|---|---|---|
| `DOC_TTL_DAYS` | 30 | how long an unchanged indexed doc stays fresh |
| `KEY_STRATEGY` | `derived` | how a record's identity key is chosen (`derived` / `llm`) |
| `CONFIRM_ENDPOINT` | True | CLI: confirm the selected endpoint before continuing |
| `MAX_RETRIES` | 5 | bounded retry budget for the self-healing loop |
| `FORCE_FAILURE` | False | demo lever: corrupt the first script to force a real error |
| `USE_SANDBOX` | True | Docker sandbox vs. local execution fallback |
| `SANDBOX_MEM_LIMIT` / `SANDBOX_CPUS` / `SANDBOX_TIMEOUT` | 256m / 0.5 / 30s | container caps |
| `GLOBAL_CONTEXT_MAX_CHUNKS` | 8 | cap on global-section chunks added to extraction |
| `QA_TOP_K` | 8 | chunks retrieved per documentation question |
| `CHUNK_MAX_CHARS` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1200 / 800 / 100 | chunking |
| `*_MAX_OUTPUT_TOKENS` | — | LLM output budgets per task |

---

## Known limitations

These are measured and deliberately unfixed — see `DESIGN.md` for the evidence and the
trigger that would justify fixing each.

- **Egress is not restricted.** The sandbox blocks host filesystem access and caps
  resources, but bridge networking lets the container reach any host.
- **Most modern API doc sites are JavaScript-rendered** and cannot be read by the HTML
  path. OpenAPI/Swagger support is the highest-value planned fix.
- **Tables are flattened, not parsed**, and the **PDF path is untested**.
- **No DB-level schema enforcement** — JSONB accepts any valid JSON; we validate the
  batch's structure, not per-record fields.
- **The Groq fallback cannot serve large prompts** (8000 TPM vs. ~9700-token extraction),
  so it fails on exactly the heaviest requests.
- **Provider fallback is non-deterministic** — Gemini and Groq can select different
  endpoints on identical input.
- **Changing `KEY_STRATEGY` invalidates idempotency** for already-stored rows.
- **Self-correction is proven against an injected error**, not yet a spontaneous one.
- **Single-page pagination only**; multi-page "fetch all" with backoff is deferred.
- **Documentation is not versioned** — re-indexing replaces the previous version.

---

## Development line

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs → validated `ApiSchema` | done |
| 2 | Schema → generated script → real fetch | done |
| 3 | Docker sandbox (contained execution) | done |
| 4 | Self-healing loop (LangGraph cyclic graph) | done |
| 4.1 | Endpoint selection from the goal (top-3 / deny / confirm) | done |
| 4.2 | `persist_and_verify` — Postgres JSONB, idempotent upserts | done |
| 4.3 | Doc caching, fetchability verdicts, schema sanity check | done |
| 4.4 | Streamlit interface | done |
| 4.5 | OpenAPI/Swagger input, table parsing, PDF headings | planned |
| 5 | HITL: on repeated failure offer a different endpoint; checkpoint/resume | planned |
| 6 | Eval harness (goals × APIs, scored) | planned |
| 7 | AWS deployment & hardening; egress lockdown | planned |
| 8 | Safe write operations (idempotency, HITL-before-mutate) | future |