# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
works out which endpoint fits, generates code to fetch the data, runs that code inside
an isolated sandbox, and self-corrects from real HTTP errors — retrying until it
succeeds or exhausts a bounded budget. Built on a LangGraph cyclic state machine.

This README describes the **current state**, not a change history.

---

## Current capability

Give it a docs URL and a plain-English goal. It then:

1. **Reads the docs** — fetch, section-aware chunk, embed (BGE-small), index (ChromaDB).
2. **Picks the endpoint** — an LLM matches the goal against the doc's section list,
   proposes a ranked top-3 for you to confirm, or **denies the goal outright** if no
   endpoint can satisfy it (before spending anything on embedding or codegen).
3. **Comprehends it** — endpoint-scoped retrieval → a Pydantic-validated `ApiSchema`
   (auth, path, headers, params, pagination, response shape).
4. **Writes the code** — a runnable `requests` script from the schema + goal.
5. **Runs it safely** — inside a Docker container: no host filesystem, capped
   memory/CPU, read-only filesystem, non-root user, hard timeout.
6. **Fixes itself** — on failure, reads the real error, re-queries the docs *using that
   error* to find the section explaining it, rewrites the script, and retries (bounded,
   with an append-only memory of every past attempt so a failed fix is never repeated).

Not yet implemented: database sync (`persist_and_verify`), human-in-the-loop
escalation, checkpoint/resume.

### Validated on
- **GitHub REST API** (`/orgs/{org}/repos`) — page-based pagination, top-level array.
  Fetches 50 repos. With the endpoint path deliberately corrupted, the agent gets a
  real **404**, retrieves the relevant docs, corrects the path, and succeeds on attempt 2.
- **PokéAPI** (`/pokemon`) — offset/limit pagination, records nested under `results`.
  Succeeds with no pipeline changes. (See *Known limitations* — on this group-structured
  doc the agent sometimes produces a working but inefficient per-record fetch.)
- **Goal denial** — asked to "delete all repositories and drop the database" against
  GitHub's docs, the selector correctly refuses: no such endpoint exists, and it does
  not force a match onto the unrelated "Delete a repository" section.

---

## Architecture

```
docs URL  +  natural-language goal
        |
   [ SETUP — runs once ]
   fetcher -> chunker -> section list
                            |
                    select_endpoint  (LLM: top-3 or DENY)  <-- human confirms
                            |
             embeddings -> vectorstore -> endpoint-scoped retrieval
                            |
                     extractor -> ApiSchema (validated)
        |
   [ AGENT GRAPH — LangGraph, cyclic ]
   START -> generate_code -> execute -> <route_result>
                                ^             |
                                |             +-- success            -> END
                                |             +-- retries exhausted  -> END (failed)
                                |             +-- retry -> diagnose_and_fix --+
                                +----------------------------------------------+

   execute  = Docker sandbox (capped, read-only, non-root, no host mount)
   diagnose = real error + the doc section retrieved USING that error + error_history
              -> full script regeneration

   backend/llm.py underlies selection / extraction / codegen / diagnosis:
   Gemini primary, automatic Groq fallback on 503.
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                  # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── selector.py         # goal -> endpoint section (top-3 / deny)
│   │   ├── state.py            # AgentState (error_history has an append reducer)
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
├── data/chroma_db/             # vector store (gitignored — generated)
├── main.py                     # setup + agent graph
├── discover.py                 # list a doc's endpoint sections (no LLM)
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
python main.py                                     # defaults (GitHub repos)
python main.py <docs-url> "your goal in English"   # any docs, any goal
python discover.py <docs-url>                      # list a doc's sections (no LLM)
python test_sandbox_isolation.py                   # prove the sandbox guarantees
```

Examples:
```bash
python main.py https://pokeapi.co/docs/v2 "Fetch the first 50 Pokemon with their name and URL."
python main.py https://docs.github.com/en/rest/repos/repos "List the first 50 public repositories of the 'github' organization."
```

To watch the self-healing loop, set `FORCE_FAILURE = True` in `settings.py`: the first
generated script is deliberately corrupted, producing a genuine HTTP 404 the agent must
diagnose and repair.

---

## Configuration (settings.py)

| Setting | Purpose |
|---|---|
| `CONFIRM_ENDPOINT` | ask the human to confirm the selected endpoint (False = auto-accept top pick) |
| `MAX_RETRIES` | bounded retry budget for the self-healing loop (default 5) |
| `FORCE_FAILURE` | demo lever: corrupt the first script to force a real error |
| `USE_SANDBOX` | `True` = Docker sandbox; `False` = local execution fallback |
| `SANDBOX_*` | image, memory/CPU caps, timeout, read-only filesystem |
| `*_MAX_OUTPUT_TOKENS` | LLM output budgets (selection / extraction / codegen) |
| `TOP_K`, `CHUNK_*` | retrieval / chunking knobs |

---

## Known limitations

- **Group-structured docs weaken endpoint scoping.** GitHub documents one endpoint per
  heading; PokéAPI groups many under one (144 chunks under "Pokémon (group)"), and
  documents pagination in a *separate* section. Because retrieval is scoped to the
  selected section, the pagination docs can be excluded — in one run this produced a
  *working but inefficient* solution (50 individual requests instead of one paginated
  call). Fix scheduled for Phase 4.5 (cross-section retrieval for global concerns).
- **Egress is not restricted.** The sandbox blocks host filesystem access and caps
  resources, but default bridge networking lets the container reach any host, not just
  the target API. Phase 7.
- **Tables are flattened, not parsed**; the **PDF input path is untested**. Both are
  scheduled for Phase 4.5, measure-driven.
- **Self-correction is proven against an injected error**, not yet a spontaneous one:
  on both tested APIs, codegen succeeded first try, so the agent never made a mistake of
  its own to fix.
- **Single-page pagination only** — "first N" fits one page; true multi-page "fetch all"
  with backoff is deferred.

---

## Development line

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs -> validated `ApiSchema` | done |
| 2 | Schema -> generated script -> real fetch | done |
| 3 | Docker sandbox (contained execution) | done |
| 4 | Self-healing loop (LangGraph cyclic graph) | done |
| 4.1 | Endpoint selection from the goal (top-3 / deny / confirm) | done |
| 4.2 | `persist_and_verify` — sync fetched data into Postgres (JSONB) | next |
| 4.5 | Comprehension hardening: cross-section retrieval, tables, PDF headings | planned |
| 5 | Human-in-the-loop (`interrupt`) + checkpoint/resume | planned |
| 6 | Streamlit surface (browse endpoints, live trace) + eval harness | planned |
| 7 | AWS deployment & hardening; egress lockdown | planned |
| 8 | Safe write operations (idempotency, HITL-before-mutate) | future |