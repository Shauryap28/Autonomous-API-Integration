# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
generates and runs code to fetch the requested data, self-corrects from real HTTP
errors, and escalates to a human when genuinely stuck. Built on a LangGraph cyclic
state machine.

This README describes the **current state** of the project, not a change history.

---

## Current capability (Phase 1 complete)

The system can take one API's documentation and a goal, and produce a clean,
validated **`ApiSchema`** — the structured "cheat-sheet" describing how to call a
single endpoint (auth, path, headers, parameters, pagination, response shape).

Phase 1 is the **doc-comprehension layer**. It does not yet generate or run code —
that begins in Phase 2.

### What works today
- Fetch API docs from a **URL** or a local **PDF/HTML** file, normalized to clean
  text with markdown headings preserved.
- **Section-aware chunking** that tags every chunk with its enclosing endpoint.
- Local **BGE-small** embeddings + **ChromaDB** vector store.
- **Endpoint-scoped retrieval** via a metadata filter, so a multi-endpoint docs
  page doesn't contaminate results with the wrong endpoint's sections.
- **Structured extraction** with Gemini → a Pydantic-validated `ApiSchema`.

### Validated on
- **GitHub REST API**, endpoint *"List organization repositories"*
  (`GET /orgs/{org}/repos`). Extraction returns the correct auth method, path,
  `Accept` header, parameters, page-based pagination, and response shape.

---

## Architecture (Phase 1 pipeline)

```
docs URL / PDF  +  natural-language goal
        |
        v
  fetcher      -> clean text, headings as markdown (#, ##, ###)
        |
        v
  chunker      -> section-aware chunks, each tagged with {endpoint_section,
        |          section_title, chunk_index, doc_name}  (LangChain Documents)
        v
  embeddings   -> BGE-small-en-v1.5 (local, free)
        |
        v
  vectorstore  -> ChromaDB (persistent, cosine)
        |
        v
  retrieval    -> metadata filter to the TARGET endpoint's chunks only
        |
        v
  extractor    -> Gemini structured output -> ApiSchema (Pydantic validated)
        |
        v
  ApiSchema    -> the contract every later phase will read
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── agent/
│   │   ├── schemas.py        # ApiSchema — the Phase 1 output contract
│   │   └── state.py          # AgentState skeleton (filled in later phases)
│   ├── rag/
│   │   ├── fetcher.py        # URL/PDF -> clean text + fetchability report
│   │   ├── chunker.py        # section-aware -> LangChain Documents
│   │   ├── embeddings.py     # BGE-small (reused from MDIS)
│   │   ├── vectorstore.py    # ChromaDB wrapper + get_endpoint_chunks
│   │   ├── retriever.py      # endpoint-scoped similarity retrieval
│   │   └── extractor.py      # retrieved chunks -> Gemini -> ApiSchema
│   └── config/
│       └── settings.py       # single source of truth for config
├── data/chroma_db/           # local vector store (gitignored)
├── tests/                    # (test_rag.py — Phase 1 acceptance test)
├── main.py                   # end-to-end Phase 1 runner
├── requirements.txt
└── .env                      # GEMINI_API_KEY, GROQ_API_KEY
```

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your keys
```

`.env` needs:
```
GEMINI_API_KEY=...            # used now (extraction)
GROQ_API_KEY=...              # reserved for Phase 2+
```

## Run

```bash
python main.py                       # default: GitHub repos docs URL
python main.py path/to/doc.pdf       # a saved PDF/HTML/markdown file
```

Prints the validated `ApiSchema` for the target endpoint. First run downloads
the BGE-small model (~130MB).

---

## The `ApiSchema` contract

| Field | Meaning |
|---|---|
| `auth_method` | `none` / `api_key` / `bearer` / `oauth2` |
| `base_url`, `endpoint`, `http_method` | where and how to call it |
| `required_headers` | list of `{name, value}` |
| `parameters` | list of `{name, location, required, description}` |
| `pagination` | `{type, param_names, notes}` |
| `response_data_path` | JSON path to the records (empty = top-level array) |
| `rate_limit`, `success_criteria`, `notes` | extras used by later phases |

---

## Roadmap (next)
- **Phase 2** — generate a Python fetch script from the schema + goal, run it
  locally, get one real successful fetch (no sandbox, no loop yet).
- **Phase 3** — move execution into a Docker sandbox.
- **Phase 4** — the self-healing loop (diagnose real errors, retry).
- **Phase 5** — human-in-the-loop + checkpointing.
- **Phase 7** — AWS cloud deployment & hardening.
