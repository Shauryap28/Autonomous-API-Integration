# Autonomous API Integration Engine

An agentic AI system that reads an API's documentation plus a natural-language goal,
generates and runs code to fetch the requested data, self-corrects from real HTTP
errors, and escalates to a human when genuinely stuck. Built on a LangGraph cyclic
state machine.

This README describes the **current state**, not a change history.

---

## Current capability (Phases 1-2 complete)

Given an API's docs (URL or PDF) and a goal, the system:
1. **comprehends the docs** into a validated `ApiSchema` (Phase 1), and
2. **generates a Python script from that schema and runs it** to fetch the data,
   printing it as JSON (Phase 2).

It does not yet sandbox execution, self-correct from errors, or sync to a database —
those are Phases 3, 4, and the persist step.

### What works today
- Fetch docs from a **URL** or local **PDF/HTML**, normalized to clean text with
  markdown headings.
- **Section-aware chunking** tagging each chunk with its enclosing endpoint.
- Local **BGE-small** embeddings + **ChromaDB**.
- **Endpoint-scoped retrieval** (metadata filter) — no cross-endpoint contamination.
- **Structured extraction** → a Pydantic-validated `ApiSchema`.
- **Code generation** from schema + goal → a runnable `requests` script.
- **Local execution** of that script, capturing stdout/stderr/exit code.
- **LLM resilience:** Gemini primary with 503 retry/backoff; automatic **Groq
  fallback** (`openai/gpt-oss-120b`) when Gemini is unavailable.

### Validated on
- **GitHub REST API**, *"List organization repositories"* — extracts the correct
  schema, generates a script, and fetches 50 real repos end-to-end (verified even
  under a live Gemini outage, via the Groq fallback).

---

## Architecture (current pipeline)

```
docs URL / PDF  +  natural-language goal
        |
   [ Phase 1: comprehension ]
        |
  fetcher ----> chunker ----> embeddings ----> vectorstore
   (clean text   (section-      (BGE-small)      (ChromaDB)
    + headings)   aware Docs)                         |
                                                      v
                                         retrieval (endpoint filter)
                                                      |
                                                      v
                                            extractor --> ApiSchema (validated)
        |
   [ Phase 2: action ]
        |
  codegen ----> local_runner ----> fetched JSON
  (schema+goal   (subprocess;
   -> script)     stdout/stderr/exit)

  backend/llm.py underlies extractor + codegen: Gemini primary, Groq fallback.
```

---

## Project layout
```
autonomous-api-integration-engine/
├── backend/
│   ├── llm.py                # LLM seam: Gemini primary + Groq fallback
│   ├── agent/
│   │   ├── schemas.py        # ApiSchema — the comprehension contract
│   │   ├── codegen.py        # schema + goal -> Python fetch script
│   │   └── state.py          # AgentState skeleton (later phases)
│   ├── rag/
│   │   ├── fetcher.py        # URL/PDF -> clean text + fetchability report
│   │   ├── chunker.py        # section-aware -> LangChain Documents
│   │   ├── embeddings.py     # BGE-small
│   │   ├── vectorstore.py    # ChromaDB + get_endpoint_chunks
│   │   ├── retriever.py      # endpoint-scoped similarity retrieval
│   │   └── extractor.py      # retrieved chunks -> ApiSchema
│   └── config/settings.py    # single source of truth
├── data/chroma_db/           # local vector store (gitignored)
├── tests/                    # test_rag.py
├── main.py                   # end-to-end runner (docs -> schema -> code -> fetch)
├── discover.py               # generality probe: list a doc's endpoints (no LLM)
├── requirements.txt
└── .env                      # GEMINI_API_KEY, GROQ_API_KEY
```

---

## Setup & run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# put GEMINI_API_KEY and GROQ_API_KEY in .env
```

```bash
python main.py                        # full pipeline on the GitHub repos endpoint
python discover.py <url-or-file>      # probe any doc's endpoint structure
```

First run downloads the BGE-small model (~130MB).

---

## Development line (roadmap)

| Phase | Capability | Status |
|---|---|---|
| 1 | Docs → validated `ApiSchema` | ✅ done |
| 2 | Schema → generated script → real fetch (local) | ✅ done |
| 3 | Docker sandbox (replace `local_runner`, contained) | next |
| 4 | Self-healing loop + multi-API gauntlet (PokeAPI / REST Countries / Jikan) | planned |
| **4.5** | **Comprehension hardening — see below** | **planned, measure-driven** |
| 5 | Human-in-the-loop + checkpointing | planned |
| 6 | Demo surface + eval harness | planned |
| 7 | AWS deployment & hardening (RDS, Parameter Store, CloudWatch) | planned |
| 8 | Safe write operations (POST/PUT/DELETE, idempotency, HITL-before-mutate) | future |

### Phase 4.5 — Comprehension hardening (scheduled after Phase 4)
Two known limitations are deliberately deferred to here, because the multi-API
gauntlet in Phase 4 is what brings the varied/messy docs that would actually
trigger them (measure-then-fix; built only if a real doc breaks):
- **Table parsing in `fetcher`.** Today HTML tables are *flattened* (cell text is
  captured, structure isn't). Worked on GitHub. If a future API's params come out
  scrambled, render `<table>` rows as structured text
  (`name: org | required: true | desc: ...`) instead of flattening.
- **PDF heading detection.** The PDF input path (PyMuPDF) is wired but untested;
  PDFs return text with no heading markers, so chunking falls back to fixed-size
  windows and loses endpoint structure. If an API ships PDF-only docs, add
  heading-detection heuristics (font size / numbering) so PDFs chunk like HTML.