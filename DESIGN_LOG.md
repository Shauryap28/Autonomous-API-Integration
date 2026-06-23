# Design & Rationale Log

For each decision: **What → Why → Alternatives → Parameters → Trade-offs.**
Plus a running "problems faced", "known limitations", and per-phase personal notes.
This is the defensibility record — every choice here should be one I can justify.

---

## Phase 1 — Docs → `ApiSchema`

### D1. Section-aware chunking (not fixed-size)
- **What:** Split docs at markdown headings into sections; sub-split only a
  section longer than `CHUNK_MAX_CHARS` using `RecursiveCharacterTextSplitter`.
- **Why:** API docs group auth / parameters / pagination under their own
  headings. Keeping a section intact means a later query for one aspect retrieves
  the whole relevant block, not a fragment.
- **Alternatives:** Pure fixed-size splitting (what MDIS used) — rejected here
  because it slices a section mid-thought and hurts targeted retrieval. Semantic
  splitting — rejected as premature complexity.
- **Parameters:** `CHUNK_MAX_CHARS=1200`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`.
- **Trade-offs:** Depends on the doc having real headings; the char-window
  fallback covers docs that don't (e.g. plain-text PDFs), but loses structure.

### D2. LangChain `Document` stack (reuse from MDIS)
- **What:** `langchain_huggingface` (BGE-small), `langchain_chroma`,
  `RecursiveCharacterTextSplitter`, and `Document` as the universal shape.
- **Why:** Reuse the working MDIS RAG code; same ecosystem as LangGraph; clean
  seam (everything flows as `Document`).
- **Alternatives:** Raw `chromadb` + `sentence-transformers` — rejected: rewrites
  code we already own, for no gain.
- **Trade-offs:** A heavier dependency tree than raw libs; acceptable for reuse.

### D3. Endpoint-scoped retrieval via metadata filter  ⭐ (measured)
- **What:** Tag each chunk with `endpoint_section` (its enclosing `##` heading);
  retrieve with a ChromaDB `where` filter scoped to the target endpoint.
- **Why — the measurement:** The GitHub repos page is ONE doc covering **34
  endpoints / 427 chunks**; the target endpoint held only **13**. A probe query
  ("pagination, page and per_page") run two ways:
    - **Unfiltered: 3/3 top hits from the WRONG endpoints** (sibling list-repos
      endpoints whose Parameters sections embed almost identically).
    - **Filtered to the target endpoint: 0/3 contamination**, correct sections.
  Pure semantic similarity could not distinguish near-identical sibling
  endpoints; the metadata filter fixes it deterministically.
- **Alternatives:** Rely on similarity alone — rejected by the 3/3 result.
- **Trade-offs:** Requires reliable `endpoint_section` tagging (the `##` boundary).
- **Note:** Same class of bug — and same fix — as the multi-*document*
  interference handled in MDIS, applied one level down (endpoint within a doc).

### D4. Extraction uses ALL endpoint chunks, not per-aspect retrieval
- **What:** For extraction, fetch every chunk of the target endpoint (via the
  metadata filter) in document order and send them whole to Gemini.
- **Why:** The endpoint section is small (~13 chunks); the big cost win (427→13)
  is already banked by the filter. Within a tiny section, completeness beats
  token-shaving — per-aspect similarity could silently drop a field-bearing chunk.
- **Alternatives:** 5 per-aspect similarity queries unioned — rejected: more
  complex, and risks dropping a needed chunk. (The similarity `retriever` is kept
  in the codebase for Phase 4 diagnosis, where it is genuinely needed.)
- **Trade-offs:** Doesn't scale to a huge single-endpoint section; revisit then.

### D5. Structured output via Gemini `response_schema` (Pydantic)
- **What:** `generate_content` with `response_mime_type="application/json"` +
  `response_schema=ApiSchema`, `temperature=0`; read `response.parsed`.
- **Why:** SDK-enforced JSON matching the schema is the most reliable path;
  Pydantic validates. Schema is NOT duplicated in the prompt (degrades quality).
- **Alternatives:** Free-form "return JSON" + `json.loads` — rejected (fragile);
  function-calling — heavier than a single extraction needs.
- **Parameters:** `temperature=0`, `MAX_OUTPUT_TOKENS=1024`.
- **Trade-offs:** Relies on the SDK handling nested models; a `model_validate_json`
  fallback covers the case where `response.parsed` is None.

### D6. `ApiSchema` shape — moderately strict
- **What:** Enums for `auth_method`/`pagination.type`; headers as `list[Header]`
  not an open dict; no `Optional` (defaults instead).
- **Why:** Enums give codegen a clean contract; controlled generation handles
  fixed-property objects/arrays more reliably than arbitrary maps; defaults avoid
  null-handling edge cases.
- **Trade-offs:** Too strict could fail on quirky docs; loosen if a real API breaks it.

---

## Hybrid search + reranker — evaluated, deferred (no gap to close)
- In MDIS, hybrid search was justified by a measured gap (dense missed 2/3 exact
  tokens). Here, after the endpoint filter, extraction got **every field right on
  the first try** — there is no measured gap.
- Also redundant by design: hybrid/reranker *rank* a candidate pool; extraction
  sends ALL endpoint chunks whole, so there is nothing to re-rank.
- **Trigger to revisit:** a future endpoint section too large to send wholesale
  AND a measured case where similarity drops a field-bearing chunk. Until then,
  not built. ("Evaluated and chose not to" is the outcome.)

---

## Known limitations (measured / acknowledged, not yet fixed)

1. **Code-example chunks are noisy.** cURL/JSON response examples fragment under
   char-window splitting. Harmless: schema fields live in prose sections, so the
   noisy chunks sit unused. Not worth special-casing yet.
2. **Tables are flattened, not parsed.** `fetcher` extracts `h1–h4, p, li, pre,
   code` — it does NOT handle `<table>`. Table cell text is captured only because
   `get_text()` flattens descendant text into the stream. Extraction worked
   because Gemini reconstructs param/description pairs from flattened prose, but a
   wide table could blur which value maps to which column.
   - **Trigger to fix:** params come out scrambled on some API → teach `fetcher`
     to render `<table>` rows as structured text (e.g. "name: org | required:
     true | desc: ...").
3. **PDF path is untested.** `fetch_pdf_file` (PyMuPDF) is wired but unused —
   GitHub fetched fine as a URL. PDFs return raw text with NO heading tags, so the
   chunker would fall through to the char-window fallback and lose endpoint
   structure (which the retrieval filter depends on).
   - **Trigger to fix:** an API that only ships PDF docs → add heading-detection
     heuristics for PDFs (font size / numbering).
4. **Single API tested.** Validated on GitHub only. Other APIs (PokeAPI/REST
   Countries, Jikan) arrive in later phases; the quirky one (Jikan) is introduced
   in Phase 4 so the self-healing loop has real inconsistency to handle.

---

## Problems faced & resolved
- **Predicted GitHub docs would be a JS shell (`thin=True`); they weren't**
  (245K chars, `thin=False`). Lesson: run the fetchability probe, don't assume.
- **Cross-endpoint retrieval contamination (3/3 wrong)** → fixed with the
  `endpoint_section` metadata filter (3/3 → 0/3). See D3.

---

## Personal note — Phase 1
- **Problem:** turn messy multi-endpoint API docs into a reliable, structured
  schema for one endpoint.
- **Cause of the one real snag:** a single docs page documents ~30 sibling
  endpoints with near-identical parameter sections, so semantic similarity alone
  retrieved the wrong endpoint.
- **Fix:** tag chunks with their enclosing endpoint and filter retrieval by it —
  the same metadata-scoping idea from MDIS, reused one level down.
- **Takeaway:** the win came from *measuring* the contamination (3/3 → 0/3), not
  from guessing. And reusing a proven pattern beat inventing a new one.
