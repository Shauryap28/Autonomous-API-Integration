"""
main.py — docs URL + goal -> comprehend -> generate -> run (sandboxed) -> persist.

Documents are CACHED: a doc that is already indexed, unchanged, and within
DOC_TTL_DAYS is reused instantly instead of being re-embedded.

    python main.py
    python main.py <docs-url-or-file> "your natural-language goal"
    python main.py <docs-url-or-file> "goal" --reindex     # force a fresh index
"""
import json
import sys

from backend.config import settings
from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import (
    get_vectorstore, add_documents, count, reindex_doc, get_doc_sections,
)
from backend.rag import doc_cache
from backend.rag.extractor import extract_api_schema
from backend.agent.selector import select_endpoint, confirm_endpoint
from backend.agent.global_sections import identify_global_sections
from backend.agent.schema_check import check_schema, summarise
from backend.rag.doc_identity import doc_display_name
from backend.agent.graph import build_graph

DEFAULT_SRC = "https://docs.github.com/en/rest/repos/repos"
DEFAULT_GOAL = "List the first 50 public repositories of the 'github' organization."


def prepare_docs(vectorstore, src, force_reindex=False):
    """Fetch, decide freshness, and index only if needed. Returns (sections, ok)."""
    r = fetch(src)
    print(f"fetch:  {r.verdict.upper()} — {r.message}")
    if not r.ok:
        return [], False
    print(f"        chars={r.char_count:,}  headings={r.heading_count}")

    state, meta = doc_cache.check_cache(vectorstore, src, r.content_hash)
    if force_reindex and state == doc_cache.FRESH:
        print("cache:  fresh, but --reindex was given — re-indexing anyway.")
        state = doc_cache.STALE
    else:
        print(f"cache:  {doc_cache.explain(state, meta)}")

    if not doc_cache.should_reindex(state):
        return get_doc_sections(vectorstore, src), True

    docs = chunk(r.text, doc_display_name(src), doc_url=src, content_hash=r.content_hash)
    print(f"chunk:  {len(docs)} chunks")
    print("embed:  embedding (BGE-small)...")
    removed = reindex_doc(vectorstore, src, docs)   # delete-then-add
    if removed:
        print(f"store:  replaced {removed} old chunks")
    print(f"store:  {count(vectorstore)} chunks in the store")
    return get_doc_sections(vectorstore, src), True


def main():
    args = [a for a in sys.argv[1:] if a != "--reindex"]
    force_reindex = "--reindex" in sys.argv
    src = args[0] if args else DEFAULT_SRC
    goal = args[1] if len(args) > 1 else DEFAULT_GOAL

    print(f"Source: {src}")
    print(f"Goal:   {goal}\n")

    vectorstore = get_vectorstore(get_embeddings())
    sections_with_counts, ok = prepare_docs(vectorstore, src, force_reindex)
    if not ok:
        return
    sections = [name for name, _ in sections_with_counts]
    print(f"doc:    {len(sections)} sections available\n")

    print("select: matching your goal to a documentation section...")
    result = select_endpoint(goal, sections)
    target_endpoint = confirm_endpoint(result, sections)
    if target_endpoint is None:
        print("\nStopping: no suitable endpoint. Try a different goal or docs source.")
        return
    print(f"target: {target_endpoint}")

    global_sections = identify_global_sections(sections, exclude=target_endpoint)
    print(f"global: {', '.join(global_sections)}" if global_sections
          else "global: none (this doc documents concerns per-endpoint)")

    print("extract: asking the LLM to fill the ApiSchema...")
    schema = extract_api_schema(vectorstore, goal, target_endpoint, global_sections)
    print(f"schema:  OK  (pagination.type = {schema.pagination.type.value})")
    warnings = check_schema(schema)
    if warnings:
        print(f"\n!! {summarise(warnings)}")
        if input("\nContinue anyway? [y/N]: ").strip().lower() != "y":
            print("Stopping. Try a different section, or a documentation page that "
                  "actually specifies the endpoint.")
            return
    print()

    initial_state = {
        "goal": goal, "api_schema": schema.model_dump(),
        "current_code": "", "execution_result": {},
        "attempt_number": 0, "max_retries": settings.MAX_RETRIES,
        "error_history": [], "status": "running", "fetched_data": None,
        "rows_upserted": None, "rows_for_endpoint": None, "persist_error": None,
    }

    backend = "Docker sandbox" if settings.USE_SANDBOX else "local subprocess"
    print(f"--- agent graph (execution via {backend}"
          f"{'; FORCE_FAILURE on' if settings.FORCE_FAILURE else ''}) ---")

    graph = build_graph(vectorstore, target_endpoint, doc_display_name(src))
    final = graph.invoke(initial_state, {"recursion_limit": 50})

    print(f"\n=== outcome: {final['status'].upper()} in {final['attempt_number']} attempt(s) ===")

    for h in final.get("error_history", []):
        err = str(h["error"]).strip().splitlines()[-1] if h["error"] else "?"
        print(f"  attempt {h['attempt']} (exit {h['exit_code']}): {err[:110]}")

    if final.get("rows_upserted") is not None:
        print(f"\npersisted: {final['rows_upserted']} upserted; "
              f"{final['rows_for_endpoint']} rows for this endpoint")
    elif final.get("persist_error"):
        print(f"\npersist FAILED: {final['persist_error']}")

    data = final.get("fetched_data")
    if isinstance(data, list):
        print(f"\n--- fetched {len(data)} records; first 2 ---")
        print(json.dumps(data[:2], indent=2)[:700])


if __name__ == "__main__":
    main()