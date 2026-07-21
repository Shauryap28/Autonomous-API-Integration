"""
main.py — docs URL + goal -> comprehend -> generate -> run (sandboxed) -> persist.

  SETUP (once):
    fetch -> chunk -> select endpoint -> identify global sections
          -> embed -> store -> extract ApiSchema
  GRAPH (loops, then persists):
    generate_code -> execute -> [diagnose_and_fix -> execute ...] -> persist_and_verify

Run from the repo root:
    python main.py
    python main.py <docs-url-or-file> "your natural-language goal"
"""
import json
import sys

from backend.config import settings
from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import get_vectorstore, add_documents, count, clear
from backend.rag.extractor import extract_api_schema
from backend.agent.selector import select_endpoint, confirm_endpoint
from backend.agent.global_sections import identify_global_sections
from backend.agent.graph import build_graph

DEFAULT_SRC = "https://docs.github.com/en/rest/repos/repos"
DEFAULT_GOAL = "List the first 50 public repositories of the 'github' organization."


def _doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def _section_names(docs):
    counts = {}
    for d in docs:
        name = d.metadata["endpoint_section"]
        counts[name] = counts.get(name, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    goal = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GOAL

    print(f"Source: {src}")
    print(f"Goal:   {goal}\n")

    r = fetch(src)
    print(f"fetch:  chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    if r.looks_thin:
        print("  !! thin — likely a JS-rendered shell. Save as PDF and pass the file path.")
        return

    source = _doc_name_from(src)
    docs = chunk(r.text, source)
    sections = _section_names(docs)
    print(f"chunk:  {len(docs)} chunks across {len(sections)} sections")

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
    print()

    print("embed:  loading BGE-small + embedding chunks...")
    vectorstore = get_vectorstore(get_embeddings())
    clear(vectorstore)
    add_documents(vectorstore, docs)
    print(f"store:  {count(vectorstore)} chunks indexed")

    print("extract: asking the LLM to fill the ApiSchema...")
    schema = extract_api_schema(vectorstore, goal, target_endpoint, global_sections)
    print(f"schema:  OK  (pagination.type = {schema.pagination.type.value})\n")

    initial_state = {
        "goal": goal,
        "api_schema": schema.model_dump(),
        "current_code": "",
        "execution_result": {},
        "attempt_number": 0,
        "max_retries": settings.MAX_RETRIES,
        "error_history": [],
        "status": "running",
        "fetched_data": None,
        "rows_upserted": None,
        "rows_for_endpoint": None,
        "persist_error": None,
    }

    backend = "Docker sandbox" if settings.USE_SANDBOX else "local subprocess"
    print(f"--- agent graph (execution via {backend}"
          f"{'; FORCE_FAILURE on' if settings.FORCE_FAILURE else ''}) ---")

    graph = build_graph(vectorstore, target_endpoint, source)
    final = graph.invoke(initial_state, {"recursion_limit": 50})

    print(f"\n=== outcome: {final['status'].upper()} in {final['attempt_number']} attempt(s) ===")

    history = final.get("error_history", [])
    if history:
        print(f"\n--- error history ({len(history)} failed attempt(s)) ---")
        for h in history:
            err = str(h["error"]).strip().splitlines()[-1] if h["error"] else "?"
            print(f"  attempt {h['attempt']} (exit {h['exit_code']}): {err[:110]}")

    if final.get("rows_upserted") is not None:
        print(f"\n--- persistence ---")
        print(f"  upserted this run     : {final['rows_upserted']}")
        print(f"  rows for this endpoint: {final['rows_for_endpoint']}   "
              f"(re-run: this should NOT grow)")
    elif final.get("persist_error"):
        print(f"\n--- persistence FAILED ---\n  {final['persist_error']}")

    data = final.get("fetched_data")
    if isinstance(data, list):
        print(f"\n--- fetched {len(data)} records; first 2 ---")
        print(json.dumps(data[:2], indent=2)[:800])


if __name__ == "__main__":
    main()