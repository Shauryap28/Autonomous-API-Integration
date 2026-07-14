"""
main.py — docs URL + goal -> the agent does the rest.

  SETUP (once):
    fetch -> chunk -> SELECT ENDPOINT (LLM proposes, human confirms) -> embed -> store
          -> extract the ApiSchema
  GRAPH (loops):
    generate_code -> execute -> [success | diagnose_and_fix -> execute -> ...]

The endpoint is no longer hardcoded: it is derived from the goal, with the human as a
check. An impossible goal is denied up front — before any embedding or codegen.

Run from the repo root:
    python main.py                          # uses DEFAULT_SRC + GOAL below
    python main.py <docs-url-or-file>
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
from backend.agent.graph import build_graph

DEFAULT_SRC = "https://docs.github.com/en/rest/repos/repos"
DEFAULT_GOAL = "List the first 50 public repositories of the 'github' organization."


def _doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def _section_names(docs):
    """Unique endpoint-section names, most chunks first (the doc's real structure)."""
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

    # --- fetch + chunk (cheap; gives us the section list) ---
    r = fetch(src)
    print(f"fetch:  chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    if r.looks_thin:
        print("  !! thin — likely a JS-rendered shell. Save as PDF and pass the file path.")
        return

    docs = chunk(r.text, _doc_name_from(src))
    sections = _section_names(docs)
    print(f"chunk:  {len(docs)} chunks across {len(sections)} sections")

    # --- select the endpoint from the goal (deny early if impossible) ---
    print("select: matching your goal to a documentation section...")
    result = select_endpoint(goal, sections)
    target_endpoint = confirm_endpoint(result, sections)
    if target_endpoint is None:
        print("\nStopping: no suitable endpoint. Try a different goal or docs source.")
        return
    print(f"target: {target_endpoint}\n")

    # --- embed + store + extract (only now — after we know the goal is achievable) ---
    print("embed:  loading BGE-small + embedding chunks...")
    vectorstore = get_vectorstore(get_embeddings())
    clear(vectorstore)
    add_documents(vectorstore, docs)
    print(f"store:  {count(vectorstore)} chunks indexed")

    print("extract: asking the LLM to fill the ApiSchema...")
    schema = extract_api_schema(vectorstore, goal, target_endpoint)
    print("schema:  OK\n")

    # --- run the agent graph ---
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
    }

    backend = "Docker sandbox" if settings.USE_SANDBOX else "local subprocess"
    print(f"--- agent graph (execution via {backend}"
          f"{'; FORCE_FAILURE on' if settings.FORCE_FAILURE else ''}) ---")

    graph = build_graph(vectorstore, target_endpoint)
    final = graph.invoke(initial_state, {"recursion_limit": 50})

    print(f"\n=== outcome: {final['status'].upper()} in {final['attempt_number']} attempt(s) ===")

    history = final.get("error_history", [])
    if history:
        print(f"\n--- error history ({len(history)} failed attempt(s)) ---")
        for h in history:
            err = str(h["error"]).strip().splitlines()[-1] if h["error"] else "?"
            print(f"  attempt {h['attempt']} (exit {h['exit_code']}): {err[:110]}")

    print("\n----- final script -----")
    print(final["current_code"])
    print("------------------------")

    data = final.get("fetched_data")
    if isinstance(data, list):
        print(f"\n--- result: fetched {len(data)} records; first 2 ---")
        print(json.dumps(data[:2], indent=2)[:1000])
    elif final["status"] != "success":
        print("\n--- last stderr ---")
        print((final.get("execution_result", {}).get("stderr") or "")[:1000])


if __name__ == "__main__":
    main()