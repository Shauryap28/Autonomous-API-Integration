"""
main.py — docs -> ApiSchema (setup) -> LangGraph self-healing agent.

SETUP (once): fetch -> chunk -> embed -> store -> extract the ApiSchema.
GRAPH (loops): generate_code -> execute -> [success | diagnose_and_fix -> execute...]

Set settings.FORCE_FAILURE = True to break the first attempt on purpose and watch
the agent read the real error, re-check the docs, and repair itself.

Run from the repo root:
    python main.py                  # default: GitHub repos docs URL
    python main.py path/to/doc.pdf
"""
import json
import sys

from backend.config import settings
from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import get_vectorstore, add_documents, count, clear
from backend.rag.extractor import extract_api_schema
from backend.agent.graph import build_graph

DEFAULT_SRC = "https://pokeapi.co/docs/v2"
GOAL = "Fetch the first 50 Pokémon with their name and URL."
TARGET_ENDPOINT = "Pokémon (group)"


def _doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def comprehend(src, vectorstore):
    """SETUP (runs once): docs -> validated ApiSchema."""
    r = fetch(src)
    print(f"fetch:  chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    docs = chunk(r.text, _doc_name_from(src))
    print(f"chunk:  {len(docs)} chunks")
    print("embed:  loading BGE-small + embedding chunks...")
    clear(vectorstore)
    add_documents(vectorstore, docs)
    print(f"store:  {count(vectorstore)} chunks indexed")
    print("extract: asking the LLM to fill the ApiSchema...")
    schema = extract_api_schema(vectorstore, GOAL, TARGET_ENDPOINT)
    print("schema:  OK\n")
    return schema


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    print(f"Source: {src}")
    print(f"Goal:   {GOAL}\n")

    # The vectorstore is a LIVE connection: created here, injected into the graph
    # via a closure. It must never go into the graph state (state gets serialized).
    vectorstore = get_vectorstore(get_embeddings())
    schema = comprehend(src, vectorstore)

    initial_state = {
        "goal": GOAL,
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

    graph = build_graph(vectorstore, TARGET_ENDPOINT)
    final = graph.invoke(initial_state, {"recursion_limit": 50})

    # ---- trace summary ----
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
        result = final.get("execution_result", {})
        print("\n--- last stderr ---")
        print((result.get("stderr") or "")[:1000])


if __name__ == "__main__":
    main()