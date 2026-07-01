"""
main.py — full pipeline: docs -> ApiSchema -> generated script -> run -> fetched JSON.

Execution backend is chosen by settings.USE_SANDBOX (Docker sandbox by default,
local runner as fallback). Run from the repo root:
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
from backend.agent.codegen import generate_code
from backend.sandbox.runner import get_runner

DEFAULT_SRC = "https://docs.github.com/en/rest/repos/repos"
GOAL = "List the first 50 public repositories of the 'github' organization."
TARGET_ENDPOINT = "List organization repositories"


def _doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    print(f"Source: {src}")
    print(f"Goal:   {GOAL}\n")

    # --- Phase 1: docs -> schema ---
    r = fetch(src)
    print(f"fetch:  chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    docs = chunk(r.text, _doc_name_from(src))
    print(f"chunk:  {len(docs)} chunks")
    print("embed:  loading BGE-small + embedding chunks...")
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    clear(vs)
    add_documents(vs, docs)
    print(f"store:  {count(vs)} chunks indexed")
    print("extract: asking the LLM to fill the ApiSchema...")
    schema = extract_api_schema(vs, GOAL, TARGET_ENDPOINT)
    print("schema:  OK\n")

    # --- Phase 2: schema -> code ---
    print("codegen: generating the fetch script...")
    code = generate_code(schema, GOAL)
    print("----- generated script -----")
    print(code)
    print("----------------------------\n")

    # --- Phase 3: run it (sandbox or local, per settings.USE_SANDBOX) ---
    backend = "Docker sandbox" if settings.USE_SANDBOX else "local subprocess"
    print(f"execute: running via {backend}...")
    runner = get_runner()
    result = runner.run_script(code)
    print(f"exit_code = {result.exit_code}")
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr[:2000])

    print("--- result ---")
    try:
        data = json.loads(result.stdout)
        if isinstance(data, list):
            print(f"fetched {len(data)} records; first 2:")
            print(json.dumps(data[:2], indent=2)[:1500])
        else:
            print(json.dumps(data, indent=2)[:1500])
    except (json.JSONDecodeError, ValueError):
        print("stdout was not valid JSON; raw stdout (first 1000 chars):")
        print(result.stdout[:1000])


if __name__ == "__main__":
    main()