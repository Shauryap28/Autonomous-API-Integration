"""
main.py — Phase 1 end-to-end: docs -> validated ApiSchema.

Run from the repo root:
    python main.py                  # default: GitHub repos docs URL
    python main.py path/to/doc.pdf

Pipeline: fetch -> chunk (section-aware, endpoint-tagged) -> embed (BGE-small)
-> store (ChromaDB) -> extract the target endpoint's schema via Gemini -> print.
"""
import sys

from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import get_vectorstore, add_documents, count, clear
from backend.rag.extractor import extract_api_schema

DEFAULT_SRC = "https://docs.github.com/en/rest/repos/repos"
GOAL = "List all public repositories of a GitHub organization."
TARGET_ENDPOINT = "List organization repositories"


def _doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    print(f"Source: {src}")
    print(f"Goal:   {GOAL}\n")

    r = fetch(src)
    print(f"fetch:  chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    if r.looks_thin:
        print("  !! thin — save as PDF and re-run: python main.py your.pdf")

    docs = chunk(r.text, _doc_name_from(src))
    n_target = sum(1 for d in docs if d.metadata["endpoint_section"] == TARGET_ENDPOINT)
    print(f"chunk:  {len(docs)} chunks; target '{TARGET_ENDPOINT}' -> {n_target} chunks")

    print("embed:  loading BGE-small + embedding chunks...")
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    clear(vs)
    add_documents(vs, docs)
    print(f"store:  {count(vs)} chunks indexed\n")

    print("extract: asking Gemini to fill the ApiSchema (temperature=0)...")
    schema = extract_api_schema(vs, GOAL, TARGET_ENDPOINT)

    print("\n=== Extracted ApiSchema ===")
    print(schema.model_dump_json(indent=2))


if __name__ == "__main__":
    main()