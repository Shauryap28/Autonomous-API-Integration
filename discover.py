"""
discover.py — generality probe for the comprehension layer.

Fetches + chunks ANY API doc and lists the endpoint sections it found.
NO LLM, NO embeddings — cheap, free, and it answers one question: does our
fetch->chunk pipeline generalize beyond GitHub, or is it secretly GitHub-shaped?

Run from the repo root:
    python discover.py <url-or-file>

Use the printed endpoint-section names to pick a TARGET_ENDPOINT for main.py.
"""
import sys
from collections import Counter

from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk


def main():
    if len(sys.argv) < 2:
        print("usage: python discover.py <url-or-file>")
        return

    src = sys.argv[1]
    print(f"Source: {src}\n")

    r = fetch(src)
    print(f"fetch: chars={r.char_count}  headings={r.heading_count}  thin={r.looks_thin}")
    if r.looks_thin:
        print("\n!! thin — likely a JS-rendered SPA that returns a shell to requests.")
        print("   Options: save the page as PDF/HTML and pass the file path, or")
        print("   find an OpenAPI/Swagger spec or raw-markdown version of the docs.")
        return

    doc_name = src.rstrip("/").split("/")[-1] or src
    docs = chunk(r.text, doc_name)
    counts = Counter(d.metadata["endpoint_section"] for d in docs)

    print(f"chunk: {len(docs)} chunks across {len(counts)} endpoint sections\n")
    print("endpoint sections  (chunks | name):")
    for name, c in counts.most_common():
        print(f"  {c:3d} | {name[:72]}")
    print("\nPick one of these names as TARGET_ENDPOINT in main.py to run the full pipeline.")


if __name__ == "__main__":
    main()