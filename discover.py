"""
discover.py — can this project use this documentation URL?

Fetches and chunks any doc (NO LLM, NO embeddings — free and instant) and reports:
  • a plain verdict: usable / flat / not usable / unreachable
  • the endpoint sections it found

Run from the repo root:
    python discover.py <url-or-file>
"""
import sys
from collections import Counter

from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk

_ICON = {"usable": "[OK]  ", "flat": "[WARN]", "not_usable": "[NO]  ", "error": "[FAIL]"}


def main():
    if len(sys.argv) < 2:
        print("usage: python discover.py <url-or-file>")
        return

    src = sys.argv[1]
    print(f"Source: {src}\n")

    r = fetch(src)
    print(f"{_ICON.get(r.verdict, '')} {r.verdict.upper()}: {r.message}")

    if not r.ok:
        return

    print(f"       chars={r.char_count:,}  headings={r.heading_count}")
    docs = chunk(r.text, src.rstrip('/').split('/')[-1] or src,
                 doc_url=src, content_hash=r.content_hash)
    counts = Counter(d.metadata["endpoint_section"] for d in docs)

    print(f"\nchunk: {len(docs)} chunks across {len(counts)} sections\n")
    print("sections  (chunks | name):")
    for name, c in counts.most_common():
        print(f"  {c:4d} | {name[:70]}")


if __name__ == "__main__":
    main()