"""
fetcher — turn a doc source (URL or PDF) into clean raw text.

  HTML : requests + BeautifulSoup
  PDF  : PyMuPDF (import fitz)

FIRST thing to verify in Phase 1 (the fetchability check): is the chosen doc
fetchable as plain text, or is it a JS-rendered SPA that returns an empty shell
to `requests`? Prefer an OpenAPI/Swagger spec, raw markdown, or a PDF over
scraping a JS-heavy page.

TODO (Phase 1): fetch_url(url) -> str ; fetch_pdf(path) -> str ; detect source.
"""
"""
fetcher — turn a doc source into clean text with headings preserved as markdown.

  • http(s) URL  -> requests + BeautifulSoup
  • local .html  -> same HTML path, from disk
  • local .pdf   -> PyMuPDF (fitz)

Headings are normalized to markdown ('#', '##') so a SINGLE chunker can handle
both HTML and PDF. fetch() returns the text plus a small fetchability report so
we can see whether we got real content (not a JS-rendered shell) before trusting it.
"""
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADING_TAGS = ["h1", "h2", "h3", "h4"]
_THIN_THRESHOLD = 600  # chars; below this, assume a shell rather than the real doc


@dataclass
class FetchResult:
    text: str
    source: str
    char_count: int
    heading_count: int
    looks_thin: bool


def _html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in HEADING_TAGS:
            level = int(el.name[1])
            lines.append("\n" + "#" * level + " " + txt)
        elif el.name in ("pre", "code"):
            lines.append("    " + txt)
        else:
            lines.append(txt)
    return "\n".join(lines).strip()


def fetch_url(url, timeout=20):
    headers = {"User-Agent": "aaie-phase1/0.1 (doc fetcher)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return _html_to_text(resp.text)


def fetch_pdf_file(path):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    parts = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(parts).strip()


def fetch(source):
    """Dispatch on source type; return text + a fetchability report."""
    if source.startswith(("http://", "https://")):
        text = fetch_url(source)
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = fetch_pdf_file(p)
        elif suffix in (".html", ".htm"):
            text = _html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
        else:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()

    heading_count = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#"))
    return FetchResult(text, source, len(text), heading_count, len(text) < _THIN_THRESHOLD)
