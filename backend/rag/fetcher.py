"""
fetcher — turn a doc source into clean text with headings preserved as markdown.

  • http(s) URL  -> requests + BeautifulSoup
  • local .html  -> same HTML path, from disk
  • local .pdf   -> PyMuPDF (fitz)

Headings are normalised to markdown so a SINGLE chunker handles both HTML and PDF.

fetch() also returns a VERDICT — an upfront, honest answer to "can this project use
this URL?", so a user finds out before spending anything:

  usable     : real text with section headings — go ahead
  flat       : real text but ~no headings, so endpoint selection will be limited
  not_usable : a JavaScript-rendered shell (requests sees an empty page)
  error      : the site could not be reached at all

Network failures are caught and returned as a verdict rather than raised, so a dead
domain gives a one-line message instead of a stack trace.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADING_TAGS = ["h1", "h2", "h3", "h4"]
_THIN_THRESHOLD = 600   # chars; below this, assume a shell rather than the real doc
_FLAT_HEADINGS = 1      # at or below this, the doc has no usable section structure


@dataclass
class FetchResult:
    text: str
    source: str
    char_count: int
    heading_count: int
    looks_thin: bool
    verdict: str = "usable"     # usable | flat | not_usable | error
    message: str = ""
    content_hash: str = ""

    @property
    def ok(self):
        return self.verdict in ("usable", "flat")


def content_hash(text):
    """SHA-256 of the EXTRACTED text (not raw HTML).

    Hashing the cleaned text means we detect real documentation changes and ignore
    volatile markup (session ids, ad slots, timestamps) that would otherwise make the
    hash differ on every fetch. Costs ~1 ms on a 245k-char doc.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            lines.append("\n" + "#" * int(el.name[1]) + " " + txt)
        elif el.name in ("pre", "code"):
            lines.append("    " + txt)
        else:
            lines.append(txt)
    return "\n".join(lines).strip()


def fetch_url(url, timeout=20):
    headers = {"User-Agent": "aaie/0.1 (doc fetcher)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return _html_to_text(resp.text)


def fetch_pdf_file(path):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    parts = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(parts).strip()


def _error(source, message):
    return FetchResult("", source, 0, 0, True, verdict="error", message=message)


def fetch(source):
    """Dispatch on source type; never raises for network/file problems."""
    try:
        if source.startswith(("http://", "https://")):
            text = fetch_url(source)
        else:
            p = Path(source)
            if not p.exists():
                return _error(source, f"No such file: {p}")
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                text = fetch_pdf_file(p)
            elif suffix in (".html", ".htm"):
                text = _html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
            else:
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
    except requests.exceptions.ConnectionError:
        return _error(source, "Could not reach that host — the domain may be offline or misspelled.")
    except requests.exceptions.Timeout:
        return _error(source, "The request timed out.")
    except requests.exceptions.HTTPError as e:
        return _error(source, f"The server returned an error: {e}")
    except requests.exceptions.RequestException as e:
        return _error(source, f"Request failed: {str(e)[:150]}")
    except OSError as e:
        return _error(source, f"Could not read the file: {str(e)[:150]}")

    headings = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#"))
    thin = len(text) < _THIN_THRESHOLD

    if thin:
        verdict = "not_usable"
        message = (
            "This page returns almost no text to a plain HTTP request — it is very "
            "likely JavaScript-rendered. Try an OpenAPI/Swagger spec, a raw-markdown "
            "version of the docs, or save the page as a PDF and load the file."
        )
    elif headings <= _FLAT_HEADINGS:
        verdict = "flat"
        message = (
            "Usable, but this page has no section headings, so it is treated as one "
            "block. Endpoint selection will have nothing to choose between."
        )
    else:
        verdict = "usable"
        message = f"{headings} headings found — ready to use."

    return FetchResult(text, source, len(text), headings, thin,
                       verdict=verdict, message=message, content_hash=content_hash(text))