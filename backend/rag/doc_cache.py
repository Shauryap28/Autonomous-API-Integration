"""
doc_cache — decide whether an already-indexed document can be reused.

Freshness is judged by TWO independent signals, because either alone is wrong:

  content hash : we always fetch (cheap, ~1 s) and hash the extracted text. A different
                 hash means the documentation genuinely CHANGED -> re-embed, regardless
                 of age. A pure time check would happily serve docs that changed today.
  age (TTL)    : even when the hash matches, anything older than DOC_TTL_DAYS is
                 refreshed. A backstop so the store cannot ossify.

Age is not tracked or incremented — `embedded_at` is written once at index time and the
age is computed on demand as (now - embedded_at), in UTC.

Four states, each with a sensible default the user can override:
    not_cached : never seen  -> index
    fresh      : hash matches, within TTL -> REUSE (instant)
    changed    : hash differs -> re-index (docs updated)
    stale      : hash matches but older than TTL -> re-index
"""
from datetime import datetime, timezone

from backend.config import settings

NOT_CACHED, FRESH, CHANGED, STALE = "not_cached", "fresh", "changed", "stale"


def _parse(ts):
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def describe_age(embedded_at):
    """'3 days ago' / 'today' — for display."""
    dt = _parse(embedded_at)
    if dt is None:
        return "unknown"
    days = (datetime.now(timezone.utc) - dt).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def get_cached_meta(vectorstore, doc_url):
    """Metadata of ONE chunk of this document (they all share the doc-level fields)."""
    data = vectorstore.get(where={"doc_url": doc_url}, limit=1, include=["metadatas"])
    metas = data.get("metadatas") or []
    return metas[0] if metas else None


def check_cache(vectorstore, doc_url, fresh_hash):
    """Return (state, meta). `fresh_hash` is the hash of the doc we JUST fetched."""
    meta = get_cached_meta(vectorstore, doc_url)
    if not meta:
        return NOT_CACHED, None

    if meta.get("content_hash") and meta["content_hash"] != fresh_hash:
        return CHANGED, meta

    dt = _parse(meta.get("embedded_at", ""))
    if dt is None:
        return STALE, meta
    age_days = (datetime.now(timezone.utc) - dt).days
    return (STALE if age_days > settings.DOC_TTL_DAYS else FRESH), meta


def explain(state, meta):
    """A one-line, human-readable verdict for the CLI/UI."""
    if state == NOT_CACHED:
        return "Not indexed yet — it will be embedded now."
    age = describe_age((meta or {}).get("embedded_at", ""))
    if state == FRESH:
        return f"Cached and current (indexed {age}) — reusing the existing index."
    if state == CHANGED:
        return f"The documentation has CHANGED since it was indexed ({age}) — re-indexing."
    return (f"Cached but older than {settings.DOC_TTL_DAYS} days (indexed {age}) "
            "— re-indexing.")


def should_reindex(state):
    return state in (NOT_CACHED, CHANGED, STALE)