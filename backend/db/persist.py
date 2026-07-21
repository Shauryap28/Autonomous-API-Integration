"""
persist — the TRUSTED persistence layer (runs OUTSIDE the sandbox).

Security shape: the generated script fetches and prints JSON; it never holds the DB
credentials and never touches Postgres. THIS module holds DB_URL and does the writing.
That separation is what makes "the sandbox never holds DB credentials" real.

Flow: structural validation -> ensure table -> upsert rows -> verify count.

Storage model: one row per fetched RECORD, stored whole in a JSONB column, so any
API's shape is accepted with no per-API table design.

Identity = (source, endpoint, record_key). Deliberately NOT the goal: the same repo
fetched by "first 50" and by "first 100" is the SAME record and must update, not
duplicate. The goal IS stored as a plain column for provenance — information, not identity.
"""
import hashlib
import json

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

from backend.config import settings
from backend.llm import complete_json


# ---------------------------------------------------------------- validation

class FetchedRecord(BaseModel):
    """Structural only: a record is a JSON object; ANY fields are allowed.

    We deliberately do NOT validate per-record fields — the whole point of JSONB is
    accepting arbitrary API shapes. This validates the ENVELOPE, not the contents.
    """
    model_config = ConfigDict(extra="allow")


def validate_batch(data):
    """Check the fetched payload is a non-empty list of JSON objects. Returns the list."""
    if data is None:
        raise ValueError("nothing to persist: fetched_data is None")
    if not isinstance(data, list):
        raise ValueError(
            f"expected a JSON array of records, got {type(data).__name__}. "
            "The goal should fetch a collection."
        )
    if not data:
        raise ValueError("fetch returned an empty list — nothing to persist")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"record {i} is {type(item).__name__}, expected a JSON object")
        FetchedRecord.model_validate(item)      # structural check
        json.dumps(item)                        # must be JSON-serializable for JSONB
    return data


# ---------------------------------------------------------------- record keys

_KEY_CANDIDATES = ("id", "name", "slug", "key", "uuid", "number", "full_name")


def _hash_record(record):
    """Fingerprint of the record's JSON — the fallback when no ID-like field exists.

    Stable for identical content, but CHANGES if the content changes, so a hash-keyed
    record that later changes will insert as new rather than update. Fallback only.
    """
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def derive_key(record, key_field=None):
    """Return a stable identifier for one record.

    With key_field (LLM strategy): use that field if present.
    Otherwise (derived strategy): first ID-like field, else a content hash.
    """
    if key_field and record.get(key_field) is not None:
        return str(record[key_field])
    for field in _KEY_CANDIDATES:
        value = record.get(field)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return _hash_record(record)


class KeyField(BaseModel):
    field: str = ""     # "" means: no single field identifies these records


def choose_key_field_llm(sample_record):
    """KEY_STRATEGY='llm': ask the LLM which field is the stable identifier."""
    prompt = (
        "Here is one record returned by an API:\n\n"
        f"{json.dumps(sample_record, indent=2)[:1500]}\n\n"
        "Which single top-level field is this record's STABLE UNIQUE IDENTIFIER — the "
        "value that stays the same when the record's other data changes? Return just "
        "that field name. If no single field identifies it, return an empty string."
    )
    try:
        result = complete_json(prompt, KeyField, 256)
        field = (result.field or "").strip()
        return field if field and field in sample_record else None
    except Exception as e:
        print(f"[persist] LLM key selection failed ({str(e)[:60]}); using derived keys.")
        return None


def resolve_key_field(records):
    """Pick the key strategy from settings; returns a field name or None (derived)."""
    if settings.KEY_STRATEGY == "llm" and records:
        field = choose_key_field_llm(records[0])
        print(f"[persist] key strategy: llm -> field '{field}'" if field
              else "[persist] key strategy: llm -> no single field; using derived")
        return field
    print("[persist] key strategy: derived (id -> name -> ... -> content hash)")
    return None


# ---------------------------------------------------------------- database

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fetched_records (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,
    endpoint    TEXT        NOT NULL,
    record_key  TEXT        NOT NULL,
    goal        TEXT,
    record      JSONB       NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, endpoint, record_key)
);
"""

_UPSERT = """
INSERT INTO fetched_records (source, endpoint, record_key, goal, record)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (source, endpoint, record_key)
DO UPDATE SET record = EXCLUDED.record,
              goal   = EXCLUDED.goal,
              fetched_at = now();
"""

_COUNT = "SELECT count(*) FROM fetched_records WHERE source = %s AND endpoint = %s;"


def persist_records(data, source, endpoint, goal):
    """Validate, upsert, and verify. Returns {validated, upserted, rows_for_endpoint}."""
    records = validate_batch(data)
    if not settings.DB_URL:
        raise RuntimeError("DB_URL is not set — add it to .env")

    key_field = resolve_key_field(records)
    rows = [
        (source, endpoint, derive_key(rec, key_field), goal, Jsonb(rec))
        for rec in records
    ]

    with psycopg.connect(settings.DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
            cur.executemany(_UPSERT, rows)
            conn.commit()
            cur.execute(_COUNT, (source, endpoint))
            total = cur.fetchone()[0]

    return {
        "validated": len(records),
        "upserted": len(rows),
        "rows_for_endpoint": total,   # re-run this and the number should NOT grow
    }