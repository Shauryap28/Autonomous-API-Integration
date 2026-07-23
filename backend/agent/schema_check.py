"""
schema_check — is the extracted ApiSchema actually usable, or did the LLM invent it?

Motivation (measured): Open Library's "developers/api" page is a NAVIGATION page — it
lists and links to APIs but documents none of them. It passes the fetchability verdict
(9 headings, plenty of text) yet contains no paths, no parameters, no base URL. The
extractor had nothing real to work with, so the LLM produced a placeholder host
(api.example.com). The agent then burned attempts trying to call a domain that does not
exist.

"Has headings" is not the same as "documents endpoints". This catches the difference
AFTER extraction, where the evidence is concrete, using cheap deterministic checks —
no extra LLM call.

Returns a list of warnings. Empty list == the schema looks real.
"""

# hosts that documentation uses as stand-ins — a real base_url is never one of these
_PLACEHOLDER_HOSTS = (
    "example.com", "example.org", "example.net", "api.example",
    "yourdomain", "your-domain", "your-api", "myapi", "my-api",
    "localhost", "127.0.0.1", "<host>", "{host}", "hostname",
)


def check_schema(schema):
    """Return a list of human-readable warnings about an extracted ApiSchema."""
    warnings = []
    base = (schema.base_url or "").lower()
    endpoint = (schema.endpoint or "").strip()

    if not base:
        warnings.append("No base URL was found in the documentation.")
    elif any(h in base for h in _PLACEHOLDER_HOSTS):
        warnings.append(
            f"The base URL '{schema.base_url}' looks like a placeholder rather than a "
            "real host — the page probably does not document this endpoint."
        )
    elif not base.startswith(("http://", "https://")):
        warnings.append(f"The base URL '{schema.base_url}' is not a valid URL.")

    if not endpoint:
        warnings.append("No endpoint path was found in the documentation.")
    elif not endpoint.startswith("/"):
        warnings.append(f"The endpoint path '{endpoint}' does not look like a path.")

    if not schema.parameters and schema.pagination.type.value == "none":
        warnings.append(
            "No parameters and no pagination were found — this section may be prose or "
            "a navigation page rather than an endpoint reference."
        )
    return warnings


def summarise(warnings):
    """One-line verdict for the CLI/UI."""
    if not warnings:
        return ""
    return ("This schema may not be based on real endpoint documentation:\n  - "
            + "\n  - ".join(warnings))