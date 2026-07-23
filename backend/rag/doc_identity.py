"""
doc_identity — human-readable names for documentation sources.

The old naming took the URL's last path segment, which produced useless labels:
    https://pokeapi.co/docs/v2              -> "v2"
    https://openlibrary.org/developers/api  -> "api"
Both meaningless in a list of stored documents.

We use the DOMAIN (the thing that identifies the API) plus the last meaningful path
segment when it adds information:
    https://pokeapi.co/docs/v2                    -> pokeapi.co/v2
    https://docs.github.com/en/rest/repos/repos   -> docs.github.com/repos
    https://openlibrary.org/developers/api        -> openlibrary.org/api
    https://restcountries.com/                    -> restcountries.com
    C:/docs/stripe_api.pdf                        -> stripe_api.pdf
"""
from pathlib import Path
from urllib.parse import urlparse

# path segments that carry no information about WHICH api this is
_NOISE = {"docs", "doc", "documentation", "api", "apis", "reference", "ref",
          "en", "developers", "developer", "guide", "guides", "latest", "index"}


def doc_display_name(source):
    """A short, recognisable label for a documentation source."""
    if not source.startswith(("http://", "https://")):
        return Path(source).name or source

    parsed = urlparse(source)
    domain = parsed.netloc.removeprefix("www.")
    segments = [s for s in parsed.path.split("/") if s]

    # keep the last segment that actually distinguishes this doc
    meaningful = [s for s in segments if s.lower() not in _NOISE]
    if meaningful:
        return f"{domain}/{meaningful[-1]}"
    if segments:
        return f"{domain}/{segments[-1]}"
    return domain