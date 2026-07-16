"""
global_sections — identify which doc sections document GLOBAL concerns
(pagination, auth, rate limits) rather than one specific endpoint.

Why this exists: our extractor scopes retrieval to the TARGET endpoint's section —
which is correct for docs like GitHub (each endpoint documents its own pagination),
but WRONG for docs like PokéAPI, where pagination lives in a separate shared section.
Scoping to the endpoint then EXCLUDES the pagination docs, so the extractor concludes
"pagination: none" and codegen falls back to one-request-per-record.

This adds the global sections back into the extraction context — ADDITIVELY. It does
NOT loosen the endpoint filter (the Phase 1 fix against sibling-endpoint contamination
stays intact). Pagination docs are shared context every endpoint needs, not a
competing endpoint, so including them is safe.

On docs where pagination is per-endpoint (GitHub), the LLM returns none -> no-op.
"""
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm import complete_json


class GlobalSections(BaseModel):
    sections: list[str] = Field(
        default_factory=list,
        description="section names (copied EXACTLY from the list) that document "
                    "global/shared concerns like pagination, authentication, or rate "
                    "limits — NOT a specific data endpoint. Empty if there are none.",
    )


_INSTRUCTION = """You are analyzing the structure of an API's documentation.

Below is the list of documentation section names.

SECTIONS:
{sections}

TASK:
Return the names of sections that document GLOBAL, SHARED concerns that apply across
many endpoints — specifically PAGINATION, AUTHENTICATION, or RATE LIMITING — rather
than one specific data endpoint.

Rules:
- Copy names EXACTLY from the list above.
- Include a section like "Resource Lists/Pagination" or "Authentication".
- Do NOT include endpoint sections (e.g. "List organization repositories", "Pokémon").
- If pagination/auth/rate-limits are NOT documented in their own separate sections
  (i.e. each endpoint documents its own), return an EMPTY list.
"""


def identify_global_sections(section_names, exclude=None):
    """Return the subset of section_names that document global concerns.

    `exclude` (the target endpoint) is never returned, so we can't accidentally
    duplicate the endpoint's own chunks.
    """
    listing = "\n".join(f"- {name}" for name in section_names)
    prompt = _INSTRUCTION.format(sections=listing)
    result = complete_json(prompt, GlobalSections, settings.SELECT_MAX_OUTPUT_TOKENS)

    valid = set(section_names)
    exclude = exclude or ""
    # keep only real names, drop the target endpoint if the LLM listed it
    return [s for s in result.sections if s in valid and s != exclude]