"""
extractor — turn the relevant doc chunks into a validated ApiSchema.

Context sent to the LLM =
    the TARGET endpoint's chunks
  + the GLOBAL-concern sections' chunks (pagination / auth / rate limits), if the
    doc documents those separately (e.g. PokéAPI). On docs where each endpoint
    documents its own pagination (e.g. GitHub), there are no global sections and this
    is a no-op.

This keeps the Phase 1 endpoint filter intact (no sibling-endpoint contamination)
while adding back the shared context that endpoint-scoping alone would exclude.
"""
from backend.config import settings
from backend.agent.schemas import ApiSchema
from backend.rag.vectorstore import get_section_chunks
from backend.llm import complete_json

_INSTRUCTION = (
    "You are an API integration assistant. From the API documentation below, "
    "extract the integration schema for the endpoint that satisfies this goal:\n"
    "GOAL: {goal}\n\n"
    "Use ONLY the documentation provided. If a field is not stated in the docs, "
    "use the schema's default rather than guessing. Capture the path template "
    "(e.g. /orgs/{{org}}/repos), the auth method, required headers, the query/path "
    "parameters, and how pagination works (some docs describe pagination in a "
    "separate 'pagination' section — use it).\n\n"
    "DOCUMENTATION:\n{context}"
)


def extract_api_schema(vectorstore, goal, endpoint_section, global_sections=None):
    """Extract the ApiSchema from the endpoint's chunks + any global-concern chunks."""
    endpoint_chunks = get_section_chunks(vectorstore, endpoint_section)
    if not endpoint_chunks:
        raise ValueError(f"No chunks found for endpoint_section '{endpoint_section}'")

    context_parts = [f"### ENDPOINT: {endpoint_section}", *endpoint_chunks]

    if global_sections:
        # Cap how many global chunks we add so a big doc can't blow the token budget.
        global_chunks = get_section_chunks(vectorstore, global_sections)
        global_chunks = global_chunks[: settings.GLOBAL_CONTEXT_MAX_CHUNKS]
        if global_chunks:
            context_parts.append(f"\n### GLOBAL (pagination/auth/rate limits): {', '.join(global_sections)}")
            context_parts.extend(global_chunks)

    context = "\n\n".join(context_parts)
    prompt = _INSTRUCTION.format(goal=goal, context=context)
    return complete_json(prompt, ApiSchema, settings.EXTRACT_MAX_OUTPUT_TOKENS)