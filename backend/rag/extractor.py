"""
extractor — turn one endpoint's retrieved doc chunks into a validated ApiSchema.

Pulls ALL chunks for the target endpoint (metadata-filtered, so the other ~30
endpoints on the page are excluded), concatenates them in document order, and
asks the LLM seam to fill the ApiSchema. Provider choice + fallback live in
backend/llm.py — this module is provider-agnostic.
"""
from backend.config import settings
from backend.agent.schemas import ApiSchema
from backend.rag.vectorstore import get_endpoint_chunks
from backend.llm import complete_json

_INSTRUCTION = (
    "You are an API integration assistant. From the API documentation below, "
    "extract the integration schema for the endpoint that satisfies this goal:\n"
    "GOAL: {goal}\n\n"
    "Use ONLY the documentation provided. If a field is not stated in the docs, "
    "use the schema's default rather than guessing. Capture the path template "
    "(e.g. /orgs/{{org}}/repos), the auth method, required headers, the query/path "
    "parameters, and how pagination works.\n\n"
    "DOCUMENTATION:\n{context}"
)


def extract_api_schema(vectorstore, goal, endpoint_section):
    chunks = get_endpoint_chunks(vectorstore, endpoint_section)
    if not chunks:
        raise ValueError(f"No chunks found for endpoint_section '{endpoint_section}'")

    context = "\n\n".join(chunks)
    prompt = _INSTRUCTION.format(goal=goal, context=context)
    return complete_json(prompt, ApiSchema, settings.EXTRACT_MAX_OUTPUT_TOKENS)