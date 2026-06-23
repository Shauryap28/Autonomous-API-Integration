"""
extractor — turn one endpoint's retrieved doc chunks into a validated ApiSchema.

Pulls ALL chunks for the target endpoint (metadata-filtered, so the other ~30
endpoints on the page are excluded), concatenates them in document order, and
asks Gemini to fill the ApiSchema via structured output. The SDK enforces the
schema; Pydantic gives us the validated object back.
"""
from google import genai
from google.genai import types

from backend.config import settings
from backend.agent.schemas import ApiSchema
from backend.rag.vectorstore import get_endpoint_chunks

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

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ApiSchema,
            temperature=0,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        ),
    )

    # response.parsed is the Pydantic object when response_schema is a model;
    # fall back to validating the raw JSON text if the SDK returns None.
    schema = resp.parsed
    if schema is None:
        schema = ApiSchema.model_validate_json(resp.text)
    return schema