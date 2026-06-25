"""
codegen — turn an ApiSchema + goal into a runnable Python fetch script.

Fixed instruction template + variable slots (the schema JSON + the goal). The
LLM call goes through backend/llm.py (Gemini primary, Groq fallback), so this
module doesn't know which provider answered. Returns a clean Python script string.
"""
from backend.config import settings
from backend.llm import complete_text

_CODEGEN_INSTRUCTION = """You are a Python code generator for API integration.
Write a SINGLE runnable Python 3 script that fetches data from an API and prints it as JSON to stdout.

GOAL: {goal}

API SCHEMA (extracted from the documentation):
{schema_json}

REQUIREMENTS — follow exactly:
- Use ONLY the `requests` library and the standard library (`json`, `sys`).
- Build the URL from base_url + endpoint. Substitute any path parameters (e.g. {{org}})
  with the concrete value implied by the GOAL.
- Set ALL required headers exactly as given in the schema.
- Set the query parameters needed to satisfy the GOAL (e.g. the result count via the
  pagination params, plus any filter the goal implies).
- Call resp.raise_for_status() so HTTP errors surface as a non-zero exit code.
- Parse the JSON response and print it to stdout with print(json.dumps(data)).
  Print NOTHING else to stdout. Any diagnostics go to stderr.
- Do NOT write to any database, file, or other network target besides the API call.
- Output ONLY the Python code: no explanations, no prose, no markdown fences.
"""


def _strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def generate_code(api_schema, goal):
    prompt = _CODEGEN_INSTRUCTION.format(
        goal=goal,
        schema_json=api_schema.model_dump_json(indent=2),
    )
    text = complete_text(prompt, settings.CODEGEN_MAX_OUTPUT_TOKENS, temperature=0)
    return _strip_fences(text)