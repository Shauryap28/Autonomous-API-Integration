"""
llm — the single seam for all LLM calls, with provider fallback.

Primary: Gemini (google-genai). On a transient overload (503) we retry with
backoff; if Gemini stays unavailable or rate-limits, we fall back to Groq
(openai/gpt-oss-120b). Centralizing here means extractor.py / codegen.py don't
know or care which provider answered.

  complete_text(prompt, max_tokens)               -> str
  complete_json(prompt, SchemaModel, max_tokens)  -> validated SchemaModel instance

Note: clients are created ONCE and reused. Creating a genai.Client per call lets
the wrapper be garbage-collected mid-request, which closes its httpx transport
("Cannot send a request, as the client has been closed").
"""
import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from backend.config import settings

_GEMINI_ERRORS = (genai_errors.ServerError, genai_errors.ClientError)
_RETRIES = 3
_BACKOFF_BASE = 2  # seconds -> 2, 4 (between 3 attempts)

# Module-level singletons (lazy). Held here so they are never GC'd mid-call.
_gemini = None
_groq = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        _gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini


def _groq_client():
    global _groq
    if _groq is None:
        from groq import Groq  # lazy import: only needed if the fallback fires
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "Gemini is unavailable and GROQ_API_KEY is not set — add it to "
                ".env to enable the Groq fallback."
            )
        _groq = Groq(api_key=settings.GROQ_API_KEY)
    return _groq


def _retry_gemini(call):
    """Run a Gemini call; retry transient 5xx with backoff; break early on 4xx."""
    last = None
    for i in range(_RETRIES):
        try:
            return call()
        except genai_errors.ServerError as e:       # 5xx incl. 503 overload
            last = e
            if i < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2 ** i))
        except genai_errors.ClientError as e:        # e.g. 429 — don't burn retries
            last = e
            break
    raise last


# ---------- text ----------

def complete_text(prompt, max_output_tokens, temperature=0):
    try:
        client = _gemini_client()
        resp = _retry_gemini(lambda: client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        ))
        return resp.text or ""
    except _GEMINI_ERRORS as e:
        print(f"[llm] Gemini unavailable ({_short(e)}); falling back to Groq.")
        resp = _groq_client().chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return resp.choices[0].message.content or ""


# ---------- json (structured) ----------

def complete_json(prompt, schema_model, max_output_tokens, temperature=0):
    """Return a validated `schema_model` instance (Gemini primary, Groq fallback)."""
    try:
        return _gemini_json(prompt, schema_model, max_output_tokens, temperature)
    except _GEMINI_ERRORS as e:
        print(f"[llm] Gemini unavailable ({_short(e)}); falling back to Groq.")
        return _groq_json(prompt, schema_model, max_output_tokens, temperature)


def _gemini_json(prompt, schema_model, max_output_tokens, temperature):
    client = _gemini_client()
    resp = _retry_gemini(lambda: client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    ))
    if resp.parsed is not None:
        return resp.parsed
    text = (resp.text or "").strip()
    if not text or _finish_reason(resp) == "MAX_TOKENS":
        raise RuntimeError(
            f"Gemini output truncated (finish_reason={_finish_reason(resp)}); "
            "raise the relevant *_MAX_OUTPUT_TOKENS setting."
        )
    return schema_model.model_validate_json(text)


def _groq_json(prompt, schema_model, max_output_tokens, temperature):
    # Groq JSON mode needs the literal word 'json' + the target shape in the prompt.
    augmented = (
        prompt
        + "\n\nRespond with ONLY a single valid json object that conforms to this "
        "JSON schema (no prose, no markdown):\n"
        + str(schema_model.model_json_schema())
    )
    resp = _groq_client().chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": augmented}],
        temperature=temperature,
        max_tokens=max_output_tokens,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    return schema_model.model_validate_json(content)


# ---------- helpers ----------

def _finish_reason(resp):
    try:
        return str(resp.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        return "unknown"


def _short(e):
    return str(e)[:80]