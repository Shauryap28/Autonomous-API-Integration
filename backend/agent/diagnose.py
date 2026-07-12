"""
diagnose — repair a failed script by reading the REAL error.

This is the heart of the self-healing loop. It reads three things:
  1. the actual error (stderr / exit_code) from the sandbox run
  2. the doc section RELEVANT TO THAT ERROR — retrieved by using the error text as
     a semantic query, scoped to our endpoint (a 401 pulls the auth section, a
     pagination bug pulls the pagination section). This is the genuine retrieval
     problem the Phase 1 retriever was built for.
  3. the error history — what was already tried and how it failed, so the agent
     does not repeat a fix that has already been proven wrong.

It then regenerates the FULL script (not a patch — regenerating whole is simpler
and avoids compounding broken edits).
"""
from backend.config import settings
from backend.llm import complete_text
from backend.rag.retriever import get_retriever
from backend.agent.codegen import _strip_fences

_DIAGNOSE_INSTRUCTION = """You are debugging a Python API-integration script that FAILED.

GOAL: {goal}

API SCHEMA (from the documentation):
{schema_json}

THE SCRIPT THAT FAILED:
{code}

THE ACTUAL ERROR (exit_code={exit_code}):
{error}

RELEVANT DOCUMENTATION (retrieved because it matches this error):
{doc_context}

{history_block}
YOUR TASK:
1. Work out WHY it failed, from the real error above — not from guesswork.
2. Write a CORRECTED, complete Python 3 script that fixes that cause.

REQUIREMENTS — follow exactly:
- Use ONLY the `requests` library and the standard library (`json`, `sys`).
- Build the URL from base_url + endpoint; substitute path params from the GOAL.
- Set ALL required headers from the schema.
- Call resp.raise_for_status() so HTTP errors surface as a non-zero exit code.
- Print the parsed JSON to stdout with print(json.dumps(data)). Nothing else on stdout.
- Do NOT repeat a fix listed under PREVIOUS ATTEMPTS — those already failed.
- Output ONLY the corrected Python code: no explanation, no markdown fences.
"""

_HISTORY_HEADER = "PREVIOUS ATTEMPTS (do NOT repeat these — they already failed):\n"


def _error_text(execution_result):
    stderr = (execution_result.get("stderr") or "").strip()
    stdout = (execution_result.get("stdout") or "").strip()
    return stderr or stdout or "(no error output captured)"


def _format_history(error_history):
    if not error_history:
        return ""
    lines = [_HISTORY_HEADER]
    for h in error_history:
        lines.append(
            f"- attempt {h.get('attempt')}: error was: {str(h.get('error'))[:300]}"
        )
    return "\n".join(lines) + "\n"


def retrieve_error_context(vectorstore, endpoint_section, error_text):
    """Use the ERROR as the search query -> the doc section that explains it."""
    retriever = get_retriever(vectorstore, endpoint_section=endpoint_section)
    hits = retriever.invoke(error_text[:500])
    if not hits:
        return "(no relevant doc section retrieved)"
    return "\n\n".join(
        f"[{h.metadata.get('section_title', '?')}]\n{h.page_content}" for h in hits
    )


def diagnose_and_fix(state, vectorstore, endpoint_section):
    """Return (new_code, error_text, doc_context) — the repaired script + what we read."""
    error_text = _error_text(state["execution_result"])
    doc_context = retrieve_error_context(vectorstore, endpoint_section, error_text)

    prompt = _DIAGNOSE_INSTRUCTION.format(
        goal=state["goal"],
        schema_json=state["api_schema"],
        code=state["current_code"],
        exit_code=state["execution_result"].get("exit_code"),
        error=error_text[:2000],
        doc_context=doc_context[:4000],
        history_block=_format_history(state.get("error_history", [])),
    )
    new_code = _strip_fences(
        complete_text(prompt, settings.CODEGEN_MAX_OUTPUT_TOKENS, temperature=0)
    )
    return new_code, error_text, doc_context