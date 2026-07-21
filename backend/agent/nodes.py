"""
nodes — the functions that make up the agent graph.

A node takes the state and returns a PARTIAL dict of updates. Nodes stay thin: they
call the modules we already built and record the result.

Nodes needing a live resource (vectorstore, DB context) are built as CLOSURES in
graph.py — state holds DATA, never live connections (state is serialized per checkpoint).
"""
import json

from backend.config import settings
from backend.agent.codegen import generate_code as _generate_code
from backend.agent.schemas import ApiSchema
from backend.agent.diagnose import diagnose_and_fix
from backend.sandbox.runner import get_runner
from backend.db.persist import persist_records


def _break_code(code):
    """FORCE_FAILURE (demo only): corrupt the endpoint path so the API returns a real 404."""
    return code.replace("/orgs/", "/org/", 1)


def generate_code_node(state):
    schema = ApiSchema.model_validate(state["api_schema"])
    code = _generate_code(schema, state["goal"])

    if settings.FORCE_FAILURE:
        code = _break_code(code)
        print("[node] generate_code -> code generated, then BROKEN on purpose (FORCE_FAILURE)")
    else:
        print(f"[node] generate_code -> {len(code)} chars")

    return {"current_code": code, "attempt_number": state.get("attempt_number", 0) + 1}


def execute_node(state):
    runner = get_runner()
    result = runner.run_script(state["current_code"])

    execution_result = {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    fetched, status = None, "failed"
    if result.exit_code == 0:
        try:
            fetched = json.loads(result.stdout)
            status = "success"
        except (json.JSONDecodeError, ValueError):
            execution_result["stderr"] += "\n[runner] exit 0 but stdout was not valid JSON"

    n = len(fetched) if isinstance(fetched, list) else "-"
    print(f"[node] execute (attempt {state['attempt_number']}) -> exit_code={result.exit_code}, records={n}")
    return {"execution_result": execution_result, "fetched_data": fetched, "status": status}


def make_diagnose_node(vectorstore, endpoint_section):
    """CLOSURE: the node needs the live vectorstore, which must NOT live in state."""

    def diagnose_node(state):
        attempt = state["attempt_number"]
        new_code, error_text, _ = diagnose_and_fix(state, vectorstore, endpoint_section)

        first_line = error_text.strip().splitlines()[-1] if error_text.strip() else "?"
        print(f"[node] diagnose_and_fix (after attempt {attempt}) -> read error: {first_line[:110]}")
        print("[node] diagnose_and_fix -> regenerated the script")

        return {
            "current_code": new_code,
            "error_history": [{
                "attempt": attempt,
                "error": error_text[:500],
                "exit_code": state["execution_result"].get("exit_code"),
            }],
            "attempt_number": attempt + 1,
        }

    return diagnose_node


def make_persist_node(source, endpoint):
    """CLOSURE: source/endpoint are setup context, not agent state.

    Runs in the TRUSTED backend — this is the only place holding DB credentials.
    """

    def persist_node(state):
        try:
            result = persist_records(
                data=state.get("fetched_data"),
                source=source,
                endpoint=endpoint,
                goal=state["goal"],
            )
            print(f"[node] persist_and_verify -> upserted {result['upserted']} record(s); "
                  f"table now holds {result['rows_for_endpoint']} row(s) for this endpoint")
            return {
                "rows_upserted": result["upserted"],
                "rows_for_endpoint": result["rows_for_endpoint"],
                "status": "persisted",
            }
        except Exception as e:
            print(f"[node] persist_and_verify -> FAILED: {str(e)[:200]}")
            return {"persist_error": str(e)[:500], "status": "persist_failed"}

    return persist_node


def route_result(state):
    """Conditional edge: where do we go after execute?"""
    if state["status"] == "success":
        print("[route] success -> persist_and_verify")
        return "success"

    if state["attempt_number"] >= state["max_retries"]:
        print(f"[route] failed, retries exhausted ({state['attempt_number']}/{state['max_retries']}) -> END")
        return "give_up"

    print(f"[route] failed (attempt {state['attempt_number']}/{state['max_retries']}) -> diagnose_and_fix")
    return "retry"