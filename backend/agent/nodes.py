"""
nodes — the functions that make up the agent graph.

A LangGraph node takes the current state and returns a PARTIAL dict of updates;
LangGraph merges those back into the state. Nodes stay thin — they call the modules
we already built (codegen, the sandbox runner, diagnose) and record the result.

Nodes that need a live resource (the vectorstore) are built as CLOSURES in graph.py:
state holds DATA, never live connections (state is serialized into every checkpoint).
"""
import json

from backend.config import settings
from backend.agent.codegen import generate_code as _generate_code
from backend.agent.schemas import ApiSchema
from backend.agent.diagnose import diagnose_and_fix
from backend.sandbox.runner import get_runner


def _break_code(code):
    """FORCE_FAILURE (demo only): corrupt the endpoint path so the API returns a real 404.

    Our GitHub call succeeds first try, so nothing would naturally exercise the loop.
    This makes the self-heal demonstrable ON DEMAND with a genuine HTTP error from a
    real API — not a fake/simulated one.
    """
    return code.replace("/orgs/", "/org/", 1)


def generate_code_node(state):
    """First attempt: write the fetch script from the schema + goal."""
    schema = ApiSchema.model_validate(state["api_schema"])
    code = _generate_code(schema, state["goal"])

    if settings.FORCE_FAILURE:
        code = _break_code(code)
        print("[node] generate_code -> code generated, then BROKEN on purpose (FORCE_FAILURE)")
    else:
        print(f"[node] generate_code -> {len(code)} chars")

    return {"current_code": code, "attempt_number": state.get("attempt_number", 0) + 1}


def execute_node(state):
    """Run the current script in the sandbox and record the raw result."""
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
            # error_history has an APPEND reducer -> this entry is added, not overwritten
            "error_history": [{
                "attempt": attempt,
                "error": error_text[:500],
                "exit_code": state["execution_result"].get("exit_code"),
            }],
            "attempt_number": attempt + 1,
        }

    return diagnose_node


def route_result(state):
    """Conditional edge: where do we go after execute?"""
    if state["status"] == "success":
        print("[route] success -> END")
        return "success"

    if state["attempt_number"] >= state["max_retries"]:
        print(f"[route] failed, retries exhausted ({state['attempt_number']}/{state['max_retries']}) -> END")
        return "give_up"

    print(f"[route] failed (attempt {state['attempt_number']}/{state['max_retries']}) -> diagnose_and_fix")
    return "retry"