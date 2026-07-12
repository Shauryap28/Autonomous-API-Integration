"""
AgentState — the typed, shared state that flows through the LangGraph state machine.

Every node receives the whole state and returns a PARTIAL dict of just the fields
it changed; LangGraph merges those updates back in.

Merge rules matter:
  • plain fields          -> last write wins (overwrite)
  • Annotated[..., add]   -> APPEND (concatenate) instead of overwrite

`error_history` MUST use the append reducer. Without it, each diagnose step would
overwrite the record of past failures — and that memory is precisely what stops the
agent from repeating a fix that already failed.

The FULL schema (including the loop/HITL fields) is defined now, even though Piece 1
only uses some of it: changing the schema later invalidates saved checkpoints
(Phase 5), so it's cheaper to get it right up front.
"""
import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict):
    # --- inputs (produced by the one-time comprehension step) ---
    goal: str
    api_schema: dict              # the validated ApiSchema, as a dict

    # --- codegen + execution ---
    current_code: str             # the latest script attempt
    execution_result: dict        # {exit_code, stdout, stderr} from the sandbox

    # --- the retry loop (used from Piece 2 on) ---
    attempt_number: int
    max_retries: int
    error_history: Annotated[list, operator.add]   # APPENDS — see note above

    # --- outcome ---
    status: str                   # running | success | failed
    fetched_data: Optional[list]  # parsed from stdout on success