"""
AgentState — the typed, shared state that flows through the LangGraph state machine.

Nodes receive the whole state and return a PARTIAL dict of what they changed;
LangGraph merges those updates.

Merge rules:
  • plain fields        -> last write wins (overwrite)
  • Annotated[..., add] -> APPEND (concatenate)

`error_history` MUST use the append reducer, or each diagnose step would erase the
record of past failures — and that memory is what stops the agent repeating a fix
that already failed.
"""
import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict):
    # --- inputs (from the one-time comprehension step) ---
    goal: str
    api_schema: dict

    # --- codegen + execution ---
    current_code: str
    execution_result: dict

    # --- the retry loop ---
    attempt_number: int
    max_retries: int
    error_history: Annotated[list, operator.add]

    # --- outcome ---
    status: str                    # running | success | failed | persisted | persist_failed
    fetched_data: Optional[list]

    # --- persistence (Phase 4.2) ---
    rows_upserted: Optional[int]
    rows_for_endpoint: Optional[int]
    persist_error: Optional[str]