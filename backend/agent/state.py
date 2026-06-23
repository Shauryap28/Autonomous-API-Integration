"""
AgentState — the typed, shared state object that flows through the LangGraph
state machine in later phases.

Defined early (Phase 1) on purpose: the `api_schema` field is the *output
contract* of the RAG pipeline we build now. Phase 1 ends when we can produce a
validated api_schema; everything else in this state gets filled in later.

TODO (Phase 1): define AgentState as a TypedDict carrying at least the doc
inputs (goal, doc_url) and api_schema. Add codegen / execution / retry / HITL
fields in their respective phases — not before we need them.

Gotcha to remember for later: list fields like `error_history` need an
annotated reducer (Annotated[list, operator.add]) or LangGraph overwrites
instead of appends.
"""
