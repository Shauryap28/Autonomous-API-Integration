"""
graph — the LangGraph state machine.

    START -> generate_code -> execute -> <route_result>
                                 ^            |
                                 |            +-- success           -> persist_and_verify -> END
                                 |            +-- retries exhausted -> END (failed)
                                 |            +-- retry -> diagnose_and_fix --+
                                 +-----------------------------------------------+

execute            = Docker sandbox: fetches, holds NO DB credentials
persist_and_verify = trusted backend: validates + writes to Postgres

Live resources (vectorstore) and setup context (source/endpoint) are injected via
closures, never stored in state — state is serialized into every checkpoint.
"""
from langgraph.graph import END, START, StateGraph

from backend.agent.state import AgentState
from backend.agent.nodes import (
    generate_code_node,
    execute_node,
    make_diagnose_node,
    make_persist_node,
    route_result,
)


def build_graph(vectorstore, endpoint_section, source):
    builder = StateGraph(AgentState)

    builder.add_node("generate_code", generate_code_node)
    builder.add_node("execute", execute_node)
    builder.add_node("diagnose_and_fix", make_diagnose_node(vectorstore, endpoint_section))
    builder.add_node("persist_and_verify", make_persist_node(source, endpoint_section))

    builder.add_edge(START, "generate_code")
    builder.add_edge("generate_code", "execute")

    builder.add_conditional_edges(
        "execute",
        route_result,
        {
            "success": "persist_and_verify",
            "retry": "diagnose_and_fix",
            "give_up": END,
        },
    )

    builder.add_edge("diagnose_and_fix", "execute")
    builder.add_edge("persist_and_verify", END)

    return builder.compile()