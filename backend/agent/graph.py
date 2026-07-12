"""
graph — the LangGraph state machine (Piece 2: now CYCLIC).

    START -> generate_code -> execute -> <route_result>
                                 ^            |
                                 |            +-- success           -> END
                                 |            +-- retries exhausted -> END (failed)
                                 |            +-- retry -> diagnose_and_fix --+
                                 +---------------------------------------------+

The cycle (execute -> diagnose_and_fix -> execute) is the self-healing loop. It is
impossible with a linear chain — this is precisely why LangGraph is here.

The vectorstore is injected via a closure (see make_diagnose_node), never stored in
state: state is serialized into every checkpoint, so it holds DATA, not connections.
"""
from langgraph.graph import END, START, StateGraph

from backend.agent.state import AgentState
from backend.agent.nodes import (
    generate_code_node,
    execute_node,
    make_diagnose_node,
    route_result,
)


def build_graph(vectorstore, endpoint_section):
    builder = StateGraph(AgentState)

    builder.add_node("generate_code", generate_code_node)
    builder.add_node("execute", execute_node)
    builder.add_node("diagnose_and_fix", make_diagnose_node(vectorstore, endpoint_section))

    builder.add_edge(START, "generate_code")
    builder.add_edge("generate_code", "execute")

    # The conditional edge: the agent's decision point.
    builder.add_conditional_edges(
        "execute",
        route_result,
        {
            "success": END,
            "retry": "diagnose_and_fix",
            "give_up": END,
        },
    )

    # ...and the edge that closes the loop.
    builder.add_edge("diagnose_and_fix", "execute")

    return builder.compile()