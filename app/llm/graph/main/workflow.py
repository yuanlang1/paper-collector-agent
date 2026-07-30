from langgraph.graph import (
    END,
    START,
    StateGraph,
)


from app.llm.graph.main.condition import (
    after_confirm,
    after_prepare_paper_search,
    after_solve,
    after_tool,
)
from app.llm.graph.main.nodes.confirm import (
    confirm_node,
)
from app.llm.graph.main.nodes.final import (
    final_node,
)
from app.llm.graph.main.nodes.paper_search import (
    complete_paper_search_node,
    prepare_paper_search_node,
)
from app.llm.graph.main.nodes.tool import (
    tool_node,
)
from app.llm.graph.main.state import (
    MainAgentState,
)


def build_main_agent_workflow(
    *,
    solve_node,
    paper_search_graph,
    checkpointer=None,
):
    builder = StateGraph(MainAgentState)

    builder.add_node("solve", solve_node)
    builder.add_node("tool", tool_node)
    builder.add_node(
        "prepare_paper_search",
        prepare_paper_search_node,
    )
    builder.add_node("paper_search", paper_search_graph)
    builder.add_node(
        "complete_paper_search",
        complete_paper_search_node,
    )
    builder.add_node("confirm", confirm_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "solve")

    builder.add_conditional_edges(
        "solve",
        after_solve,
        {
            "solve": "solve",
            "confirm": "confirm",
            "tool": "tool",
            "paper_search": "prepare_paper_search",
            "final": "final",
        },
    )

    builder.add_conditional_edges(
        "confirm",
        after_confirm,
        {
            "solve": "solve",
            "tool": "tool",
            "paper_search": "prepare_paper_search",
            "final": "final",
        },
    )

    builder.add_conditional_edges(
        "tool",
        after_tool,
        {
            "solve": "solve",
            "final": "final",
        },
    )
    builder.add_conditional_edges(
        "prepare_paper_search",
        after_prepare_paper_search,
        {
            "run": "paper_search",
            "complete": "complete_paper_search",
        },
    )
    builder.add_edge("paper_search", "complete_paper_search")
    builder.add_edge("complete_paper_search", "solve")
    builder.add_edge("final", END)

    return builder.compile(checkpointer = checkpointer)
