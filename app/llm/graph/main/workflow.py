from langgraph.graph import (
    END,
    START,
    StateGraph,
)


from app.llm.graph.main.condition import (
    after_confirm,
    after_dispatch,
    after_prepare_paper_search,
    after_prepare_task_review,
    after_solve,
    after_tool,
)
from app.llm.graph.main.nodes.confirm import (
    confirm_node,
)
from app.llm.graph.main.nodes.final import (
    final_node,
)
from app.llm.graph.main.nodes.dispatch import dispatch_tool_call_node
from app.llm.graph.main.nodes.paper_search import (
    complete_paper_search_node,
    prepare_paper_search_node,
)
from app.llm.graph.main.nodes.safe_subgraph import SafeSubgraphNode
from app.llm.graph.main.nodes.task_review import (
    complete_task_review_node,
    prepare_task_review_node,
)
from app.llm.graph.main.nodes.tool import (
    tool_node,
)
from app.llm.graph.main.state import (
    MainAgentState,
)


async def _passthrough_memory_node(
    _state: MainAgentState,
) -> dict[str, object]:
    return {}


def build_main_agent_workflow(
    *,
    memory_node=None,
    solve_node,
    paper_search_graph,
    task_review_graph,
    checkpointer=None,
):
    builder = StateGraph(MainAgentState)

    builder.add_node("memory", memory_node or _passthrough_memory_node)
    builder.add_node("solve", solve_node)
    builder.add_node("dispatch", dispatch_tool_call_node)
    builder.add_node("tool", tool_node)
    builder.add_node(
        "prepare_paper_search",
        prepare_paper_search_node,
    )
    builder.add_node(
        "paper_search",
        SafeSubgraphNode(
            subgraph=paper_search_graph,
            handoff_key="paper_search_handoff",
            subagent="paper_search_agent",
            error_code="PAPER_SEARCH_SUBGRAPH_FAILED",
            summary="论文检索工作流执行失败，未完成结果保存。",
        ),
    )
    builder.add_node(
        "complete_paper_search",
        complete_paper_search_node,
    )
    builder.add_node(
        "prepare_task_review",
        prepare_task_review_node,
    )
    builder.add_node(
        "task_review",
        SafeSubgraphNode(
            subgraph=task_review_graph,
            handoff_key="task_review_handoff",
            subagent="task_review_agent",
            error_code="TASK_REVIEW_SUBGRAPH_FAILED",
            summary="文献综述工作流执行失败，未完成结果保存。",
        ),
    )
    builder.add_node(
        "complete_task_review",
        complete_task_review_node,
    )
    builder.add_node("confirm", confirm_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "memory")
    builder.add_edge("memory", "solve")

    builder.add_conditional_edges(
        "solve",
        after_solve,
        {
            "dispatch": "dispatch",
            "final": "final",
        },
    )

    builder.add_conditional_edges(
        "dispatch",
        after_dispatch,
        {
            "solve": "solve",
            "confirm": "confirm",
            "tool": "tool",
            "paper_search_agent": "prepare_paper_search",
            "task_review_agent": "prepare_task_review",
        },
    )

    builder.add_conditional_edges(
        "confirm",
        after_confirm,
        {
            "dispatch": "dispatch",
            "tool": "tool",
            "paper_search_agent": "prepare_paper_search",
            "task_review_agent": "prepare_task_review",
        },
    )

    builder.add_conditional_edges(
        "tool",
        after_tool,
        {
            "dispatch": "dispatch",
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
    builder.add_edge("complete_paper_search", "dispatch")
    builder.add_conditional_edges(
        "prepare_task_review",
        after_prepare_task_review,
        {
            "run": "task_review",
            "complete": "complete_task_review",
        },
    )
    builder.add_edge("task_review", "complete_task_review")
    builder.add_edge("complete_task_review", "dispatch")
    builder.add_edge("final", END)

    return builder.compile(checkpointer = checkpointer)
