from functools import partial

from langgraph.graph import (
    END,
    START,
    StateGraph,
)


from app.llm.graph.main.condition import (
    after_confirm,
    after_dispatch,
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
from app.llm.graph.main.nodes.safe_subgraph import SubAgentNode
from app.llm.graph.main.nodes.tool import (
    tool_node,
)
from app.llm.graph.main.state import (
    MainAgentState,
)
from app.llm.subagents.registry import SubAgentRegistry
from app.llm.tools.registry import ToolRegistry, build_tool_registry


async def _passthrough_memory_node(
    _state: MainAgentState,
) -> dict[str, object]:
    return {}


def build_main_agent_workflow(
    *,
    memory_node=None,
    solve_node,
    subagent_registry: SubAgentRegistry,
    tool_registry: ToolRegistry | None = None,
    checkpointer=None,
):
    registry = tool_registry or build_tool_registry()
    builder = StateGraph(MainAgentState)

    builder.add_node("memory", memory_node or _passthrough_memory_node)
    builder.add_node("solve", solve_node)
    builder.add_node("dispatch", dispatch_tool_call_node)
    builder.add_node("tool", partial(tool_node, tool_registry=registry))
    builder.add_node(
        "subagent",
        SubAgentNode(subagent_registry=subagent_registry),
    )
    builder.add_node(
        "confirm",
        partial(confirm_node, subagent_registry=subagent_registry),
    )
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
            "subagent": "subagent",
        },
    )

    builder.add_conditional_edges(
        "confirm",
        after_confirm,
        {
            "dispatch": "dispatch",
            "tool": "tool",
            "subagent": "subagent",
        },
    )

    builder.add_conditional_edges(
        "tool",
        after_tool,
        {
            "dispatch": "dispatch",
        },
    )
    builder.add_edge("subagent", "dispatch")
    builder.add_edge("final", END)

    return builder.compile(checkpointer = checkpointer)
