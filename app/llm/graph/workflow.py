from typing import Any
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import StateGraph
from app.llm.graph.nodes import AgentGraphNodes
from app.llm.graph.state import AgentState

def after_route(state: AgentState) -> str:    
    if state.route == "agent":
        return "agent"
    
    return "ask"

def after_agent(state: AgentState) -> str:
    if not state.messages:
        return "final"

    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.requires_confirmation:
            return "human_approval"
        return "tool"

    return "final"

def after_human_approval(state: AgentState) -> str:
    if state.human_decision == "approved":
        return "tool"

    return "final"

def build_agent_workflow(
    *,
    router: Any,
    ask_llm: Any,
    agent_llm: Any,
    tool_factory: Any,
    build_system_prompt: Any,
    checkpointer: Any = None,
):
    nodes = AgentGraphNodes(
        router = router,
        ask_llm = ask_llm,
        agent_llm = agent_llm,
        tool_factory = tool_factory,
        build_system_prompt = build_system_prompt,
    )

    workflow = StateGraph(AgentState)

    # nodes: route, ask, agent, tool, final
    workflow.add_node("route", nodes.route_node)
    workflow.add_node("ask", nodes.ask_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("tool", nodes.tool_node)
    workflow.add_node("final", nodes.final_node)
    workflow.add_node("human_approval", nodes.human_approval_node)

    # START -> route
    workflow.add_edge(START, "route")

    # route -> ask / agent/ final
    workflow.add_conditional_edges(
        "route",
        after_route,
        {
            "ask": "ask",
            "agent": "agent",
            "final": "final",
        },
    )

    # agent -> tool / final
    workflow.add_conditional_edges(
        "agent",
        after_agent,
        {
            "human_approval": "human_approval",
            "tool": "tool",
            "final": "final",
        },
    )

    workflow.add_conditional_edges(
        "human_approval",
        after_human_approval,
        {
            "tool": "tool",
            "final": "final",
        },
    )

    # tool -> agent
    workflow.add_edge("tool", "agent")
    # ask -> final
    workflow.add_edge("ask", "final")
    
    # final -> END
    workflow.add_edge("final", END)

    return workflow.compile(checkpointer=checkpointer)