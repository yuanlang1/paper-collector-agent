from collections.abc import Sequence
from contextlib import aclosing
from typing import Any

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessageLikeRepresentation
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt import tools_condition
from langgraph.stream.run_stream import AsyncIterator
from langgraph.types import Command

from app.llm.context import BaseAgentContext
from app.llm.loop.node.compact import CompactNode
from app.llm.loop.node.solve import SolveNode
from app.llm.loop.node.tool import build_tool_node
from app.llm.loop.state import LoopState


class AgentLoop:
    def __init__(
        self,
        *,
        context: BaseAgentContext,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None,
        checkpointer=None,
        store=None,
        name: str | None = None,
    ) -> None:
        self.context = context
        self.model = model
        self.tools = list(tools or [])

        builder = StateGraph(state_schema=LoopState, context_schema=type(context))

        solve = SolveNode(
            model=self.model, tools=self.tools, system_prompt=system_prompt
        )
        builder.add_node("solve", solve)

        builder.add_node("tool", build_tool_node(self.tools))

        compact = CompactNode(
            model=self.model, tools=self.tools, system_prompt=system_prompt
        )
        builder.add_node("compact", compact)

        builder.add_edge(START, "solve")
        builder.add_conditional_edges(
            "solve", tools_condition, {"tools": "tool", END: END}
        )

        builder.add_conditional_edges(
            "tool", compact.route, {"compact": "compact", "solve": "solve"}
        )
        builder.add_edge("compact", "solve")

        self.graph = builder.compile(checkpointer=checkpointer, store=store, name=name)

    async def run(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        return await self.graph.ainvoke(
            {"messages": messages}, config=config, context=self.context
        )

    async def stream(
        self,
        messages: Sequence[MessageLikeRepresentation] | Command,
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with aclosing(
            self.graph.astream(
                messages if isinstance(messages, Command) else {"messages": messages},
                config=config,
                context=self.context,
                stream_mode=["messages", "custom"],
                subgraphs=True,
                version="v2",
            )
        ) as stream:
            async for event in stream:
                yield event
