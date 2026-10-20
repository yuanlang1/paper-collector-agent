from collections.abc import Sequence
from contextlib import aclosing
from operator import add
from typing import Annotated, Any, TypedDict

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import MessageLikeRepresentation
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt import tools_condition
from langgraph.stream.run_stream import AsyncIterator
from langgraph.types import Command

from app.llm.loop.node.solve import SolveNode
from app.llm.loop.node.tool import build_tool_node


class SummaryItem(TypedDict):
    content: str
    start_id: str
    end_id: str
    created_at: str


class LoopState(MessagesState):
    summaries: Annotated[list[SummaryItem], add]


class AgentLoop:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None,
        checkpointer=None,
        store=None,
        context_schema: type | None = None,
        name: str | None = None,
    ) -> None:
        self.tools = list(tools or [])

        builder = StateGraph(state_schema=MessagesState, context_schema=context_schema)

        builder.add_node(
            "solve",
            SolveNode(model=model, tools=self.tools, system_prompt=system_prompt),
        )

        builder.add_node("tool", build_tool_node(self.tools))

        builder.add_edge(START, "solve")
        builder.add_conditional_edges(
            "solve", tools_condition, {"tools": "tool", END: END}
        )

        builder.add_edge("tool", "solve")

        self.graph = builder.compile(checkpointer=checkpointer, store=store, name=name)

    async def run(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        config: RunnableConfig | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        return await self.graph.ainvoke(
            {"messages": messages}, config=config, context=context
        )

    async def stream(
        self,
        messages: Sequence[MessageLikeRepresentation] | Command,
        *,
        config: RunnableConfig | None = None,
        context: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with aclosing(
            self.graph.astream_events(
                messages if isinstance(messages, Command) else {"messages": messages},
                config=config,
                context=context,
                version="v2",
            )
        ) as stream:
            async for event in stream:
                yield event
