from collections.abc import Sequence
from typing import Any
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import MessageLikeRepresentation
from langgraph.graph.state import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.prebuilt import tools_condition
from langgraph.stream.run_stream import AsyncIterator

from app.llm.loop.node.solve import SolveNode
from app.llm.loop.node.tool import build_tool_node

class AgentLoop:
    def __init__(
        self,
        *,
        db: AsyncSession,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None,
        checkpointer = None,
        store = None,
        context_schema: type | None = None,
        name: str | None = None
    ) -> None:
        self.tools = list(tools or [])

        builder = StateGraph(
            state_schema = MessagesState,
            context_schema = context_schema
        )

        builder.add_node(
            "solve",
            SolveNode(
                model = model,
                tools = self.tools,
                system_prompt = system_prompt
            )
        )

        builder.add_node(
            "tool",
            build_tool_node(self.tools)
        )

        builder.add_edge(START, "solve")
        builder.add_conditional_edges(
            "solve",
            tools_condition,
            {
                "tools": "tool",
                END: END
            }
        )

        builder.add_edge("tool", "solve")

        self.graph = builder.compile(
            checkpointer = checkpointer,
            store = store,
            name = name
        )

    async def run(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        config: RunnableConfig | None = None,
        context: Any = None
    ) -> dict[str, Any]:
        return await self.graph.ainvoke(
            {"messages": messages},
            config = config,
            context = context
        )

    async def stream(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        config: RunnableConfig | None = None,
        context: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in self.graph.astream_events(
            {"messages": messages},
            config = config,
            context = context,
            version = "v2",
        ):
            yield event
