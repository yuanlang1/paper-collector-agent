from collections.abc import Sequence
from typing import Any

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.llm.loop.node.compact import model_messages
from app.llm.loop.state import LoopState


class SolveNode:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model.bind_tools(tools) if tools else model
        self.system_prompt = system_prompt

    async def __call__(self, state: LoopState, config: RunnableConfig) -> Any:
        messages = model_messages(state, self.system_prompt)
        response = await self.model.ainvoke(messages, config=config)

        return {"messages": [response]}
