
from collections.abc import Sequence
from typing import Any
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState


class SolveNode:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None
    ) -> None:
        self.model = model.bind_tools(tools) if tools else model
        self.system_prompt = system_prompt

    async def __call__(
        self, 
        state: MessagesState,
        config: RunnableConfig
    ) -> Any:
        messages = state["messages"]

        if self.system_prompt:
            messages = [
                SystemMessage(content = self.system_prompt),
                *messages
            ]
        
        response = await self.model.ainvoke(messages, config = config)

        return {"messages": [response]}

    