from collections.abc import Sequence
from typing import Literal

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.llm.context import AgentRunContext, BaseAgentContext
from app.llm.loop.state import LoopState
from app.services import agent_thread_summary_service


class SolveNode:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.model_with_tools = model.bind_tools(tools) if tools else model
        self.tools = list(tools or [])
        self.system_prompt = system_prompt

    async def __call__(
        self,
        state: LoopState,
        config: RunnableConfig,
        runtime: Runtime[BaseAgentContext],
    ) -> Command[Literal["tool", "compact", "__end__"]]:
        context = runtime.context
        history = state["messages"]
        messages = (
            [SystemMessage(content=self.system_prompt)] if self.system_prompt else []
        )
        if isinstance(context, AgentRunContext):
            summary = await agent_thread_summary_service.get_summary(
                user_id=context.user_id, thread_id=context.thread_id
            )
            if summary is not None:
                for index, message in enumerate(history):
                    if message.id == summary.end_message_id:
                        messages.append(SystemMessage(content=summary.content))
                        history = history[index + 1 :]
                        break

        inputs = [*messages, *history]
        response = await self.model_with_tools.ainvoke(inputs, config=config)
        update = {"messages": [response]}

        if isinstance(response, AIMessage) and response.tool_calls:
            return Command(update=update, goto="tool")

        if (
            isinstance(context, AgentRunContext)
            and context.auto_threshold > 0
            and isinstance(response, AIMessage)
            and not response.invalid_tool_calls
        ):
            window = context.context_window_tokens
            if window is None:
                window = (self.model.profile or {}).get("max_input_tokens")
            if window and window > 0:
                tokens = response.usage_metadata["input_tokens"]
                if tokens > window * context.auto_threshold:
                    return Command(update=update, goto="compact")

        return Command(update=update, goto=END)
