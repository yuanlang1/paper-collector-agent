import asyncio
from typing import Any
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.errors import GraphRecursionError
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.llm.context import BaseAgentContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.web_tools.search_web import search_web_tool

SUBAGENT_RECURSION_LIMIT = 25


class SubagentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    description: str = Field(
        min_length=1, description="用于展示执行状态的简短子任务说明。"
    )
    prompt: str = Field(
        min_length=1, description="子任务的完整指令，包含必要背景、约束和预期结果。"
    )


class SubagentToolArgs(SubagentArgs):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, arbitrary_types_allowed=True
    )

    runtime: ToolRuntime[Any] = Field(exclude=True)


def build_subagent_tools() -> list[BaseTool]:
    return [search_web_tool]


def build_subagent_tool(*, model: BaseChatModel) -> BaseTool:
    @tool(
        "subagent",
        description=(
            "将独立子任务委派给拥有独立上下文的子代理。"
            "适用于资料检索、信息核对和结果整理。"
            "输入 description 为简短任务说明，prompt 为完整且自包含的任务指令。"
            "子代理可以使用网络搜索，完成后返回最终结果。"
        ),
        args_schema=SubagentToolArgs,
    )
    async def subagent(description: str, prompt: str, runtime: ToolRuntime) -> str:
        response = interrupt(
            {
                "action_requests": [
                    {
                        "name": "subagent",
                        "args": {"description": description, "prompt": prompt},
                        "description": f"是否执行当前子代理\ndescription: {description},\nprompt: {prompt}",
                    }
                ],
                "review_configs": [
                    {
                        "action_name": "subagent",
                        "allowed_decisions": ["approve", "reject", "edit"],
                    }
                ],
            }
        )

        decision = response.get("decisions")[0]

        type = decision.get("type")
        if type == "edit":
            edited_action = decision.get("edited_action")
            args = edited_action.get("args")
            description = args["description"]
            prompt = args["prompt"]

        elif type == "reject":
            raise ToolException("拒绝执行")

        parent = runtime.context
        context = BaseAgentContext(
            thread_id=f"thread_{uuid4().hex}",
            run_id=f"child_{uuid4().hex}",
            user_id=parent.user_id,
            auto_threshold=parent.auto_threshold,
            context_window_tokens=parent.context_window_tokens,
            keep_last_n_turns=parent.keep_last_n_turns,
        )
        config = patch_config(
            runtime.config,
            configurable={"thread_id": context.thread_id},
            recursion_limit=SUBAGENT_RECURSION_LIMIT,
            run_name="subagent",
        )
        config.pop("run_id", None)
        config["metadata"] = {
            **config.get("metadata", {}),
            "agent_scope": "subagent",
            "app_run_id": context.run_id,
            "parent_run_id": parent.run_id,
            "parent_thread_id": parent.thread_id,
        }
        agent = AgentLoop(
            context=context,
            model=model,
            tools=build_subagent_tools(),
            system_prompt=prompt,
            checkpointer=False,
            name="subagent",
        )
        try:
            output = await asyncio.wait_for(
                agent.run([HumanMessage(content=prompt)], config=config),
                timeout=settings.SUBAGENT_DISPATCH_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise ToolException("子代理执行超时，请缩小任务范围后重试。") from exc
        except GraphRecursionError as exc:
            raise ToolException(
                "子代理达到执行步数上限，请缩小任务范围后重试。"
            ) from exc

        response = output["messages"][-1]
        if (
            not isinstance(response, AIMessage)
            or response.tool_calls
            or not response.text.strip()
        ):
            raise ToolException("子代理未生成最终回答。")
        return response.text.strip()

    return subagent
