from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.model_factory import (
    create_validated_structured_chat_model,
    use_llm_runtime_config,
)
from app.llm.structured_output import ValidatedJsonInvoker
from app.services.llm_profile_service import LlmRuntimeConfig


logger = logging.getLogger(__name__)


class RetrievalDecision(BaseModel):
    retrieve: bool
    query: str = Field(default="", max_length=200)


GATE_PROMPT = """
    你是论文研究助手的长期记忆检索门控器。

    判断回答当前消息是否需要读取用户的长期记忆，例如：
    研究偏好、关注主题、筛选条件、过去讨论过的决定或会话事件。

    不需要读取的情况包括：
    通用知识问答、数学、独立的代码解释、与用户历史无关的请求。

    如果需要读取：
    - retrieve=true
    - query 给出简短的检索关键词，不超过 200 个字符。

    如果不需要读取：
    - retrieve=false
    - query 为空。
"""


class MemoryRetrievalGate:
    def __init__(
        self,
        invoker: ValidatedJsonInvoker[RetrievalDecision] | None = None,
    ) -> None:
        self.invoker = invoker

    async def decide(
        self,
        *,
        user_message: str,
        llm_config: LlmRuntimeConfig | None,
    ) -> RetrievalDecision:
        message = user_message.strip()
        if not message:
            return RetrievalDecision(retrieve=False)

        try:
            with use_llm_runtime_config(llm_config):
                invoker = self.invoker or create_validated_structured_chat_model(
                    RetrievalDecision,
                    temperature=0,
                )
                result = await invoker.ainvoke(
                    [
                        SystemMessage(content=GATE_PROMPT),
                        HumanMessage(content=message),
                    ]
                )
            return RetrievalDecision(
                retrieve=result.retrieve,
                query=result.query.strip(),
            )
        except Exception:
            logger.exception("Memory retrieval gate failed; retrieving by default")
            return RetrievalDecision(
                retrieve=True,
                query=message[:200],
            )