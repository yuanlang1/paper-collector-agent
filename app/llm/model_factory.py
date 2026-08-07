from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import settings
from app.llm.structured_output import ValidatedJsonInvoker

StructuredModelT = TypeVar("StructuredModelT", bound = BaseModel)


class ReasoningPreservingChatDeepSeek(ChatDeepSeek):
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(
            input_,
            stop = stop,
            **kwargs,
        )
        messages = self._convert_input(input_).to_messages()

        for source, request_message in zip(messages, payload["messages"]):
            if not isinstance(source, AIMessage):
                continue

            reasoning_content = source.additional_kwargs.get(
                "reasoning_content",
            )
            if isinstance(reasoning_content, str):
                request_message["reasoning_content"] = reasoning_content

        return payload


def _get_llm_provider() -> str:
    provider = getattr(settings, "LLM_PROVIDER", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()

    base_url = (settings.OPENAI_BASE_URL or "").lower()
    return "deepseek" if "deepseek" in base_url else "openai"


def create_chat_model(
    *, 
    temperature: float = 0.7,
    streaming: bool = False,
) -> BaseChatModel:
    if not settings.OPENAI_MODEL:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 或环境变量中配置")
    
    if _get_llm_provider() == "deepseek":
        kwargs = {
            "model": settings.OPENAI_MODEL,
            "api_key": settings.OPENAI_API_KEY,
            "api_base": settings.OPENAI_BASE_URL,
            "streaming": streaming,
            "reasoning_effort": settings.REASONING_EFFORT,
            "extra_body": {
                "thinking": {
                    "type": settings.THINKING_TYPE,
                },
            },
            "disabled_params": {"parallel_tool_calls": None},
        }

        if settings.THINKING_TYPE != "enabled":
            kwargs["temperature"] = temperature

        return ReasoningPreservingChatDeepSeek(**kwargs)

    return ChatOpenAI(
        model = settings.OPENAI_MODEL,
        temperature = temperature,
        api_key = settings.OPENAI_API_KEY,
        base_url = settings.OPENAI_BASE_URL or None,
        streaming = streaming,
    )


def create_structured_chat_model(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0,
) -> Runnable:
    return create_chat_model(temperature = temperature).with_structured_output(
        schema,
        method = "function_calling",
    )


def create_validated_structured_chat_model(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0,
    max_attempts: int = 2,
) -> ValidatedJsonInvoker[StructuredModelT]:
    return ValidatedJsonInvoker(
        model = create_chat_model(temperature = temperature),
        schema = schema,
        max_attempts = max_attempts,
    )
