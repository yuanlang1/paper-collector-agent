from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import settings
from app.services.llm_profile_service import LlmRuntimeConfig
from app.llm.structured_output import StructuredOutputT, ValidatedJsonInvoker

StructuredModelT = TypeVar("StructuredModelT", bound = BaseModel)
ModelPurpose = Literal["agent", "memory"]
_runtime_config: ContextVar[LlmRuntimeConfig | None] = ContextVar("llm_runtime_config", default=None)


@contextmanager
def use_llm_runtime_config(config: LlmRuntimeConfig | None):
    token = _runtime_config.set(config)
    try:
        yield
    finally:
        _runtime_config.reset(token)


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
    runtime_config = _runtime_config.get()
    if runtime_config is not None:
        return runtime_config.provider
    provider = getattr(settings, "LLM_PROVIDER", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()

    base_url = (settings.OPENAI_BASE_URL or "").lower()
    return "deepseek" if "deepseek" in base_url else "openai"


def create_chat_model(
    *, 
    temperature: float = 0.7,
    streaming: bool = False,
    purpose: ModelPurpose = "agent",
) -> BaseChatModel:
    runtime_config = _runtime_config.get()
    model_name = runtime_config.model if runtime_config else settings.OPENAI_MODEL
    api_key = runtime_config.api_key if runtime_config else settings.OPENAI_API_KEY
    base_url = runtime_config.base_url if runtime_config else settings.OPENAI_BASE_URL
    if not model_name:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 或环境变量中配置")
    
    if _get_llm_provider() == "deepseek":
        thinking_type = (
            "disabled"
            if purpose == "memory"
            else settings.THINKING_TYPE
        )
        reasoning_effort = (
            "low"
            if purpose == "memory"
            else settings.REASONING_EFFORT
        )
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "api_base": base_url,
            "streaming": streaming,
            "reasoning_effort": reasoning_effort,
            "extra_body": {
                "thinking": {
                    "type": thinking_type,
                },
            },
            "disabled_params": {"parallel_tool_calls": None},
        }

        if thinking_type != "enabled":
            kwargs["temperature"] = temperature

        return ReasoningPreservingChatDeepSeek(**kwargs)

    return ChatOpenAI(
        model = model_name,
        temperature = temperature,
        api_key = api_key,
        base_url = base_url or None,
        streaming = streaming,
    )


def create_structured_chat_model(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0,
    purpose: ModelPurpose = "agent",
) -> Runnable:
    return create_chat_model(
        temperature = temperature,
        purpose = purpose,
    ).with_structured_output(
        schema,
        method = "function_calling",
    )


def create_validated_structured_chat_model(
    schema: type[StructuredOutputT],
    *,
    temperature: float = 0,
    max_attempts: int = 4,
    purpose: ModelPurpose = "agent",
) -> ValidatedJsonInvoker[StructuredOutputT]:
    return ValidatedJsonInvoker(
        model = create_chat_model(
            temperature = temperature,
            purpose = purpose,
        ),
        schema = schema,
        max_attempts = max_attempts,
    )
