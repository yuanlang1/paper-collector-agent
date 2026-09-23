from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from app.config import settings
from app.llm.provider.base import ModelOptions, ProviderConfig


class ReasoningPreservingChatDeepSeek(ChatDeepSeek):
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()

        for source, request_message in zip(messages, payload["messages"]):
            if isinstance(source, AIMessage):
                reasoning_content = source.additional_kwargs.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    request_message["reasoning_content"] = reasoning_content
        return payload


class DeepSeekProvider:
    def create_model(
        self,
        config: ProviderConfig,
        options: ModelOptions,
    ) -> ReasoningPreservingChatDeepSeek:
        thinking_type = "disabled" if options.purpose == "memory" else settings.THINKING_TYPE
        reasoning_effort = "low" if options.purpose == "memory" else settings.REASONING_EFFORT
        kwargs: dict[str, Any] = {
            "model": config.model,
            "api_key": config.api_key,
            "api_base": config.base_url,
            "streaming": options.streaming,
            "reasoning_effort": reasoning_effort,
            "extra_body": {
                "thinking": {"type": thinking_type}
            },
            "disabled_params": {
                "parallel_tool_calls": None
            },
        }
        if thinking_type != "enabled":
            kwargs["temperature"] = options.temperature
        return ReasoningPreservingChatDeepSeek(**kwargs)


class OpenAICompatibleProvider:
    def create_model(
        self,
        config: ProviderConfig,
        options: ModelOptions,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=config.model,
            temperature=options.temperature,
            api_key=config.api_key,
            base_url=config.base_url or None,
            streaming=options.streaming,
        )
