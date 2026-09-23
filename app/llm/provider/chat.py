from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from app.llm.provider.base import ModelOptions
from app.llm.provider.registry import get_provider, resolve_provider_config
from app.llm.provider.structured_output import StructuredOutputT, ValidatedJsonInvoker
from app.services.llm_profile_service import LlmRuntimeConfig


class ChatClient:
    def __init__(
        self,
         runtime_config: LlmRuntimeConfig | None = None
    ) -> None:
        self._config = resolve_provider_config(runtime_config)
        self._provider = get_provider(self._config.provider)
        self._models: dict[ModelOptions, Any] = {}

    def model(
        self, 
        options: ModelOptions | None = None
    ) -> Any:
        options = options or ModelOptions()
        if options not in self._models:
            self._models[options] = self._provider.create_model(self._config, options)

        return self._models[options]

    @staticmethod
    def messages(
        messages: Sequence[BaseMessage],
        *,
        system_prompt: str | None = None,
    ) -> list[BaseMessage]:
        if not system_prompt:
            return list(messages)
        return [SystemMessage(content=system_prompt), *messages]

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        system_prompt: str | None = None,
        options: ModelOptions | None = None,
    ) -> Any:
        return await self.model(options).ainvoke(self.messages(messages, system_prompt))

    async def astream(
        self,
        messages: Sequence[BaseMessage],
        *,
        system_prompt: str | None = None,
        options: ModelOptions | None = None,
    ) -> AsyncIterator[Any]:
        async for chunk in self.model(options).astream(self.messages(messages, system_prompt)):
            yield chunk

    def bind_tools(
        self, 
        tools: list[dict[str, Any]],
        *, 
        options: ModelOptions | None = None
    ) -> Any:
        return self.model(options).bind_tools(tools)

    def structured(
        self,
        schema: type[StructuredOutputT],
        *,
        max_attempts: int = 4,
        options: ModelOptions | None = None,
    ) -> ValidatedJsonInvoker[StructuredOutputT]:
        return ValidatedJsonInvoker(
            model=self.model(options),
            schema=schema,
            max_attempts=max_attempts
        )
