from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class ModelOptions:
    temperature: float = 0.7
    streaming: bool = True


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


class ChatProvider(Protocol):
    def create_model(
        self,
        config: ProviderConfig,
        options: ModelOptions,
    ) -> BaseChatModel:
        ...
