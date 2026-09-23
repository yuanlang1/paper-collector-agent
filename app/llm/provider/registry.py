from __future__ import annotations

from app.config import settings
from app.llm.provider.base import ChatProvider, ProviderConfig
from app.llm.provider.provider import DeepSeekProvider, OpenAICompatibleProvider
from app.services.llm_profile_service import LlmRuntimeConfig


def resolve_provider_config(
    runtime_config: LlmRuntimeConfig | None,
) -> ProviderConfig:
    if runtime_config is not None:
        return ProviderConfig(
            provider=runtime_config.provider,
            model=runtime_config.model,
            api_key=runtime_config.api_key,
            base_url=runtime_config.base_url,
        )
    return ProviderConfig(
        provider=settings.LLM_PROVIDER,
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def get_provider(name: str) -> ChatProvider:
    if name.strip().lower() == "deepseek":
        return DeepSeekProvider()
    return OpenAICompatibleProvider()
