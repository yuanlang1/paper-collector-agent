from langchain_openai import ChatOpenAI
from openai.resources.chat.chat import Chat
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from typing import TypeVar
from app.config import settings
from app.llm.structured_output import ValidatedJsonInvoker

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

def create_chat_model(
    *, 
    temperature: float = 0.7,
    streaming: bool = False,
) -> ChatOpenAI:
    if not settings.OPENAI_MODEL:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 或环境变量中配置")
    
    kwargs = {
        "model": settings.OPENAI_MODEL,
        "temperature": temperature,
        "api_key": settings.OPENAI_API_KEY,
        "streaming": streaming,
    }

    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    
    
    if settings.OPENAI_BASE_URL and "deepseek" in settings.OPENAI_BASE_URL:
        kwargs["disabled_params"] = {"parallel_tool_calls": None}
        kwargs["extra_body"] = {
            "thinking" : {
                "type" : "disabled"
            }
        }

    return ChatOpenAI(**kwargs)


def create_structured_chat_model(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0,
) -> Runnable:
    return create_chat_model(temperature=temperature).with_structured_output(
        schema,
        method="function_calling",
    )


def create_validated_structured_chat_model(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0,
    max_attempts: int = 2,
) -> ValidatedJsonInvoker[StructuredModelT]:
    return ValidatedJsonInvoker(
        model=create_chat_model(temperature=temperature),
        schema=schema,
        max_attempts=max_attempts,
    )
