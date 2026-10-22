import json
from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError


def as_chat_model(invoker) -> BaseChatModel:
    model = Mock(spec=BaseChatModel)

    def with_structured_output(schema, *, method, include_raw):
        async def invoke(messages):
            result = await invoker.ainvoke(messages)
            payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
            raw = AIMessage(content=json.dumps(payload, ensure_ascii=False))
            try:
                parsed = schema.model_validate(payload)
            except ValidationError as error:
                return {"raw": raw, "parsed": None, "parsing_error": error}
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(invoke)

    model.with_structured_output.side_effect = with_structured_output
    return model
