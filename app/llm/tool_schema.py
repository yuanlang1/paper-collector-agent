from typing import Any

from pydantic import BaseModel


def build_model_tool_schema(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_model.model_json_schema(),
        },
    }
