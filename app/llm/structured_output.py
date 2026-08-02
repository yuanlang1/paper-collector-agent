from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError


StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)


class StructuredOutputError(RuntimeError):
    def __init__(
        self,
        *,
        schema_name: str,
        attempts: int,
        error: BaseException,
    ) -> None:
        self.schema_name = schema_name
        self.attempts = attempts
        self.error = error
        super().__init__(
            f"{schema_name} 输出经过 {attempts} 次尝试后仍未通过校验："
            f"{error}"
        )


def _format_validation_error(
    error: BaseException,
) -> str:
    if not isinstance(error, ValidationError):
        return str(error)

    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(
            str(part)
            for part in item["loc"]
        ) or "<root>"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def build_repair_prompt(
    *,
    schema_name: str,
    error: BaseException,
) -> str:
    return (
        f"上一次输出未通过 {schema_name} 校验。\n\n"
        f"校验错误：\n{_format_validation_error(error)}\n\n"
        "请根据原始要求修正这些错误，仅返回完整、合法的 JSON 对象。"
        "不要输出解释、Markdown 或代码块。"
    )


class ValidatedJsonInvoker(
    Generic[StructuredOutputT],
):
    def __init__(
        self,
        *,
        model: Any,
        schema: type[StructuredOutputT],
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.schema = schema
        self.max_attempts = max_attempts
        self.model = model.with_structured_output(
            schema,
            method="json_mode",
            include_raw=True,
        )

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> StructuredOutputT:
        current_messages = list(messages)

        for attempt in range(1, self.max_attempts + 1):
            result = await self.model.ainvoke(current_messages)
            parsed = result["parsed"]

            if parsed is not None:
                return self.schema.model_validate(parsed)

            error = result["parsing_error"] or ValueError(
                f"{self.schema.__name__} 输出为空"
            )

            if attempt == self.max_attempts:
                raise StructuredOutputError(
                    schema_name=self.schema.__name__,
                    attempts=attempt,
                    error=error,
                ) from error

            current_messages.extend(
                [
                    result["raw"],
                    HumanMessage(
                        content=build_repair_prompt(
                            schema_name=self.schema.__name__,
                            error=error,
                        )
                    ),
                ]
            )

        raise AssertionError("unreachable")
