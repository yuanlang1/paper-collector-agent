import unittest

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.llm.structured_output import (
    StructuredOutputError,
    ValidatedJsonInvoker,
)


class _ExampleOutput(BaseModel):
    count: int = Field(gt=0)


class _StructuredModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class _ChatModel:
    def __init__(self, structured_model):
        self.structured_model = structured_model
        self.schema = None
        self.method = None
        self.include_raw = None

    def with_structured_output(
        self,
        schema,
        *,
        method,
        include_raw,
    ):
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self.structured_model


def _validation_error() -> ValidationError:
    try:
        _ExampleOutput.model_validate({"count": 0})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected validation error")


class ValidatedJsonInvokerTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_returns_first_valid_output(self):
        structured_model = _StructuredModel(
            [
                {
                    "raw": AIMessage(content='{"count": 1}'),
                    "parsed": _ExampleOutput(count=1),
                    "parsing_error": None,
                }
            ]
        )
        chat_model = _ChatModel(structured_model)
        invoker = ValidatedJsonInvoker(
            model=chat_model,
            schema=_ExampleOutput,
        )

        result = await invoker.ainvoke(
            [HumanMessage(content="return a JSON object")]
        )

        self.assertEqual(result.count, 1)
        self.assertEqual(len(structured_model.calls), 1)
        self.assertIs(chat_model.schema, _ExampleOutput)
        self.assertEqual(chat_model.method, "json_mode")
        self.assertTrue(chat_model.include_raw)

    async def test_repairs_invalid_output(self):
        invalid_raw = AIMessage(content='{"count": 0}')
        structured_model = _StructuredModel(
            [
                {
                    "raw": invalid_raw,
                    "parsed": None,
                    "parsing_error": _validation_error(),
                },
                {
                    "raw": AIMessage(content='{"count": 2}'),
                    "parsed": _ExampleOutput(count=2),
                    "parsing_error": None,
                },
            ]
        )
        invoker = ValidatedJsonInvoker(
            model=_ChatModel(structured_model),
            schema=_ExampleOutput,
        )

        result = await invoker.ainvoke(
            [HumanMessage(content="return a JSON object")]
        )

        self.assertEqual(result.count, 2)
        self.assertEqual(len(structured_model.calls), 2)
        repair_messages = structured_model.calls[1]
        self.assertIs(repair_messages[-2], invalid_raw)
        self.assertIn(
            "count",
            repair_messages[-1].content,
        )
        self.assertIn(
            "_ExampleOutput",
            repair_messages[-1].content,
        )

    async def test_raises_after_max_attempts(self):
        error = _validation_error()
        structured_model = _StructuredModel(
            [
                {
                    "raw": AIMessage(content='{"count": 0}'),
                    "parsed": None,
                    "parsing_error": error,
                },
                {
                    "raw": AIMessage(content='{"count": -1}'),
                    "parsed": None,
                    "parsing_error": error,
                },
            ]
        )
        invoker = ValidatedJsonInvoker(
            model=_ChatModel(structured_model),
            schema=_ExampleOutput,
            max_attempts=2,
        )

        with self.assertRaises(StructuredOutputError) as raised:
            await invoker.ainvoke(
                [HumanMessage(content="return a JSON object")]
            )

        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(structured_model.calls), 2)


if __name__ == "__main__":
    unittest.main()
