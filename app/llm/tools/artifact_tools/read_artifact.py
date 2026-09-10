from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.llm.artifacts.access import (
    ArtifactAccessError,
    MAX_CONTEXT_LINES,
    MAX_JSON_ITEMS,
    MAX_SEARCH_MATCHES,
    MAX_TEXT_LINES,
)
from app.llm.tools.registry import Tool, ToolExecutionContext


class ReadArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_uri: str = Field(
        min_length=12,
        description="A real artifact:// URI returned by a prior workflow result.",
    )
    mode: Literal["summary", "text", "json", "search"] = "summary"
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=100, ge=1, le=MAX_TEXT_LINES)
    json_pointer: str | None = Field(
        default=None,
        description="RFC 6901 JSON Pointer; used only in json mode.",
    )
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_JSON_ITEMS)
    query: str | None = Field(default=None, min_length=1, max_length=200)
    context_lines: int = Field(default=0, ge=0, le=MAX_CONTEXT_LINES)
    max_matches: int = Field(default=10, ge=1, le=MAX_SEARCH_MATCHES)

    @field_validator("artifact_uri", "query")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "ReadArtifactArgs":
        if self.mode == "search" and not self.query:
            raise ValueError("query is required for search mode")
        if self.mode == "json" and self.json_pointer and not self.json_pointer.startswith("/"):
            raise ValueError("json_pointer must be empty or start with '/'")
        return self


async def read_artifact_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    args = ReadArtifactArgs.model_validate(params)
    if context.artifact_access is None:
        return {
            "ok": False,
            "error": "ARTIFACT_ACCESS_UNAVAILABLE",
            "message": "artifact 读取服务尚未初始化。",
        }
    try:
        data = await context.artifact_access.read(
            artifact_uri=args.artifact_uri,
            user_id=context.user_id,
            mode=args.mode,
            start_line=args.start_line,
            max_lines=args.max_lines,
            json_pointer=args.json_pointer,
            offset=args.offset,
            limit=args.limit,
            query=args.query,
            context_lines=args.context_lines,
            max_matches=args.max_matches,
        )
    except ArtifactAccessError:
        return {
            "ok": False,
            "error": "ARTIFACT_NOT_FOUND_OR_FORBIDDEN",
            "message": "artifact 不存在或无访问权限。",
        }
    return {"ok": True, "data": data, "artifact_refs": [args.artifact_uri]}


READ_ARTIFACT_TOOL = Tool(
    name="read_artifact",
    description=(
        "按需读取当前用户可访问的 artifact:// JSON 文件。"
        "仅使用真实工作流结果中的 URI；可查看摘要、行范围、JSON Pointer 或关键词匹配。"
    ),
    input_schema=ReadArtifactArgs.model_json_schema(),
    fn=read_artifact_handler,
)
