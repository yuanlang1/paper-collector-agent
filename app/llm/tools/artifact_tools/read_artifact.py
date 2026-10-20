from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.llm.artifacts.access import (
    ArtifactAccessError,
    ArtifactAccessService,
    MAX_CONTEXT_LINES,
    MAX_JSON_ITEMS,
    MAX_SEARCH_MATCHES,
    MAX_TEXT_LINES,
)
from langchain_core.tools import BaseTool, ToolException, tool


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
    *,
    artifact_access: ArtifactAccessService,
) -> dict[str, Any]:
    args = ReadArtifactArgs.model_validate(params)
    data = await artifact_access.read(
        artifact_uri=args.artifact_uri,
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
    return {"ok": True, "data": data, "artifact_refs": [args.artifact_uri]}


def build_read_artifact_tool(
    *,
    artifact_access: ArtifactAccessService,
) -> BaseTool:
    @tool(
        "read_artifact",
        description=(
            "按需读取已有的 artifact:// JSON 文件。"
            "当需要查看先前工作流产物的摘要、指定行、JSON 字段或关键词匹配时，调用此工具。"
            "输入必须使用先前工作流实际返回的 artifact_uri，并按需要选择读取模式及范围；不得编造 URI。"
        ),
        args_schema=ReadArtifactArgs,
    )
    async def read_artifact_tool(
        *,
        artifact_uri: str,
        mode: Literal['summary', 'text', 'json', 'search'] = 'summary',
        start_line: int = 1,
        max_lines: int = 100,
        json_pointer: str | None = None,
        offset: int = 0,
        limit: int = 20,
        query: str | None = None,
        context_lines: int = 0,
        max_matches: int = 10,
    ) -> dict[str, Any]:
        """按需读取可用的 artifact:// JSON 文件。仅使用真实工作流结果中的 URI；可查看摘要、行范围、JSON Pointer 或关键词匹配。"""
        params = {
            "artifact_uri": artifact_uri,
            "mode": mode,
            "start_line": start_line,
            "max_lines": max_lines,
            "json_pointer": json_pointer,
            "offset": offset,
            "limit": limit,
            "query": query,
            "context_lines": context_lines,
            "max_matches": max_matches,
        }
        try:
            return await read_artifact_handler(params, artifact_access=artifact_access)
        except ArtifactAccessError as exc:
            raise ToolException(str(exc)) from exc

    return read_artifact_tool
