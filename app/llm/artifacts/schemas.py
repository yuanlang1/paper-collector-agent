from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    """
    工具产物引用。

    State 中只保存 ArtifactRef，不保存完整 papers。
    """

    artifact_id: str = Field(..., description="artifact 唯一 ID")
    artifact_uri: str = Field(..., description="逻辑 URI，例如 artifact://run_xxx/search_arxiv/result.json")
    path: str = Field(..., description="当前后端机器上的真实文件路径")

    kind: str = Field(..., description="artifact 类型，例如 paper_search_result_json")
    source: str | None = Field(None, description="数据来源，例如 ARXIV / DBLP / GOOGLE_SCHOLAR")

    run_id: str = Field(..., description="workflow 运行 ID")
    step_key: str = Field(..., description="TODO step_key")

    count: int = Field(0, description="结果数量")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactWriteResult(BaseModel):
    artifact: ArtifactRef
    payload_omitted: bool = False


class PaperSearchArtifactPayload(BaseModel):
    """
    论文检索结果 JSON 文件的统一结构。
    """

    ok: bool
    source: str
    run_id: str
    step_key: str

    query: str | None = None
    search_query: str | None = None

    total_results: int = 0
    returned_count: int = 0

    papers: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)