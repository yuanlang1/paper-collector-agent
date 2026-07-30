from pydantic import BaseModel, Field


class ArtifactSaveArgs(BaseModel):
    run_id: str | None = Field(
        None,
        description="可选 workflow 运行 ID，用于生成 artifact 保存路径。",
    )

    step_key: str | None = Field(
        None,
        description="当前 TODO 的 step_key，用于生成 artifact 保存路径。",
    )

    save_json: bool = Field(
        True,
        description="是否把工具结果保存为 JSON artifact。",
    )

    return_papers: bool = Field(
        False,
        description=(
            "是否在工具返回值中直接返回 papers。"
            "大量结果建议 False，只返回 artifact 引用。"
        ),
    )