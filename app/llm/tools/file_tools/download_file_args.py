from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DownloadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="待下载文件的 HTTP 或 HTTPS 链接。",
    )

    save_dir: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "相对于 ARTIFACT_BASE_DIR 的保存目录，"
            "例如 downloads/papers/2026。"
        ),
    )

    file_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="保存后的文件名，例如 attention-is-all-you-need.pdf。",
    )

    expected_type: Literal["pdf", "any"] = Field(
        "pdf",
        description="pdf 会校验 PDF 文件头；any 不限制文件类型。",
    )

    overwrite: bool = Field(
        False,
        description="是否覆盖同名文件，默认 false。",
    )

    @field_validator("save_dir")
    @classmethod
    def validate_save_dir(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)

        if (
            not normalized
            or path.is_absolute()
            or normalized == "."
            or any(part in {".", ".."} for part in path.parts)
            or ":" in normalized
        ):
            raise ValueError("save_dir 必须是安全的相对目录。")

        return normalized

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        name = value.strip()

        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
            or ":" in name
        ):
            raise ValueError("file_name 只能是文件名，不能包含路径。")

        return name