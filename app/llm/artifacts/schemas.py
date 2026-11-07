from pydantic import BaseModel


class ArtifactRef(BaseModel):
    """工具产物的逻辑 URI 引用。"""

    artifact_uri: str
