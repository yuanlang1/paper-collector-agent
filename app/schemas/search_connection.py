from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.db.models.search_connection import SearchConnectionStatus

ProviderId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
]
DisplayName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]
ApiKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CreateSearchConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: ProviderId
    display_name: DisplayName
    api_key: ApiKey | None = Field(default=None, repr=False)
    settings: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True, strict=True)


class UpdateSearchConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: DisplayName | None = None
    api_key: ApiKey | None = Field(default=None, repr=False)
    settings: dict[str, Any] | None = None
    enabled: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def reject_null_required_fields(self):
        for name in self.model_fields_set - {"api_key"}:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self


class SearchConnectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    provider_id: str
    display_name: str
    has_api_key: bool
    settings: dict[str, Any]
    enabled: bool
    status: SearchConnectionStatus
    creation_time: datetime
    modification_time: datetime
