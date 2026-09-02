from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.response import ServiceResponse
from app.models.llm_profile import LlmProfile
from app.services.llm_profile_service import (
    DEFAULT_BASE_URLS,
    SUPPORTED_PROVIDERS,
    encrypt_api_key,
    get_profile_or_raise,
)
from app.services.setting_service import (
    get_source_limits,
    save_source_limits,
    source_limit_maxima,
    source_page_sizes,
)


router = APIRouter(prefix="/api/settings", tags=["settings"])


class LlmProfileInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    provider: str
    base_url: str | None = Field(default=None, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1)
    enabled: bool = True
    is_default: bool = False

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}")
        return value


class LlmProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    provider: str | None = None
    base_url: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    api_key: str | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    is_default: bool | None = None

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip().lower()
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}")
        return value


class LlmProfileView(BaseModel):
    id: int
    name: str
    provider: str
    base_url: str
    model: str
    key_set: bool
    key_last4: str
    enabled: bool
    is_default: bool
    version: int
    created_at: datetime
    updated_at: datetime


class LlmProfileListResponse(BaseModel):
    items: list[LlmProfileView]


class LlmProfileConnectionTestResponse(BaseModel):
    ok: bool
    model_count: int | None = None
    message: str | None = None


class DeleteLlmProfileResponse(BaseModel):
    id: int
    deleted: bool


class SourceLimitsInput(BaseModel):
    arxiv: int = Field(..., ge=1)
    dblp: int = Field(..., ge=1)
    google_scholar: int = Field(..., ge=1)


class SourceLimitsView(BaseModel):
    limits: SourceLimitsInput
    page_sizes: dict[str, int]
    maximum_limits: dict[str, int]


def _view(profile: LlmProfile) -> LlmProfileView:
    return LlmProfileView(
        id=profile.id,
        name=profile.name,
        provider=profile.provider,
        base_url=profile.base_url,
        model=profile.model,
        key_set=bool(profile.api_key_ciphertext),
        key_last4="",
        enabled=profile.enabled,
        is_default=profile.is_default,
        version=profile.version,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _set_default(db: Session, profile: LlmProfile) -> None:
    db.query(LlmProfile).filter(LlmProfile.id != profile.id).update(
        {LlmProfile.is_default: False}, synchronize_session=False
    )
    profile.is_default = True


def _source_limits_view(limits: dict[str, int]) -> SourceLimitsView:
    maxima = source_limit_maxima()
    for name, limit in limits.items():
        if limit > maxima[name]:
            raise HTTPException(
                status_code=400,
                detail=f"{name} cannot exceed {maxima[name]}",
            )
    return SourceLimitsView(
        limits=SourceLimitsInput.model_validate(limits),
        page_sizes=source_page_sizes(),
        maximum_limits=maxima,
    )


@router.get(
    "/source-limits",
    response_model=ServiceResponse[SourceLimitsView],
)
def get_paper_search_source_limits(
    db: Session = Depends(get_db),
) -> ServiceResponse[SourceLimitsView]:
    return ServiceResponse[SourceLimitsView].build_success_response(
        data=_source_limits_view(get_source_limits(db)),
        message="OK",
    )


@router.put(
    "/source-limits",
    response_model=ServiceResponse[SourceLimitsView],
)
def update_paper_search_source_limits(
    payload: SourceLimitsInput,
    db: Session = Depends(get_db),
) -> ServiceResponse[SourceLimitsView]:
    try:
        limits = save_source_limits(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ServiceResponse[SourceLimitsView].build_success_response(
        data=_source_limits_view(limits),
        message="OK",
    )


@router.get("/llm-profiles", response_model=ServiceResponse[LlmProfileListResponse])
def list_llm_profiles(db: Session = Depends(get_db)) -> ServiceResponse[LlmProfileListResponse]:
    return ServiceResponse[LlmProfileListResponse].build_success_response(
        data=LlmProfileListResponse(
            items=[_view(profile) for profile in db.query(LlmProfile).order_by(LlmProfile.id).all()]
        ),
        message="OK",
    )


@router.post("/llm-profiles", response_model=ServiceResponse[LlmProfileView], status_code=status.HTTP_201_CREATED)
def create_llm_profile(payload: LlmProfileInput, db: Session = Depends(get_db)) -> ServiceResponse[LlmProfileView]:
    if payload.is_default and not payload.enabled:
        raise HTTPException(status_code=400, detail="Disabled profile cannot be the default")
    if db.query(LlmProfile).filter(LlmProfile.name == payload.name).first():
        raise HTTPException(status_code=409, detail="LLM profile name already exists")
    profile = LlmProfile(
        name=payload.name,
        provider=payload.provider,
        base_url=(payload.base_url or DEFAULT_BASE_URLS[payload.provider]).rstrip("/"),
        model=payload.model,
        api_key_ciphertext=encrypt_api_key(payload.api_key),
        enabled=payload.enabled,
        is_default=payload.is_default,
    )
    db.add(profile)
    db.flush()
    if profile.is_default or db.query(LlmProfile).filter(LlmProfile.is_default.is_(True)).count() == 0:
        _set_default(db, profile)
    db.commit()
    db.refresh(profile)
    return ServiceResponse[LlmProfileView].build_success_response(data=_view(profile), message="OK")


@router.patch("/llm-profiles/{profile_id}", response_model=ServiceResponse[LlmProfileView])
def update_llm_profile(profile_id: int, payload: LlmProfileUpdate, db: Session = Depends(get_db)) -> ServiceResponse[LlmProfileView]:
    try:
        profile = get_profile_or_raise(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    updates = payload.model_dump(exclude_unset=True)
    if "name" in updates and db.query(LlmProfile).filter(LlmProfile.name == updates["name"], LlmProfile.id != profile_id).first():
        raise HTTPException(status_code=409, detail="LLM profile name already exists")
    previous_provider = profile.provider
    provider = updates.get("provider", previous_provider)
    for field in ("name", "provider", "model", "enabled"):
        if field in updates:
            setattr(profile, field, updates[field])
    if "base_url" in updates:
        profile.base_url = (updates["base_url"] or DEFAULT_BASE_URLS[provider]).rstrip("/")
    elif "provider" in updates and profile.base_url == DEFAULT_BASE_URLS.get(previous_provider):
        profile.base_url = DEFAULT_BASE_URLS[provider]
    if "api_key" in updates:
        profile.api_key_ciphertext = encrypt_api_key(updates["api_key"])
    if updates.get("is_default"):
        _set_default(db, profile)
    if updates.get("enabled") is False and profile.is_default:
        raise HTTPException(status_code=400, detail="Set another default profile before disabling this profile")
    profile.version += 1
    db.commit()
    db.refresh(profile)
    return ServiceResponse[LlmProfileView].build_success_response(data=_view(profile), message="OK")


@router.post("/llm-profiles/{profile_id}/set-default", response_model=ServiceResponse[LlmProfileView])
def set_default_llm_profile(profile_id: int, db: Session = Depends(get_db)) -> ServiceResponse[LlmProfileView]:
    try:
        profile = get_profile_or_raise(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not profile.enabled:
        raise HTTPException(status_code=400, detail="Disabled profile cannot be the default")
    _set_default(db, profile)
    profile.version += 1
    db.commit()
    db.refresh(profile)
    return ServiceResponse[LlmProfileView].build_success_response(data=_view(profile), message="OK")


@router.delete("/llm-profiles/{profile_id}", response_model=ServiceResponse[DeleteLlmProfileResponse])
def delete_llm_profile(profile_id: int, db: Session = Depends(get_db)) -> ServiceResponse[DeleteLlmProfileResponse]:
    try:
        profile = get_profile_or_raise(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if profile.is_default:
        raise HTTPException(status_code=400, detail="Set another default profile before deleting this profile")
    db.delete(profile)
    db.commit()
    return ServiceResponse[DeleteLlmProfileResponse].build_success_response(
        data=DeleteLlmProfileResponse(id=profile_id, deleted=True),
        message="OK",
    )


@router.post(
    "/llm-profiles/{profile_id}/test",
    response_model=ServiceResponse[LlmProfileConnectionTestResponse],
)
async def test_llm_profile(
    profile_id: int,
    db: Session = Depends(get_db),
) -> ServiceResponse[LlmProfileConnectionTestResponse]:
    try:
        profile = get_profile_or_raise(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from app.services.llm_profile_service import decrypt_api_key

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{profile.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {decrypt_api_key(profile.api_key_ciphertext)}"},
            )
        response.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        result = LlmProfileConnectionTestResponse(ok=False, message=str(exc))
    else:
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        result = LlmProfileConnectionTestResponse(ok=True, model_count=len(models))
    return ServiceResponse[LlmProfileConnectionTestResponse].build_success_response(
        data=result,
        message="OK",
    )
