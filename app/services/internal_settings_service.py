from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from app.models.llm_profile import LlmProfile
from app.services.llm_profile_service import (
    DEFAULT_BASE_URLS,
    SUPPORTED_PROVIDERS,
    decrypt_api_key,
    encrypt_api_key,
    get_profile_or_raise,
)
from app.services.setting_service import (
    get_source_limits,
    save_source_limits,
    source_limit_maxima,
    source_page_sizes,
)


class SettingsConflictError(ValueError):
    pass


class SettingsValidationError(ValueError):
    pass


@dataclass(frozen=True)
class LlmProfileData:
    id: int
    name: str
    provider: str
    base_url: str
    model: str
    key_set: bool
    key_last4: str
    enabled: bool
    is_default: bool
    is_small_model: bool
    version: int
    created_at: datetime
    updated_at: datetime


def _view(profile: LlmProfile) -> LlmProfileData:
    return LlmProfileData(
        id=profile.id,
        name=profile.name,
        provider=profile.provider,
        base_url=profile.base_url,
        model=profile.model,
        key_set=bool(profile.api_key_ciphertext),
        key_last4="",
        enabled=profile.enabled,
        is_default=profile.is_default,
        is_small_model=profile.is_small_model,
        version=profile.version,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _validate_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise SettingsValidationError(
            f"provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    return normalized


def _set_default(db: Session, profile: LlmProfile) -> None:
    db.query(LlmProfile).filter(LlmProfile.id != profile.id).update(
        {LlmProfile.is_default: False}, synchronize_session=False
    )
    profile.is_default = True


def _set_small_model(db: Session, profile: LlmProfile) -> None:
    db.query(LlmProfile).filter(LlmProfile.id != profile.id).update(
        {LlmProfile.is_small_model: False}, synchronize_session=False
    )
    profile.is_small_model = True


def get_limits(db: Session) -> dict[str, dict[str, int]]:
    return {
        "limits": get_source_limits(db),
        "page_sizes": source_page_sizes(),
        "maximum_limits": source_limit_maxima(),
    }


def update_limits(db: Session, limits: dict[str, int]) -> dict[str, dict[str, int]]:
    try:
        save_source_limits(db, limits)
    except ValueError as exc:
        raise SettingsValidationError(str(exc)) from exc
    return get_limits(db)


def list_profiles(db: Session) -> list[LlmProfileData]:
    return [_view(profile) for profile in db.query(LlmProfile).order_by(LlmProfile.id).all()]


def create_profile(
    db: Session,
    *,
    name: str,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    enabled: bool,
    is_default: bool,
    is_small_model: bool,
) -> LlmProfileData:
    if not name.strip() or not model.strip() or not api_key:
        raise SettingsValidationError("name, model and api_key are required")
    provider = _validate_provider(provider)
    if is_default and not enabled:
        raise SettingsValidationError("Disabled profile cannot be the default")
    if is_small_model and not enabled:
        raise SettingsValidationError("Disabled profile cannot be the memory small model")
    if db.query(LlmProfile).filter(LlmProfile.name == name).first():
        raise SettingsConflictError("LLM profile name already exists")
    profile = LlmProfile(
        name=name,
        provider=provider,
        base_url=(base_url or DEFAULT_BASE_URLS[provider]).rstrip("/"),
        model=model,
        api_key_ciphertext=encrypt_api_key(api_key),
        enabled=enabled,
        is_default=is_default,
        is_small_model=is_small_model,
    )
    db.add(profile)
    db.flush()
    if profile.is_default or (
        profile.enabled
        and not db.query(LlmProfile).filter(LlmProfile.is_default.is_(True)).count()
    ):
        _set_default(db, profile)
    if profile.is_small_model:
        _set_small_model(db, profile)
    db.commit()
    db.refresh(profile)
    return _view(profile)


def update_profile(
    db: Session,
    profile_id: int,
    updates: dict[str, object],
) -> LlmProfileData:
    try:
        profile = get_profile_or_raise(db, profile_id)
    except LookupError:
        raise
    if "provider" in updates:
        updates["provider"] = _validate_provider(str(updates["provider"]))
    target_enabled = bool(updates.get("enabled", profile.enabled))
    target_is_small = bool(updates.get("is_small_model", profile.is_small_model))
    if target_is_small and not target_enabled:
        raise SettingsValidationError("Disabled profile cannot be the memory small model")
    if "name" in updates:
        name = str(updates["name"])
        if not name.strip():
            raise SettingsValidationError("name is required")
        if db.query(LlmProfile).filter(LlmProfile.name == name, LlmProfile.id != profile_id).first():
            raise SettingsConflictError("LLM profile name already exists")
    if "model" in updates and not str(updates["model"]).strip():
        raise SettingsValidationError("model is required")
    provider_before = profile.provider
    provider = str(updates.get("provider", provider_before))
    for field in ("name", "provider", "model", "enabled", "is_small_model"):
        if field in updates:
            setattr(profile, field, updates[field])
    if "base_url" in updates:
        profile.base_url = (str(updates["base_url"]) or DEFAULT_BASE_URLS[provider]).rstrip("/")
    elif "provider" in updates and profile.base_url == DEFAULT_BASE_URLS.get(provider_before):
        profile.base_url = DEFAULT_BASE_URLS[provider]
    if "api_key" in updates:
        key = str(updates["api_key"])
        if not key:
            raise SettingsValidationError("api_key is required")
        profile.api_key_ciphertext = encrypt_api_key(key)
    if updates.get("is_default"):
        _set_default(db, profile)
    if updates.get("is_small_model"):
        _set_small_model(db, profile)
    if updates.get("enabled") is False and profile.is_default:
        raise SettingsValidationError("Set another default profile before disabling this profile")
    profile.version += 1
    db.commit()
    db.refresh(profile)
    return _view(profile)


def set_default_profile(db: Session, profile_id: int) -> LlmProfileData:
    profile = get_profile_or_raise(db, profile_id)
    if not profile.enabled:
        raise SettingsValidationError("Disabled profile cannot be the default")
    _set_default(db, profile)
    profile.version += 1
    db.commit()
    db.refresh(profile)
    return _view(profile)


def delete_profile(db: Session, profile_id: int) -> None:
    profile = get_profile_or_raise(db, profile_id)
    if profile.is_default:
        raise SettingsValidationError("Set another default profile before deleting this profile")
    if profile.is_small_model:
        raise SettingsValidationError("Set another memory small model before deleting this profile")
    db.delete(profile)
    db.commit()


async def test_profile(db: Session, profile_id: int) -> tuple[bool, int | None, str | None]:
    profile = get_profile_or_raise(db, profile_id)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{profile.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {decrypt_api_key(profile.api_key_ciphertext)}"},
            )
        response.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        return False, None, str(exc)
    data = response.json()
    models = data.get("data", []) if isinstance(data, dict) else []
    return True, len(models), None
