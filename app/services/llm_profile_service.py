from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from app.config import settings
from app.models.llm_profile import LlmProfile


SUPPORTED_PROVIDERS = {"openai", "deepseek"}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


@dataclass(frozen=True)
class LlmRuntimeConfig:
    profile_id: int | None
    version: int | None
    provider: str
    base_url: str
    model: str
    api_key: str

    def snapshot(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
        }


def _cipher() -> Fernet:
    key = settings.LLM_PROFILE_ENCRYPTION_KEY
    if not key:
        raise ValueError("LLM_PROFILE_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(key.encode())
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_PROFILE_ENCRYPTION_KEY is invalid") from exc


def encrypt_api_key(api_key: str) -> str:
    return _cipher().encrypt(api_key.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    try:
        return _cipher().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Unable to decrypt the LLM profile API key") from exc


def get_profile_or_raise(db: Session, profile_id: int) -> LlmProfile:
    profile = db.get(LlmProfile, profile_id)
    if profile is None:
        raise LookupError("LLM profile was not found")
    return profile


def resolve_runtime_config(
    db: Session,
    profile_id: int | None,
) -> LlmRuntimeConfig | None:
    if db is None:
        if profile_id is None:
            return None
        raise ValueError("A database session is required to resolve an LLM profile")
    if profile_id is None:
        profile = db.query(LlmProfile).filter(
            LlmProfile.is_default.is_(True), LlmProfile.enabled.is_(True)
        ).one_or_none()
        if profile is None:
            return None
    else:
        profile = get_profile_or_raise(db, profile_id)
        if not profile.enabled:
            raise ValueError("LLM profile is disabled")

    return LlmRuntimeConfig(
        profile_id=profile.id,
        version=profile.version,
        provider=profile.provider,
        base_url=profile.base_url,
        model=profile.model,
        api_key=decrypt_api_key(profile.api_key_ciphertext),
    )
