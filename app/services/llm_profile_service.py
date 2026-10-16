from dataclasses import dataclass


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
