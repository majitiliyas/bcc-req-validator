"""Настройки агента из окружения / .env (pydantic-settings). Ключи никогда не логируются."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr | None = Field(default=None, description="Ключ Google AI Studio")
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    primary_model: str = Field(default="gemini-3.8-flash", description="Основная модель (Gemini)")

    # Ретраи делает только наш роутер (backoff + jitter + каскад провайдеров);
    # встроенные ретраи SDK отключены, чтобы не было скрытых повторов и двойного расхода квоты.
    sdk_max_retries: int = Field(default=0, ge=0, le=0)
    llm_timeout_s: float = Field(default=90.0, gt=0)

    cache_dir: Path = Path(".cache")
    citation_threshold: float = Field(default=85.0, ge=50, le=100)
    enable_ocr: bool = True

    def has_any_llm_key(self) -> bool:
        return any(k is not None and k.get_secret_value().strip() and "your_" not in k.get_secret_value()
                   for k in (self.gemini_api_key, self.groq_api_key, self.openrouter_api_key))


def load_settings() -> Settings:
    return Settings()
