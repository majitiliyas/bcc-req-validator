"""Настройки агента из окружения / .env (pydantic-settings). Ключи никогда не логируются."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DEEP_MODELS = (
    "gemini/gemini-4-hydrogen,gemini/gemini-3.8-flash,gemini/gemini-3.5-flash-lite,"
    "nvidia/nvidia/nemotron-3-ultra-550b-a55b,nvidia/deepseek-ai/deepseek-v4.1-flash,"
    "openrouter/openrouter/free"
)
DEFAULT_FAST_MODELS = "groq/qwen/qwen3.8-27b,groq/openai/gpt-oss-120b,gemini/gemini-3.5-flash-lite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- ключи провайдеров (все бесплатные тарифы) ---
    gemini_api_key: SecretStr | None = Field(default=None, description="Google AI Studio")
    nvidia_api_key: SecretStr | None = Field(default=None, description="NVIDIA NIM (build.nvidia.com)")
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    # --- маршрутизация: 'провайдер/точный-slug' через запятую, порядок = приоритет ---
    deep_models: str = DEFAULT_DEEP_MODELS
    fast_models: str = DEFAULT_FAST_MODELS
    enable_ollama: bool = True
    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = Field(default="http://localhost:11434",
                                 description="Адрес Ollama; допускается с /v1 и без — нормализуется")
    ollama_timeout_s: float = Field(default=600.0, gt=0, description="Локальная модель на CPU/GPU ноутбука медленнее облака")
    ollama_context_window: int = Field(default=32_768, ge=2_048,
                                       description="Должен совпадать с OLLAMA_CONTEXT_LENGTH / num_ctx сервера")

    # --- надёжность роутера ---
    sdk_max_retries: int = Field(default=0, ge=0, le=0, description="Встроенные ретраи SDK отключены")
    llm_timeout_s: float = Field(default=120.0, gt=0)
    allowed_fails: int = Field(default=2, ge=1)
    cooldown_time: float = Field(default=30.0, ge=1, le=60)
    max_attempts_per_deployment: int = Field(default=3, ge=1)
    backoff_base: float = 2.0
    backoff_cap: float = 30.0
    max_retry_after: float = 60.0
    max_output_tokens: int = 4096

    # --- бюджет прогона (AgentBudgetGuard) ---
    budget_max_steps: int = 80
    budget_max_tokens: int = 600_000
    budget_max_wall_seconds: float = 1200.0
    budget_max_identical_calls: int = 2
    circuit_breaker_threshold: int = 4

    # --- пайплайн аудита ---
    pass1_group: str = "fast"
    pass1_batch_size: int = Field(default=5, ge=1, le=10)
    critic_batch_size: int = Field(default=6, ge=1, le=20)
    llm_concurrency: int = Field(default=3, ge=1, le=10)
    max_validation_retries: int = Field(default=2, ge=0, le=3)
    use_presidio: bool = True

    # --- прочее ---
    cache_dir: Path = Path(".cache")
    use_cache: bool = True
    citation_threshold: float = Field(default=85.0, ge=50, le=100)
    enable_ocr: bool = True

    @property
    def ollama_native_url(self) -> str:
        """Нативный API Ollama (/api/tags…): без завершающего / и /v1."""
        return self.ollama_base_url.strip().rstrip("/").removesuffix("/v1").rstrip("/")

    @property
    def ollama_openai_url(self) -> str:
        """OpenAI-совместимый endpoint Ollama — всегда с /v1."""
        return self.ollama_native_url + "/v1"

    @property
    def cache_db(self) -> Path:
        return self.cache_dir / "agent_checkpoints.db"

    @property
    def trace_dir(self) -> Path:
        return self.cache_dir / "traces"

    def configured_keys(self) -> dict[str, bool]:
        def ok(secret: SecretStr | None) -> bool:
            if secret is None:
                return False
            v = secret.get_secret_value().strip()
            return bool(v) and not v.startswith(("your_", "optional_"))

        return {"gemini": ok(self.gemini_api_key), "nvidia": ok(self.nvidia_api_key),
                "groq": ok(self.groq_api_key), "openrouter": ok(self.openrouter_api_key)}

    def has_any_llm_key(self) -> bool:
        return any(self.configured_keys().values())


def load_settings() -> Settings:
    return Settings()
