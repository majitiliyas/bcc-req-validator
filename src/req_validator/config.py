"""Настройки агента из окружения / .env (pydantic-settings). Ключи никогда не логируются."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Порядок проверен живым запросом (doctor --ping): первыми идут модели, которые на бесплатном ключе
# стабильно отвечают валидным JSON; модели, дающие 429/503/таймаут/пустой ответ, убраны из начала цепочки.
DEFAULT_DEEP_MODELS = (
    "gemini/gemini-3.6-flash,gemini/gemini-3.5-flash,"
    "openrouter/nvidia/nemotron-3.5-super-vl:free,openrouter/openrouter/free,"
    "nvidia/nvidia/nemotron-3-ultra-550b-a55b,gemini/gemini-3.5-flash-lite"
)
DEFAULT_FAST_MODELS = "gemini/gemini-3.5-flash-lite,groq/openai/gpt-oss-120b,groq/qwen/qwen3.8-27b"
# Критик начинается с другого семейства моделей, чем генераторы (Gemini), — независимое суждение.
DEFAULT_CRITIC_MODELS = (
    "openrouter/nvidia/nemotron-3.5-super-vl:free,openrouter/nvidia/nemotron-3-ultra-550b-a55b:free,"
    "gemini/gemini-3.5-flash,openrouter/openrouter/free,gemini/gemini-3.6-flash"
)


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
    critic_models: str = Field(default=DEFAULT_CRITIC_MODELS, description="Цепочка Критика; пусто — группа deep")
    enable_ollama: bool = True
    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = Field(default="http://localhost:11434",
                                 description="Адрес Ollama; допускается с /v1 и без — нормализуется")
    ollama_timeout_s: float = Field(default=600.0, gt=0, description="Локальная модель на CPU/GPU ноутбука медленнее облака")
    ollama_context_window: int = Field(default=32_768, ge=2_048,
                                       description="Должен совпадать с OLLAMA_CONTEXT_LENGTH / num_ctx сервера")

    # --- надёжность роутера ---
    sdk_max_retries: int = Field(default=0, ge=0, le=0, description="Встроенные ретраи SDK отключены")
    llm_timeout_s: float = Field(default=180.0, gt=0)
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
    full_context_max_bytes: int = Field(default=25_000, ge=0,
                                        description="Документ до этого размера целиком передаётся в Pass 1/2 и Критику")
    use_presidio: bool = True

    # --- оценка бизнес-ценности (ROI): детерминированно; все коэффициенты — допущения, печатаются в отчёте ---
    roi_review_minutes_per_req_conservative: float = Field(default=1.5, ge=0, description="Ручное ревью требования, мин")
    roi_review_minutes_per_req_base: float = Field(default=2.0, ge=0)
    roi_manual_miss_rate_conservative: float = Field(default=0.4, ge=0, le=1,
                                                     description="Доля дефектов, пропускаемых ручным ревью")
    roi_manual_miss_rate_base: float = Field(default=0.5, ge=0, le=1)
    roi_rework_hours_fatal: float = Field(default=16.0, ge=0, description="Rework на FATAL при позднем обнаружении, ч")
    roi_rework_hours_major: float = Field(default=6.0, ge=0)
    roi_rework_hours_minor_conservative: float = Field(default=0.0, ge=0)
    roi_rework_hours_minor_base: float = Field(default=0.25, ge=0)
    roi_triage_minutes_per_defect: float = Field(default=3.0, ge=0, description="Разбор одного замечания аналитиком")
    roi_hourly_rate: float = Field(default=10_000.0, ge=0, description="Стоимость часа команды (допущение)")
    roi_currency: str = "₸"
    roi_docs_per_month: int = Field(default=30, ge=0, description="Экстраполяция на поток ТЗ в месяц")

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
