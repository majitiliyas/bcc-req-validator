"""LLM-роутер (Спринт 1: только фабрики клиентов).

Ключевое решение: встроенные ретраи SDK отключены (`max_retries=0` у OpenAI-совместимых клиентов,
`HttpRetryOptions(attempts=1)` у google-genai). Повторы, backoff + jitter, каскад провайдеров
и SQLite-кэш будут реализованы здесь же в Спринте 2 — единственным, наблюдаемым слоем.
"""

from __future__ import annotations

from typing import Any

from .config import Settings

OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def make_openai_compatible_client(provider: str, api_key: str, settings: Settings) -> Any:
    """OpenAI-совместимый клиент (Groq / OpenRouter) без скрытых ретраев SDK."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=OPENAI_COMPATIBLE_BASE_URLS[provider],
        max_retries=settings.sdk_max_retries,  # = 0
        timeout=settings.llm_timeout_s,
    )


def make_gemini_client(api_key: str, settings: Settings) -> Any:
    """Клиент google-genai без встроенных повторов (attempts=1 — ровно одна попытка)."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.llm_timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=settings.sdk_max_retries + 1),
        ),
    )
