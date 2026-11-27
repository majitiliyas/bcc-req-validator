"""Отказоустойчивый LLM-роутер поверх бесплатных OpenAI-совместимых API ($0 бюджет).

* Один async-клиент `openai.AsyncOpenAI` на провайдера, **встроенные ретраи SDK отключены**
  (`max_retries=0`): все повторы — здесь, наблюдаемо и с учётом квот.
* Группы маршрутизации:
  - `deep` — длинный контекст, глубокий анализ, кросс-секционная сверка, Критик:
    Google AI Studio (Gemini, OpenAI-совместимый endpoint) → NVIDIA NIM → OpenRouter (free) → Ollama (локально);
  - `fast` — короткие шаги: Groq → Gemini Flash-Lite → (fallback) группа `deep`.
* При 429/5xx/таймауте: адаптивный backoff с full jitter, приоритет у `Retry-After`;
  после `allowed_fails` подряд деплоймент уходит в cooldown (`cooldown_time`, не более 60 с).
* Groq: до отправки жёстко проверяется размер входа (< `max_input_tokens`, по умолчанию 5 500
  под лимит 8K TPM) и скользящее окно TPM; при превышении запрос уходит в `deep`.
* `json_schema` никогда не смешивается со `stream=True` или `tools`; если провайдер отвергает
  `json_schema`, деплоймент переключается на `json_object` + схема в промпте.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .cache import canonical_json, sha256_hex
from .config import Settings
from .guards import AgentBudgetGuard, TraceLogger, estimate_tokens

ErrorKind = Literal["rate_limit", "server", "timeout", "network", "auth", "bad_request", "not_found", "json_mode"]
RETRYABLE: set[str] = {"rate_limit", "server", "timeout", "network"}


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str
    key_field: str | None  # поле Settings с ключом; None — ключ не нужен (Ollama)
    rpm: int | None
    context_window: int
    tpm_limit: int | None = None
    max_input_tokens: int | None = None
    json_mode: str = "json_schema"  # json_schema → json_object → prompt (понижается при отказе провайдера)
    max_concurrency: int | None = None  # одновременных запросов к деплойменту (локальная модель — 1)


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini_api_key", rpm=10,
                           context_window=1_000_000),
    "nvidia": ProviderSpec("https://integrate.api.nvidia.com/v1", "nvidia_api_key", rpm=40, context_window=128_000),
    # openrouter/free и *:free маршрутизируются на разные бесплатные модели — json_schema поддерживают не все.
    "openrouter": ProviderSpec("https://openrouter.ai/api/v1", "openrouter_api_key", rpm=20, context_window=64_000,
                               json_mode="json_object"),
    "groq": ProviderSpec("https://api.groq.com/openai/v1", "groq_api_key", rpm=30, context_window=128_000,
                         tpm_limit=8_000, max_input_tokens=5_500),
    "ollama": ProviderSpec("http://localhost:11434/v1", None, rpm=None, context_window=32_768, json_mode="json_object",
                           max_concurrency=1),
}


JSON_MODE_FALLBACK = {"json_schema": "json_object", "json_object": "prompt"}


# --------------------------------------------------------------------------- #
# Запрос / ответ / ошибки
# --------------------------------------------------------------------------- #


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    json_schema: dict[str, Any] | None = None  # JSON Schema ответа (strict=False)
    schema_name: str = "response"
    max_tokens: int = 4096
    temperature: float = 0.0
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    step: str = ""  # метаданные для трасс/тестов, в API не отправляются

    def validate_compat(self) -> None:
        if self.json_schema is not None and (self.stream or self.tools):
            raise ValueError("json_schema нельзя смешивать со stream=True или tools.")

    @property
    def text(self) -> str:
        return "\n".join(m["content"] for m in self.messages)

    @property
    def request_hash(self) -> str:
        return sha256_hex(canonical_json({"m": self.messages, "s": self.json_schema, "t": self.max_tokens}))


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


class AllDeploymentsFailedError(RuntimeError):
    def __init__(self, group: str, attempts: list[str]):
        super().__init__(f"Все модели группы «{group}» недоступны: " + "; ".join(attempts[-6:]))
        self.attempts = attempts


# --------------------------------------------------------------------------- #
# Деплойменты
# --------------------------------------------------------------------------- #


@dataclass
class Deployment:
    name: str  # "groq/qwen/qwen3.8-27b"
    provider: str
    model: str  # slug для API: "qwen/qwen3.8-27b"
    spec: ProviderSpec
    api_key: str | None
    base_url: str
    json_mode: str = "json_schema"
    timeout: float = 120.0
    fails: int = 0
    cooldown_until: float = 0.0
    disabled_reason: str | None = None
    _tpm: deque[tuple[float, int]] = field(default_factory=deque)
    _next_slot: float = 0.0
    _lock: asyncio.Lock | None = None
    _sem: asyncio.Semaphore | None = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def semaphore(self) -> asyncio.Semaphore | None:
        if self.spec.max_concurrency and self._sem is None:
            self._sem = asyncio.Semaphore(self.spec.max_concurrency)
        return self._sem

    def available(self, now: float) -> bool:
        return self.disabled_reason is None and now >= self.cooldown_until

    def tpm_used(self, now: float) -> int:
        while self._tpm and now - self._tpm[0][0] > 60:
            self._tpm.popleft()
        return sum(t for _, t in self._tpm)

    def record_tokens(self, now: float, tokens: int) -> None:
        self._tpm.append((now, tokens))


def parse_deployment(spec_str: str, settings: Settings) -> Deployment | None:
    """'gemini/gemini-3.8-flash' → Deployment; None, если для провайдера нет ключа."""
    provider, _, model = spec_str.strip().partition("/")
    if provider not in PROVIDERS or not model:
        return None
    spec = PROVIDERS[provider]
    key: str | None = None
    if spec.key_field:
        secret = getattr(settings, spec.key_field, None)
        key = secret.get_secret_value().strip() if secret is not None else ""
        if not key or key.startswith(("your_", "optional_")):
            return None
    base_url, timeout = spec.base_url, settings.llm_timeout_s
    if provider == "ollama":
        base_url, timeout = settings.ollama_openai_url, settings.ollama_timeout_s
        spec = replace(spec, context_window=settings.ollama_context_window)
    return Deployment(name=spec_str.strip(), provider=provider, model=model, spec=spec, api_key=key,
                      base_url=base_url, json_mode=spec.json_mode, timeout=timeout)


def ollama_reachable(settings: Settings, timeout: float = 0.7) -> bool:
    """Быстрая проверка локального Ollama (только адрес из настроек, по умолчанию localhost)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{settings.ollama_native_url}/api/tags", timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def build_groups(settings: Settings, *, probe_local: bool = False) -> dict[str, list[Deployment]]:
    """Собрать группы из настроек. probe_local=True — включать Ollama, только если он отвечает."""
    groups: dict[str, list[Deployment]] = {}
    use_ollama = settings.enable_ollama and (not probe_local or ollama_reachable(settings))
    for group, raw in (("deep", settings.deep_models), ("fast", settings.fast_models)):
        deps = [d for s in raw.split(",") if s.strip() and (d := parse_deployment(s, settings))]
        if group == "deep" and use_ollama:
            local = parse_deployment(f"ollama/{settings.ollama_model}", settings)
            if local:
                deps.append(local)
        groups[group] = deps
    return groups


# --------------------------------------------------------------------------- #
# Транспорт (OpenAI-совместимый)
# --------------------------------------------------------------------------- #

Transport = Callable[[Deployment, ChatRequest], Awaitable[ChatResponse]]


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    for name in ("retry-after-ms", "retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = headers.get(name) if hasattr(headers, "get") else None
        if not raw:
            continue
        try:
            value = float(str(raw).rstrip("s"))
            return value / 1000 if name.endswith("-ms") else value
        except ValueError:
            continue
    return None


class OpenAICompatTransport:
    """Реальный транспорт: AsyncOpenAI c max_retries=0 и таймаутом; ошибки SDK → ProviderError."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._clients: dict[str, Any] = {}

    def client(self, dep: Deployment) -> Any:
        from openai import AsyncOpenAI

        key = f"{dep.provider}|{dep.base_url}|{dep.timeout}"
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                api_key=dep.api_key or "ollama", base_url=dep.base_url,
                max_retries=self.settings.sdk_max_retries,  # = 0: никаких скрытых повторов
                timeout=dep.timeout,
            )
        return self._clients[key]

    async def __call__(self, dep: Deployment, req: ChatRequest) -> ChatResponse:
        import openai

        req.validate_compat()
        kwargs: dict[str, Any] = {"model": dep.model, "messages": req.messages, "temperature": req.temperature,
                                  "max_tokens": req.max_tokens}
        if req.json_schema is not None:
            if dep.json_mode == "json_schema":
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": req.schema_name, "schema": req.json_schema, "strict": False}}
            else:
                if dep.json_mode == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
                # json_object и prompt: схема передаётся в промпте, ответ валидирует Pydantic
                schema_hint = ("Верни только JSON-объект, строго соответствующий JSON Schema:\n"
                               + canonical_json(req.json_schema))
                kwargs["messages"] = [*req.messages, {"role": "system", "content": schema_hint}]
        try:
            resp = await self.client(dep).chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderError("timeout", f"таймаут {dep.timeout:.0f} с") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("network", f"нет соединения: {exc}") from exc
        except openai.APIStatusError as exc:
            status = exc.status_code
            body = str(exc)[:300]
            headers = getattr(exc.response, "headers", None)
            if status == 429:
                raise ProviderError("rate_limit", body, status=status, retry_after=_retry_after_seconds(headers)) from exc
            if status in (401, 403):
                raise ProviderError("auth", body, status=status) from exc
            if status == 404:
                raise ProviderError("not_found", body, status=status) from exc
            if status in (400, 422) and req.json_schema is not None and dep.json_mode != "prompt" and any(
                    w in body.lower() for w in ("response_format", "json_schema", "json_object", "json mode",
                                                "schema", "structured")):
                raise ProviderError("json_mode", body, status=status) from exc
            if status >= 500:
                raise ProviderError("server", body, status=status, retry_after=_retry_after_seconds(headers)) from exc
            raise ProviderError("bad_request", body, status=status) from exc
        if not getattr(resp, "choices", None):
            # OpenRouter и др. иногда возвращают 200 с {"error": …} без choices — это сбой апстрима, не ответ.
            err = getattr(resp, "error", None) or getattr(resp, "model_extra", {}).get("error") if resp else None
            raise ProviderError("server", f"пустой ответ провайдера: {str(err)[:200] if err else 'нет choices'}")
        choice = resp.choices[0]
        text = (choice.message.content if choice.message else "") or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None) or estimate_tokens(req.text)
        ct = getattr(usage, "completion_tokens", None) or estimate_tokens(text)
        return ChatResponse(text=text, prompt_tokens=int(pt), completion_tokens=int(ct), model=dep.name)


# --------------------------------------------------------------------------- #
# Роутер
# --------------------------------------------------------------------------- #

EMPTY_MAX_TOKENS = 5


def is_empty_response(resp: ChatResponse) -> bool:
    """Пустой ответ: нет текста или ≤ 5 токенов, которые не разбираются как JSON."""
    import json

    from .llm_schemas import extract_json_text

    text = extract_json_text(resp.text).strip()
    if not text:
        return True
    if resp.completion_tokens <= EMPTY_MAX_TOKENS:
        try:
            json.loads(text)
        except ValueError:
            return True
    return False


@dataclass
class RouterStats:
    calls: int = 0
    empty_responses: int = 0
    attempts: int = 0
    rate_limited: int = 0
    reroutes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models_used: dict[str, int] = field(default_factory=dict)


class LLMRouter:
    GROUP_FALLBACK = {"fast": ["fast", "deep"], "deep": ["deep"]}

    def __init__(
        self,
        groups: dict[str, list[Deployment]],
        transport: Transport,
        *,
        guard: AgentBudgetGuard | None = None,
        tracer: TraceLogger | None = None,
        allowed_fails: int = 2,
        cooldown_time: float = 30.0,
        max_cooldown: float = 60.0,
        max_attempts_per_deployment: int = 3,
        backoff_base: float = 2.0,
        backoff_cap: float = 30.0,
        max_retry_after: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.groups = groups
        self.transport = transport
        self.guard = guard or AgentBudgetGuard()
        self.tracer = tracer
        self.allowed_fails = allowed_fails
        self.cooldown_time = cooldown_time
        self.max_cooldown = max_cooldown
        self.max_attempts = max_attempts_per_deployment
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.max_retry_after = max_retry_after
        self.sleep = sleep
        self.clock = clock
        self.rng = rng or random.Random()
        self.stats = RouterStats()

    @classmethod
    def from_settings(cls, settings: Settings, *, guard: AgentBudgetGuard, tracer: TraceLogger | None = None,
                      transport: Transport | None = None) -> LLMRouter:
        groups = build_groups(settings, probe_local=transport is None)
        return cls(
            groups, transport or OpenAICompatTransport(settings), guard=guard, tracer=tracer,
            allowed_fails=settings.allowed_fails, cooldown_time=settings.cooldown_time,
            max_attempts_per_deployment=settings.max_attempts_per_deployment, backoff_base=settings.backoff_base,
            backoff_cap=settings.backoff_cap, max_retry_after=settings.max_retry_after,
        )

    def has_deployments(self) -> bool:
        return any(self.groups.values())

    def _trace(self, event: str, **data: Any) -> None:
        if self.tracer is not None:
            self.tracer.event(event, **data)

    def backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        """Full jitter: U(0, min(cap, base·2^attempt)); Retry-After имеет приоритет (+ небольшой jitter)."""
        if retry_after is not None:
            return min(retry_after, self.max_retry_after) + self.rng.uniform(0, 0.5)
        return self.rng.uniform(0, min(self.backoff_cap, self.backoff_base * (2 ** attempt)))

    def _cooldown(self, dep: Deployment, retry_after: float | None) -> None:
        wait = min(self.max_cooldown, max(self.cooldown_time, retry_after or 0.0))
        dep.cooldown_until = self.clock() + wait
        dep.fails = 0
        self._trace("cooldown", deployment=dep.name, seconds=wait)

    async def _throttle(self, dep: Deployment) -> None:
        if not dep.spec.rpm:
            return
        async with dep.lock:
            now = self.clock()
            wait = dep._next_slot - now
            dep._next_slot = max(now, dep._next_slot) + 60.0 / dep.spec.rpm
        if wait > 0:
            await self.sleep(wait)

    def _ordered(self, group: str, prefer_not: set[str]) -> list[Deployment]:
        deps = list(self.groups.get(group, []))
        return [d for d in deps if d.name not in prefer_not] + [d for d in deps if d.name in prefer_not]

    async def complete(self, group: str, req: ChatRequest, *, prefer_not: set[str] | None = None,
                       exclude: set[str] | None = None, pin: str | None = None) -> ChatResponse:
        """Один логический вызов LLM с маршрутизацией, ретраями, cooldown и контролем бюджета.

        prefer_not — модели, которые пробуются последними (независимость Критика);
        exclude    — модели, которые нельзя использовать (уже провалили этот шаг);
        pin        — только указанная модель (ретрай валидации на той же модели).
        """
        req.validate_compat()
        est_in = estimate_tokens(req.text)
        exclude = exclude or set()
        call_key = sha256_hex(canonical_json({"r": req.request_hash, "x": sorted(exclude), "p": pin}))
        self.guard.before_call(call_key, est_in + req.max_tokens)
        self.stats.calls += 1
        attempts_log: list[str] = []
        prefer_not = prefer_not or set()
        for g in self.GROUP_FALLBACK.get(group, [group]):
            for dep in self._ordered(g, prefer_not):
                if dep.name in exclude or (pin is not None and dep.name != pin):
                    continue
                now = self.clock()
                if not dep.available(now):
                    attempts_log.append(f"{dep.name}: недоступен ({dep.disabled_reason or 'cooldown'})")
                    continue
                if dep.spec.max_input_tokens and est_in > dep.spec.max_input_tokens:
                    self.stats.reroutes += 1
                    attempts_log.append(f"{dep.name}: вход ~{est_in} ток. > {dep.spec.max_input_tokens}")
                    self._trace("tpm_guard_reroute", deployment=dep.name, est_input=est_in,
                                limit=dep.spec.max_input_tokens, step=req.step)
                    continue
                if dep.spec.tpm_limit and dep.tpm_used(now) + est_in > dep.spec.tpm_limit:
                    self.stats.reroutes += 1
                    attempts_log.append(f"{dep.name}: окно TPM исчерпано")
                    self._trace("tpm_window_reroute", deployment=dep.name, used=dep.tpm_used(now), step=req.step)
                    continue
                if est_in + req.max_tokens > dep.spec.context_window:
                    attempts_log.append(f"{dep.name}: не помещается в контекст")
                    continue
                result = await self._try_deployment(dep, req, est_in, attempts_log)
                if result is not None:
                    self.guard.record_success(result.prompt_tokens, result.completion_tokens)
                    self.stats.prompt_tokens += result.prompt_tokens
                    self.stats.completion_tokens += result.completion_tokens
                    self.stats.models_used[dep.name] = self.stats.models_used.get(dep.name, 0) + 1
                    return result
        self.guard.record_failure()
        self._trace("all_failed", group=group, step=req.step, attempts=attempts_log[-8:])
        raise AllDeploymentsFailedError(group, attempts_log)

    async def _try_deployment(self, dep: Deployment, req: ChatRequest, est_in: int,
                              log: list[str]) -> ChatResponse | None:
        attempt = 0
        while attempt < self.max_attempts:
            await self._throttle(dep)
            self.stats.attempts += 1
            t0 = self.clock()
            try:
                sem = dep.semaphore
                if sem is not None:
                    async with sem:
                        resp = await self.transport(dep, req)
                else:
                    resp = await self.transport(dep, req)
            except ProviderError as err:
                self._trace("llm_error", deployment=dep.name, step=req.step, kind=err.kind, status=err.status,
                            retry_after=err.retry_after, attempt=attempt)
                if err.kind == "json_mode" and dep.json_mode in JSON_MODE_FALLBACK:
                    # провайдер не поддерживает режим: json_schema → json_object → схема только в промпте
                    dep.json_mode = JSON_MODE_FALLBACK[dep.json_mode]
                    self._trace("json_mode_downgrade", deployment=dep.name, mode=dep.json_mode)
                    continue
                if err.kind in ("auth", "not_found", "bad_request", "json_mode"):
                    dep.disabled_reason = f"{err.kind} {err.status or ''}".strip()
                    log.append(f"{dep.name}: {err.kind} ({err.status})")
                    return None
                if err.kind == "rate_limit":
                    self.stats.rate_limited += 1
                dep.fails += 1
                log.append(f"{dep.name}: {err.kind} ({err.status})")
                if dep.fails >= self.allowed_fails or (err.retry_after or 0) > self.max_retry_after:
                    self._cooldown(dep, err.retry_after)
                    return None
                await self.sleep(self.backoff_delay(attempt, err.retry_after))
                attempt += 1
                continue
            dep.record_tokens(self.clock(), resp.prompt_tokens + resp.completion_tokens)
            if req.json_schema is not None and is_empty_response(resp):
                # Пустой ответ (≤ 5 токенов без валидного JSON) детерминирован при temperature=0:
                # повтор на той же модели бесполезен — переходим к следующей модели цепочки.
                self.stats.empty_responses += 1
                self.guard.tokens_used += resp.prompt_tokens + resp.completion_tokens
                log.append(f"{dep.name}: пустой ответ ({resp.completion_tokens} ток.)")
                self._trace("empty_response", deployment=dep.name, step=req.step,
                            completion_tokens=resp.completion_tokens, text=resp.text[:40])
                return None
            dep.fails = 0
            self._trace("llm_call", deployment=dep.name, step=req.step, prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens, latency_s=round(self.clock() - t0, 3))
            return resp
        return None
