"""Конфигурация без Groq и NVIDIA: Gemini (deep/fast) + OpenRouter free + локальный Ollama.

Сеть не используется: транспорт подменяется, доступность Ollama — monkeypatch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from req_validator import router as router_mod
from req_validator.config import Settings
from req_validator.models import FindingSource, VerdictStatus
from req_validator.router import (
    ChatRequest,
    ChatResponse,
    Deployment,
    LLMRouter,
    OpenAICompatTransport,
    ProviderError,
    build_groups,
)
from test_auditor_pipeline import DOC, FakeLLM, _no_sleep, audit

GEMINI_ONLY_ENV = {
    "GEMINI_API_KEY": "g-test",
    "OPENROUTER_API_KEY": "or-test",
    "GROQ_API_KEY": "",  # пустые значения, как в .env.example — не ошибка
    "NVIDIA_API_KEY": "",
    "OLLAMA_BASE_URL": "http://localhost:11434",  # без /v1
}


@pytest.fixture
def free_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for k, v in GEMINI_ONLY_ENV.items():
        monkeypatch.setenv(k, v)
    for k in ("DEEP_MODELS", "FAST_MODELS", "ENABLE_OLLAMA", "OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return Settings(_env_file=None, cache_dir=tmp_path / ".cache", use_presidio=False)


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    p = tmp_path / "limits.md"
    p.write_text(DOC, encoding="utf-8")
    return p


def test_groups_without_groq_and_nvidia(free_settings: Settings) -> None:
    assert free_settings.configured_keys() == {"gemini": True, "nvidia": False, "groq": False, "openrouter": True}
    groups = build_groups(free_settings)
    names = {g: [d.name for d in deps] for g, deps in groups.items()}
    providers = {d.provider for deps in groups.values() for d in deps}
    assert providers == {"gemini", "openrouter", "ollama"}  # Groq/NVIDIA тихо пропущены
    assert names["fast"] == ["gemini/gemini-3.5-flash-lite"]  # Gemini — основной fast
    assert names["deep"][0].startswith("gemini/")  # Gemini — основной deep
    assert "openrouter/openrouter/free" in names["deep"]
    assert names["deep"][-1] == "ollama/qwen3.5:9b"  # локальный резерв последним


def test_ollama_url_normalized_and_tuned(free_settings: Settings) -> None:
    for raw in ("http://localhost:11434", "http://localhost:11434/", "http://localhost:11434/v1", "http://localhost:11434/v1/"):
        s = free_settings.model_copy(update={"ollama_base_url": raw})
        assert s.ollama_native_url == "http://localhost:11434"
        assert s.ollama_openai_url == "http://localhost:11434/v1"
    ollama = next(d for d in build_groups(free_settings)["deep"] if d.provider == "ollama")
    assert ollama.base_url == "http://localhost:11434/v1" and ollama.model == "qwen3.5:9b"
    assert ollama.timeout == free_settings.ollama_timeout_s and ollama.spec.max_concurrency == 1
    assert ollama.spec.context_window == free_settings.ollama_context_window
    client = OpenAICompatTransport(free_settings).client(ollama)
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1" and client.max_retries == 0


def test_openrouter_free_slugs_accepted(free_settings: Settings) -> None:
    s = free_settings.model_copy(update={"deep_models": "openrouter/openrouter/free,openrouter/qwen/qwen3-coder:free"})
    deps = build_groups(s)["deep"]
    assert [d.model for d in deps[:2]] == ["openrouter/free", "qwen/qwen3-coder:free"]
    assert all(d.json_mode == "json_object" for d in deps[:2])  # free-модели могут не уметь json_schema


def test_ollama_excluded_when_not_running(free_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "ollama_reachable", lambda s, timeout=0.7: False)
    assert all(d.provider != "ollama" for d in build_groups(free_settings, probe_local=True)["deep"])
    monkeypatch.setattr(router_mod, "ollama_reachable", lambda s, timeout=0.7: True)
    assert build_groups(free_settings, probe_local=True)["deep"][-1].provider == "ollama"


def _router(settings: Settings, fake: Any) -> LLMRouter:
    r = LLMRouter.from_settings(settings, guard=router_mod.AgentBudgetGuard(), transport=fake)
    r.sleep = _no_sleep
    return r


def test_full_audit_gemini_openrouter_ollama(free_settings: Settings, doc_path: Path) -> None:
    from req_validator.analyzer import run_audit

    fake = FakeLLM()
    report = run_audit(doc_path, demo=False, settings=free_settings, router=_router(free_settings, fake))
    used = {dep for dep, _, _ in fake.calls}
    assert used and all(d.startswith(("gemini/", "openrouter/", "ollama/")) for d in used)
    assert any(p.status == "ok" for p in report.passport.llm.passes)
    assert all(p.status in ("ok", "skipped") for p in report.passport.llm.passes)
    assert any(d.source == FindingSource.LLM for d in report.defects)
    assert report.verdict.status == VerdictStatus.REJECTED  # подтверждённый FATAL из Pass 2
    # Критик ушёл на другую модель, чем генератор Pass 2 (Gemini → вторая модель Gemini/OpenRouter)
    gen = {dep for dep, step, _ in fake.calls if step.startswith("pass2")}
    crit = {dep for dep, step, _ in fake.calls if step.startswith("critic")}
    assert crit and not (crit & gen)


class GeminiDown(FakeLLM):
    """Все модели Gemini отвечают 429 — работа должна продолжиться на OpenRouter free и Ollama."""

    async def __call__(self, dep: Deployment, req: ChatRequest) -> ChatResponse:
        if dep.provider == "gemini":
            self.calls.append((dep.name, req.step, req))
            raise ProviderError("rate_limit", "429 quota", status=429, retry_after=1.0)
        return await super().__call__(dep, req)


def test_gemini_quota_exhausted_falls_back_to_openrouter_and_ollama(free_settings: Settings, doc_path: Path) -> None:
    from req_validator.analyzer import run_audit

    fake = GeminiDown()
    report = run_audit(doc_path, demo=False, settings=free_settings, router=_router(free_settings, fake))
    ok_models = set(report.passport.llm.models_used)
    assert ok_models and ok_models <= {"openrouter/openrouter/free", "ollama/qwen3.5:9b"}
    assert report.passport.llm.rate_limited >= 1
    assert all(p.status in ("ok", "skipped") for p in report.passport.llm.passes)


def test_only_ollama_without_any_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc_path: Path) -> None:
    from req_validator.analyzer import run_audit

    for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "DEEP_MODELS", "FAST_MODELS"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None, cache_dir=tmp_path / ".cache", use_presidio=False)
    groups = build_groups(s)
    assert groups["fast"] == [] and [d.name for d in groups["deep"]] == ["ollama/qwen3.5:9b"]
    fake = FakeLLM()
    report = run_audit(doc_path, demo=False, settings=s, router=_router(s, fake))
    assert {dep for dep, _, _ in fake.calls} == {"ollama/qwen3.5:9b"}  # fast → deep → Ollama
    assert report.passport.llm.models_used == {"ollama/qwen3.5:9b": report.passport.llm.models_used["ollama/qwen3.5:9b"]}


class Scripted:
    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.modes: list[str] = []

    async def __call__(self, dep: Deployment, req: ChatRequest) -> ChatResponse:
        self.modes.append(dep.json_mode)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(text=item, prompt_tokens=1, completion_tokens=1, model=dep.name)


def test_json_mode_degrades_to_prompt_instead_of_disabling(free_settings: Settings) -> None:
    deep = [d for d in build_groups(free_settings)["deep"] if d.provider == "openrouter"]
    t = Scripted([ProviderError("json_mode", "response_format json_object is not supported", status=400), "{}"])
    r = LLMRouter({"deep": deep}, t, sleep=_no_sleep)
    req = ChatRequest(messages=[{"role": "user", "content": "x"}], json_schema={"type": "object"}, step="t")
    asyncio.run(r.complete("deep", req))
    assert t.modes == ["json_object", "prompt"] and deep[0].disabled_reason is None


def test_prompt_mode_sends_schema_without_response_format(free_settings: Settings) -> None:
    dep = next(d for d in build_groups(free_settings)["deep"] if d.provider == "ollama")
    dep.json_mode = "prompt"
    captured: dict[str, Any] = {}

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)

            class Msg:
                content = "{}"

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]
                usage = None

            return Resp()

    class FakeClient:
        class chat:  # noqa: N801
            completions = FakeCompletions()

    transport = OpenAICompatTransport(free_settings)
    transport._clients[f"{dep.provider}|{dep.base_url}|{dep.timeout}"] = FakeClient()
    req = ChatRequest(messages=[{"role": "user", "content": "x"}], json_schema={"type": "object"}, step="t")
    asyncio.run(transport(dep, req))
    assert "response_format" not in captured
    assert "JSON Schema" in captured["messages"][-1]["content"] and captured["model"] == "qwen3.5:9b"


def test_empty_choices_is_server_error(free_settings: Settings) -> None:
    dep = next(d for d in build_groups(free_settings)["deep"] if d.provider == "openrouter")

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            class Resp:
                choices: list[Any] = []
                error = {"message": "upstream rate limited"}

            return Resp()

    class FakeClient:
        class chat:  # noqa: N801
            completions = FakeCompletions()

    transport = OpenAICompatTransport(free_settings)
    transport._clients[f"{dep.provider}|{dep.base_url}|{dep.timeout}"] = FakeClient()
    req = ChatRequest(messages=[{"role": "user", "content": "x"}], json_schema={"type": "object"}, step="t")
    with pytest.raises(ProviderError) as exc:
        asyncio.run(transport(dep, req))
    assert exc.value.kind == "server" and "upstream" in str(exc.value)


def test_cli_runs_with_only_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc_path: Path) -> None:
    from req_validator import cli

    monkeypatch.chdir(tmp_path)
    for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "ollama_reachable", lambda s, timeout=0.7: True)
    called: dict[str, Any] = {}

    def fake_run(path: Path, *, demo: bool, settings: Settings):  # noqa: ANN202
        from req_validator.analyzer import run_audit

        called["demo"] = demo
        return run_audit(path, demo=True, settings=settings)  # без сетевых вызовов

    monkeypatch.setattr(cli, "run_audit", fake_run)
    res = CliRunner().invoke(cli.app, ["analyze", str(doc_path), "-o", str(tmp_path / "r.md")])
    assert res.exit_code == 0, res.output
    assert called["demo"] is False and "локальный Ollama" in res.output
