"""Config-declared providers must be wired like built-ins in create_model.

Regression origin: the first implementation returned early for a
config-declared provider, which skipped the shared tail of ``create_model``
— so the model context limit and, critically, the TOKEN-STATS handler were
never attached. Everything looked fine (the agent answered) while silently
recording no usage. Live-caught, not test-caught; these tests exist so it
cannot come back.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


MINIMAX_ENTRY = {
    "name": "cfgtest",
    "compatible_with": "openai",
    "base_url": "https://api.example.test/v1",
    "api_key_env": "CFGTEST_API_KEY",
    "default_model": "Cfg-Model-1",
    "default_model_env": "CFGTEST_MODEL",
    "model_markers": ["cfg-model"],
}


@pytest.fixture
def cfg_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the registry at a temp providers.json and supply a key."""
    home = tmp_path
    reg = home / ".deep_worker"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "providers.json").write_text(json.dumps({"providers": [MINIMAX_ENTRY]}))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("CFGTEST_API_KEY", "test-key-not-real")
    monkeypatch.delenv("CFGTEST_MODEL", raising=False)

    from deep_worker_cli import config as C

    monkeypatch.setattr(C, "_CONFIG_PROVIDER_CACHE", None, raising=False)
    yield C
    monkeypatch.setattr(C, "_CONFIG_PROVIDER_CACHE", None, raising=False)


def _factory():
    from deep_worker_cli.stats.callback import TokenStatsCallbackHandler

    calls: list[dict] = []

    def factory(kind, *, provider, model_name):
        calls.append({"kind": kind, "provider": provider, "model_name": model_name})
        return TokenStatsCallbackHandler(
            lambda rec: None, kind=kind, provider=provider, model_name=model_name
        )

    return factory, calls


def _has_stats(model) -> bool:
    from deep_worker_cli.stats.callback import TokenStatsCallbackHandler

    return any(
        isinstance(cb, TokenStatsCallbackHandler)
        for cb in (getattr(model, "callbacks", None) or [])
    )


class TestTokenStatsAttached:
    """THE REGRESSION: usage must be recorded for config providers too."""

    def test_stats_handler_attached_for_explicit_provider(self, cfg_provider) -> None:
        factory, calls = _factory()
        model = cfg_provider.create_model(
            provider="cfgtest", kind="agent", stats_callback_factory=factory
        )
        assert _has_stats(model), (
            "no TokenStatsCallbackHandler on a config-declared provider — "
            "usage would be silently unrecorded"
        )
        assert calls and calls[0]["provider"] == "cfgtest"

    def test_stats_handler_attached_for_autodetected_provider(
        self, cfg_provider
    ) -> None:
        factory, _ = _factory()
        model = cfg_provider.create_model(
            model_name="Cfg-Model-9", kind="agent", stats_callback_factory=factory
        )
        assert _has_stats(model)

    def test_is_standard_provider_recognises_config_providers(
        self, cfg_provider
    ) -> None:
        from deep_worker_cli.stats.token_stats import is_standard_provider

        assert is_standard_provider("cfgtest")
        assert not is_standard_provider("claude-code")


class TestSettingsReflectTheConfigProvider:
    def test_settings_record_provider_and_model(self, cfg_provider) -> None:
        cfg_provider.create_model(provider="cfgtest")
        assert cfg_provider.settings.model_provider == "cfgtest"
        assert cfg_provider.settings.model_name == "Cfg-Model-1"

    def test_autodetect_sets_provider_not_a_builtin(self, cfg_provider) -> None:
        cfg_provider.create_model(model_name="Cfg-Model-7")
        assert cfg_provider.settings.model_provider == "cfgtest"


class TestModelResolution:
    def test_default_model_used_when_none_given(self, cfg_provider) -> None:
        model = cfg_provider.create_model(provider="cfgtest")
        assert model.model_name == "Cfg-Model-1"

    def test_env_overrides_config_default(
        self, cfg_provider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CFGTEST_MODEL", "Cfg-From-Env")
        assert cfg_provider.create_model(provider="cfgtest").model_name == "Cfg-From-Env"

    def test_explicit_model_wins_and_is_verbatim(self, cfg_provider) -> None:
        """No local whitelist — the vendor owns rejection."""
        model = cfg_provider.create_model(
            provider="cfgtest", model_name="Totally-New-2027"
        )
        assert model.model_name == "Totally-New-2027"

    def test_base_url_and_key_come_from_config(self, cfg_provider) -> None:
        model = cfg_provider.create_model(provider="cfgtest")
        assert str(model.openai_api_base) == "https://api.example.test/v1"


class TestBuiltinsUnaffected:
    def test_builtin_detection_still_works(self, cfg_provider) -> None:
        """A config provider must not steal models it has no marker for."""
        from deep_worker_cli.config import _detect_provider

        assert _detect_provider("deepseek-chat") == "deepseek"
        assert _detect_provider("gpt-5.2") == "openai"
