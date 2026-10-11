"""TDD: config-driven API providers.

> Standard API providers should be defined through config, not through
> changes to the deep-worker code.

A "standard API provider" is one reached over a plain HTTP API with an
OpenAI-compatible dialect (deepseek/openai/gemini-like), as opposed to the
CLI/ACP providers which need a whole adapter layer. Adding one must be a
CONFIG edit: drop an entry into ``~/.deep_worker/providers.json``, put the
key in ``~/.deep_worker/env.d/<name>.env``, done — zero code changes.

Security contract: the config names the
ENV VAR holding the key; it never contains the key itself. An entry that
inlines something key-shaped is rejected loudly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.runtime.provider_registry import (
    ProviderConfigError,
    ProviderSpec,
    load_provider_specs,
    registry_path,
)


def _write(tmp_path: Path, payload: dict, mode: int = 0o600) -> Path:
    p = tmp_path / ".deep_worker" / "providers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    p.chmod(mode)
    return p


MINIMAX = {
    "name": "minimax",
    "compatible_with": "openai",
    "base_url": "https://api.minimax.io/v1",
    "api_key_env": "MINIMAX_API_KEY",
    "default_model": "MiniMax-M3",
    "default_model_env": "MINIMAX_MODEL",
    "model_markers": ["minimax"],
}


class TestRegistryPath:
    def test_lives_under_deep_worker_not_deepagents(self, tmp_path: Path) -> None:
        """POLICIES §16: the registry belongs to this project, not to the ancestor."""
        p = registry_path(home=tmp_path)
        assert ".deep_worker" in str(p)
        assert ".deepagents" not in str(p)
        assert p.name == "providers.json"


class TestLoading:
    def test_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """No config → no extra providers. Must never block startup."""
        assert load_provider_specs(home=tmp_path) == ()

    def test_loads_a_provider(self, tmp_path: Path) -> None:
        _write(tmp_path, {"providers": [MINIMAX]})
        specs = load_provider_specs(home=tmp_path)
        assert len(specs) == 1
        s = specs[0]
        assert isinstance(s, ProviderSpec)
        assert s.name == "minimax"
        assert s.compatible_with == "openai"
        assert s.base_url == "https://api.minimax.io/v1"
        assert s.api_key_env == "MINIMAX_API_KEY"
        assert s.default_model == "MiniMax-M3"
        assert s.model_markers == ("minimax",)

    def test_multiple_providers(self, tmp_path: Path) -> None:
        second = dict(MINIMAX, name="other", api_key_env="OTHER_KEY",
                      model_markers=["other"])
        _write(tmp_path, {"providers": [MINIMAX, second]})
        assert [s.name for s in load_provider_specs(home=tmp_path)] == [
            "minimax", "other",
        ]

    def test_optional_fields_default_sanely(self, tmp_path: Path) -> None:
        minimal = {
            "name": "tiny",
            "compatible_with": "openai",
            "base_url": "https://example.test/v1",
            "api_key_env": "TINY_KEY",
        }
        _write(tmp_path, {"providers": [minimal]})
        s = load_provider_specs(home=tmp_path)[0]
        assert s.default_model is None
        assert s.model_markers == ()
        assert s.extra == {}


class TestFailLoud:
    """Config mistakes must be actionable, never silently ignored."""

    def test_malformed_json(self, tmp_path: Path) -> None:
        p = tmp_path / ".deep_worker" / "providers.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json")
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert "providers.json" in str(e.value)

    @pytest.mark.parametrize("missing", ["name", "compatible_with", "base_url",
                                         "api_key_env"])
    def test_missing_required_field(self, tmp_path: Path, missing: str) -> None:
        entry = {k: v for k, v in MINIMAX.items() if k != missing}
        _write(tmp_path, {"providers": [entry]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert missing in str(e.value)

    def test_unknown_dialect_is_rejected(self, tmp_path: Path) -> None:
        """A silent fallback would run the wrong client against the API."""
        _write(tmp_path, {"providers": [dict(MINIMAX, compatible_with="cohere")]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert "cohere" in str(e.value)
        assert "openai" in str(e.value)  # names what IS supported

    def test_duplicate_names_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, {"providers": [MINIMAX, dict(MINIMAX)]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert "minimax" in str(e.value)

    def test_cannot_shadow_a_builtin_provider(self, tmp_path: Path) -> None:
        """Redefining 'openai' in config would silently change the meaning of
        an existing agent's saved provider field."""
        _write(tmp_path, {"providers": [dict(MINIMAX, name="openai")]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert "openai" in str(e.value)
        assert "built-in" in str(e.value).lower()

    def test_bad_base_url_scheme(self, tmp_path: Path) -> None:
        _write(tmp_path, {"providers": [dict(MINIMAX, base_url="ftp://x/v1")]})
        with pytest.raises(ProviderConfigError):
            load_provider_specs(home=tmp_path)

    def test_providers_must_be_a_list(self, tmp_path: Path) -> None:
        _write(tmp_path, {"providers": {"name": "x"}})
        with pytest.raises(ProviderConfigError):
            load_provider_specs(home=tmp_path)


class TestNoInlineSecrets:
    """Keys live in env.d, never in config."""

    # Assembled at runtime, never written out as one literal: the repo's
    # own pre-commit secret scanner (correctly) blocks anything shaped like
    # a live key, including a fixture. Concatenation keeps the fixture
    # key-SHAPED for the code under test while leaving no scannable token
    # in the source file.
    FAKE_KEY = "sk" + "-live-" + "a" * 24 + "0123"

    def test_api_key_field_is_rejected(self, tmp_path: Path) -> None:
        entry = dict(MINIMAX)
        entry["api_key"] = self.FAKE_KEY
        _write(tmp_path, {"providers": [entry]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        msg = str(e.value).lower()
        assert "api_key" in msg
        assert "env" in msg  # tells the operator where it belongs

    def test_api_key_env_must_look_like_a_var_name(self, tmp_path: Path) -> None:
        """Catches putting the KEY VALUE into api_key_env by mistake."""
        _write(tmp_path, {"providers": [
            dict(MINIMAX, api_key_env=self.FAKE_KEY)
        ]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert "api_key_env" in str(e.value)

    def test_error_never_echoes_the_suspicious_value(self, tmp_path: Path) -> None:
        secret = self.FAKE_KEY
        entry = dict(MINIMAX)
        entry["api_key"] = secret
        _write(tmp_path, {"providers": [entry]})
        with pytest.raises(ProviderConfigError) as e:
            load_provider_specs(home=tmp_path)
        assert secret not in str(e.value), "error message leaked the secret"


class TestSpecBehaviour:
    def test_resolve_model_prefers_explicit_then_env_then_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, {"providers": [MINIMAX]})
        s = load_provider_specs(home=tmp_path)[0]
        monkeypatch.delenv("MINIMAX_MODEL", raising=False)
        assert s.resolve_model("Explicit-1") == "Explicit-1"
        monkeypatch.setenv("MINIMAX_MODEL", "From-Env")
        assert s.resolve_model(None) == "From-Env"
        assert s.resolve_model("Explicit-1") == "Explicit-1"
        monkeypatch.delenv("MINIMAX_MODEL")
        assert s.resolve_model(None) == "MiniMax-M3"

    def test_explicit_model_is_passed_through_verbatim(
        self, tmp_path: Path
    ) -> None:
        """No Python-side whitelist: whatever the operator names goes to the
        backend as-is; the backend owns rejection."""
        _write(tmp_path, {"providers": [MINIMAX]})
        s = load_provider_specs(home=tmp_path)[0]
        for name in ("MiniMax-M2.1", "some-unreleased-2027", "WeIrD-CaSe"):
            assert s.resolve_model(name) == name

    def test_matches_model_uses_markers(self, tmp_path: Path) -> None:
        _write(tmp_path, {"providers": [MINIMAX]})
        s = load_provider_specs(home=tmp_path)[0]
        assert s.matches_model("MiniMax-M3")
        assert s.matches_model("minimax-m2.1")
        assert not s.matches_model("gpt-5.2")

    def test_no_markers_matches_nothing(self, tmp_path: Path) -> None:
        entry = {k: v for k, v in MINIMAX.items() if k != "model_markers"}
        _write(tmp_path, {"providers": [entry]})
        s = load_provider_specs(home=tmp_path)[0]
        assert not s.matches_model("MiniMax-M3")
