"""The ``compatible_with="azure"`` dialect for config-declared providers.

Azure differs from the OpenAI dialect in WHERE the model name travels: it is
the *deployment*, and it goes in the URL path rather than the request body.
Get that wrong and the client silently posts to a non-existent URL, so the
load-bearing assertion here is on the assembled base_url, not just on kwargs.

Attribute-vs-alias trap (langchain_openai 1.1.7): the constructor accepts the
aliases ``azure_deployment`` / ``api_version`` / ``api_key`` while the model
exposes ``deployment_name`` / ``openai_api_version`` / ``openai_api_key``.
These tests assert on the ATTRIBUTES, so a future refactor that passes the
attribute names as kwargs (which pydantic would reject) cannot pass silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.runtime.provider_registry import (
    SUPPORTED_AUTH_HEADERS,
    SUPPORTED_DIALECTS,
    ProviderConfigError,
    load_provider_specs,
)


AZURE_ENTRY = {
    "name": "azuretest",
    "compatible_with": "azure",
    "base_url": "https://example-resource.openai.azure.com",
    "api_key_env": "AZURETEST_API_KEY",
    "api_version": "2024-10-21",
    "default_model": "my-gpt4o-deployment",
    "default_model_env": "AZURETEST_MODEL",
    "model_markers": ["my-gpt4o"],
}


def _write_registry(home: Path, entry: dict) -> None:
    reg = home / ".deep_worker"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "providers.json").write_text(json.dumps({"providers": [entry]}))


@pytest.fixture
def azure_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the registry at a temp providers.json and supply a key."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("AZURETEST_API_KEY", "test-key-not-real")
    monkeypatch.delenv("AZURETEST_MODEL", raising=False)
    return tmp_path


# --------------------------------------------------------------------------
# Registry: dialect acceptance and fail-fast validation
# --------------------------------------------------------------------------


def test_azure_is_a_supported_dialect() -> None:
    assert "azure" in SUPPORTED_DIALECTS
    # The pre-existing dialect must not be displaced by the addition.
    assert "openai" in SUPPORTED_DIALECTS


def test_azure_entry_loads_with_api_version_and_default_auth_header(
    azure_home: Path,
) -> None:
    _write_registry(azure_home, AZURE_ENTRY)

    (spec,) = load_provider_specs(azure_home)

    assert spec.compatible_with == "azure"
    assert spec.api_version == "2024-10-21"
    # Azure's own header is the default; the gateway variant is opt-in.
    assert spec.auth_header == "api-key"


def test_azure_without_api_version_is_rejected_at_load_time(
    azure_home: Path,
) -> None:
    """Fail-fast: Azure rejects a request with no ?api-version=, and that
    error would otherwise surface opaquely on the operator's first turn."""
    entry = {k: v for k, v in AZURE_ENTRY.items() if k != "api_version"}
    _write_registry(azure_home, entry)

    with pytest.raises(ProviderConfigError) as exc:
        load_provider_specs(azure_home)

    msg = str(exc.value)
    assert "api_version" in msg
    assert "azuretest" in msg


def test_openai_dialect_does_not_require_api_version(azure_home: Path) -> None:
    """Negative control for the check above: the requirement is azure-only.

    Without this, a bug that demanded api_version from EVERY provider would
    still satisfy the azure test.
    """
    entry = dict(AZURE_ENTRY)
    entry["compatible_with"] = "openai"
    entry["base_url"] = "https://api.example.test/v1"
    del entry["api_version"]
    _write_registry(azure_home, entry)

    (spec,) = load_provider_specs(azure_home)

    assert spec.compatible_with == "openai"
    assert spec.api_version is None


@pytest.mark.parametrize("header", sorted(SUPPORTED_AUTH_HEADERS))
def test_supported_auth_headers_load(azure_home: Path, header: str) -> None:
    entry = dict(AZURE_ENTRY, auth_header=header)
    _write_registry(azure_home, entry)

    (spec,) = load_provider_specs(azure_home)

    assert spec.auth_header == header


def test_unknown_auth_header_is_rejected_at_load_time(azure_home: Path) -> None:
    entry = dict(AZURE_ENTRY, auth_header="x-my-own-header")
    _write_registry(azure_home, entry)

    with pytest.raises(ProviderConfigError) as exc:
        load_provider_specs(azure_home)

    assert "auth_header" in str(exc.value)


# --------------------------------------------------------------------------
# Client construction: the deployment must reach the URL PATH
# --------------------------------------------------------------------------


def test_deployment_goes_into_the_url_path_not_the_body(
    azure_home: Path,
) -> None:
    """The whole point of the dialect: model name == deployment == URL path."""
    _write_registry(azure_home, AZURE_ENTRY)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_azure_model

    model = _create_azure_model(spec, "my-gpt4o-deployment", None)

    assert model.deployment_name == "my-gpt4o-deployment"
    assert model.openai_api_version == "2024-10-21"
    assert model.azure_endpoint == "https://example-resource.openai.azure.com"
    # Load-bearing: the deployment is in the PATH. A regression that passed
    # the name as ``model=`` would leave it out of this URL entirely.
    base_url = str(model.client._client.base_url)
    assert "/openai/deployments/my-gpt4o-deployment" in base_url
    # And it must NOT be sent as the body's model field.
    assert model.model_name != "my-gpt4o-deployment"


def test_stream_usage_is_enabled(azure_home: Path) -> None:
    """Azure is never the DEFAULT base URL, so langchain_openai will not
    auto-enable usage reporting — without an explicit flag the token-stats
    subsystem records ``no_usage`` (same trap as the openai branch)."""
    _write_registry(azure_home, AZURE_ENTRY)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_azure_model

    model = _create_azure_model(spec, "my-gpt4o-deployment", None)

    assert model.stream_usage is True


def test_api_key_mode_sends_no_authorization_header(azure_home: Path) -> None:
    _write_registry(azure_home, AZURE_ENTRY)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_azure_model

    model = _create_azure_model(spec, "my-gpt4o-deployment", None)

    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "test-key-not-real"
    headers = model.default_headers or {}
    assert "Authorization" not in headers


def test_authorization_mode_sends_bearer_and_withholds_the_real_key(
    azure_home: Path,
) -> None:
    """Gateway mode: the credential travels in Authorization, and the
    api_key the SDK insists on must be an unused placeholder rather than the
    real secret (which would then go out in Azure's api-key header too)."""
    _write_registry(azure_home, dict(AZURE_ENTRY, auth_header="authorization"))
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_azure_model

    model = _create_azure_model(spec, "my-gpt4o-deployment", None)

    assert model.default_headers == {"Authorization": "Bearer test-key-not-real"}
    # Both arms matter: the placeholder must not be the real key.
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() != "test-key-not-real"


def test_missing_deployment_is_a_clear_error_not_an_invented_default(
    azure_home: Path,
) -> None:
    """No model name is ever hardcoded in Python
    (feedback_no_model_hardcode_2026_08_15); with nothing resolved the
    operator gets an actionable error, not a guess."""
    _write_registry(azure_home, AZURE_ENTRY)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_azure_model

    with pytest.raises(ValueError) as exc:
        _create_azure_model(spec, "", None)

    msg = str(exc.value)
    assert "deployment" in msg.lower()
    assert "azuretest" in msg


def test_dispatch_from_config_provider_factory_reaches_azure_branch(
    azure_home: Path,
) -> None:
    """The azure branch must be reachable through the real entry point, not
    only by calling _create_azure_model directly."""
    _write_registry(azure_home, AZURE_ENTRY)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_config_provider_model

    model = _create_config_provider_model(spec, "my-gpt4o-deployment", None)

    assert type(model).__name__ == "AzureChatOpenAI"
    assert model.deployment_name == "my-gpt4o-deployment"


def test_openai_dialect_still_builds_chatopenai(azure_home: Path) -> None:
    """Negative control for the dispatch above: adding azure must not
    redirect the openai dialect."""
    entry = dict(AZURE_ENTRY)
    entry["compatible_with"] = "openai"
    entry["base_url"] = "https://api.example.test/v1"
    del entry["api_version"]
    _write_registry(azure_home, entry)
    (spec,) = load_provider_specs(azure_home)

    from deep_worker_cli.config import _create_config_provider_model

    model = _create_config_provider_model(spec, "Cfg-Model-1", None)

    assert type(model).__name__ == "ChatOpenAI"
