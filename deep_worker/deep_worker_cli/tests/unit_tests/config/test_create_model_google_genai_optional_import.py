"""Tests for the optional ``langchain_google_genai`` import in
``create_model`` (B3).

DESIGN_PROVIDERS_FROM_WHEEL §1 B3: the google branch and the vertexai
else-branch (non-claude gemini models) import
``langchain_google_genai.ChatGoogleGenerativeAI``. The package is not a
hard dependency of every deployment, so the import MUST be optional: when
the package is missing the branch MUST raise ``CapabilityError`` with the
canonical capability name and an actionable remedy — never a raw
``ModuleNotFoundError``.

The google branch is guarded with ``capability="google"`` and the vertexai
gemini branch with ``capability="vertexai"`` (the remedy points at the
``[vertexai]`` extra, which is what enables Vertex AI gemini models).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.exceptions import CapabilityError


def _fake_import_error(name: str) -> ImportError:
    exc = ImportError(f"cannot import name {name!r}")
    exc.name = name
    return exc


def _make_import_guard(
    missing_module: str,
) -> tuple[object, ...]:
    """Return a fake ``__import__`` that fails only for missing_module."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == missing_module or name.startswith(missing_module + "."):
            raise _fake_import_error(name)
        return real_import(name, *args, **kwargs)

    return (fake_import,)


def _mock_google_settings(mock_settings: MagicMock) -> None:
    mock_settings.has_google = True
    mock_settings.google_api_key = "test-key"
    mock_settings.google_cloud_project = "test-project"


def _mock_vertexai_settings(mock_settings: MagicMock) -> None:
    mock_settings.has_vertex_ai = True
    mock_settings.google_cloud_project = "test-project"


# ---------------------------------------------------------------------------
# google branch
# ---------------------------------------------------------------------------


def test_google_branch_raises_capability_error_when_package_missing() -> None:
    """provider == "google" without langchain-google-genai installed MUST
    raise CapabilityError (capability="google") with a pip remedy."""
    from deep_worker_cli.config import create_model

    fake_import = _make_import_guard("langchain_google_genai")[0]
    with patch("deep_worker_cli.config.settings") as mock_settings, patch(
        "builtins.__import__", side_effect=fake_import
    ):
        _mock_google_settings(mock_settings)
        with pytest.raises(CapabilityError) as exc_info:
            create_model(provider="google", model_name="gemini-2.0-flash")

    assert exc_info.value.capability == "google"
    assert "pip install" in str(exc_info.value)
    assert "langchain-google-genai" in exc_info.value.remedy


def test_google_branch_proceeds_when_package_available() -> None:
    """With the package importable the google branch constructs
    ChatGoogleGenerativeAI as before."""
    from deep_worker_cli.config import create_model

    fake_model = MagicMock()
    with patch("deep_worker_cli.config.settings") as mock_settings, patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", return_value=fake_model
    ) as mock_cls:
        _mock_google_settings(mock_settings)
        result = create_model(provider="google", model_name="gemini-2.0-flash")

    mock_cls.assert_called_once()
    assert result is fake_model


# ---------------------------------------------------------------------------
# vertexai else-branch (gemini models without "claude" in the name)
# ---------------------------------------------------------------------------


def test_vertexai_else_branch_raises_capability_error_when_package_missing() -> None:
    """provider == "vertexai" with a gemini model (no "claude" in the name)
    without langchain-google-genai installed MUST raise CapabilityError
    (capability="vertexai") with the [vertexai] extra remedy."""
    from deep_worker_cli.config import create_model

    fake_import = _make_import_guard("langchain_google_genai")[0]
    with patch("deep_worker_cli.config.settings") as mock_settings, patch(
        "builtins.__import__", side_effect=fake_import
    ):
        _mock_vertexai_settings(mock_settings)
        with pytest.raises(CapabilityError) as exc_info:
            create_model(provider="vertexai", model_name="gemini-2.0-flash")

    assert exc_info.value.capability == "vertexai"
    assert "pip install" in str(exc_info.value)
    assert "vertexai" in exc_info.value.remedy


def test_vertexai_else_branch_proceeds_when_package_available() -> None:
    """With the package importable the vertexai gemini branch constructs
    ChatGoogleGenerativeAI with vertexai=True."""
    from deep_worker_cli.config import create_model

    fake_model = MagicMock()
    with patch("deep_worker_cli.config.settings") as mock_settings, patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", return_value=fake_model
    ) as mock_cls:
        _mock_vertexai_settings(mock_settings)
        result = create_model(provider="vertexai", model_name="gemini-2.0-flash")

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["vertexai"] is True
    assert call_kwargs["project"] == "test-project"
    assert result is fake_model
