"""Tests for the provider capability guards in ``create_model`` (B2).

Capability-guard contract: the claude-code-sdk / codex / grok branches
of ``config.create_model`` must resolve the vendored provider through
``provider_paths.ensure_provider_importable`` instead of sys.path hacks.
When the provider package is unavailable (no source root, not installed)
the branch MUST raise ``CapabilityError`` with the canonical capability
name and an actionable remedy — never a raw ``ModuleNotFoundError`` and
never a silent ``sys.path`` mutation.

These tests patch ``ensure_provider_importable`` at the provider_paths
module level (the call sites import it locally inside the branch), so
the resolution result is fully controlled without touching sys.path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.provider_paths import ProviderOrigin, ProviderResolution


def _unavailable_resolution(pkg: str, remedy: str | None = None) -> ProviderResolution:
    """A resolution whose ``unavailable`` field is set (provider absent)."""
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="not found",
        unavailable="provider package not installed",
        remedy=remedy,
    )


def _available_resolution(pkg: str) -> ProviderResolution:
    """A resolution whose ``unavailable`` is None (provider importable)."""
    return ProviderResolution(
        pkg=pkg,
        mode="source",
        origin=ProviderOrigin.SOURCE,
        roots=("/tmp/example-tree/deep_worker",),
        reason="source root found",
        unavailable=None,
        remedy=None,
    )


@pytest.mark.parametrize(
    ("provider", "pkg", "capability"),
    [
        ("codex", "deep_worker.providers.codex_provider", "codex"),
        ("grok", "deep_worker.providers.grok_provider", "grok"),
    ],
)
def test_create_model_raises_capability_error_when_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch, provider: str, pkg: str, capability: str
) -> None:
    """When the provider resolution reports ``unavailable``, create_model
    MUST raise CapabilityError (not a sys.path hack, not a raw
    ModuleNotFoundError), naming the canonical capability and remedy."""
    from deep_worker_cli.config import create_model

    resolution = _unavailable_resolution(pkg, remedy=f"pip install deep-worker[{capability}]")

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ) as mock_resolve:
        with pytest.raises(CapabilityError) as exc_info:
            create_model(provider=provider, model_name="some-model")

    mock_resolve.assert_called_once_with(pkg)
    assert exc_info.value.capability == capability
    assert exc_info.value.remedy == f"pip install deep-worker[{capability}]"
    assert "pip install" in str(exc_info.value)


@pytest.mark.parametrize(
    ("provider", "pkg", "capability"),
    [
        ("codex", "deep_worker.providers.codex_provider", "codex"),
        ("grok", "deep_worker.providers.grok_provider", "grok"),
    ],
)
def test_create_model_uses_resolution_remedy_fallback(
    monkeypatch: pytest.MonkeyPatch, provider: str, pkg: str, capability: str
) -> None:
    """If the resolution has no explicit remedy, the ``unavailable`` text
    itself must be surfaced as the remedy (call sites use
    ``resolution.remedy or resolution.unavailable``)."""
    from deep_worker_cli.config import create_model

    resolution = _unavailable_resolution(pkg, remedy=None)

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ):
        with pytest.raises(CapabilityError) as exc_info:
            create_model(provider=provider, model_name="some-model")

    assert exc_info.value.capability == capability
    assert exc_info.value.remedy == "provider package not installed"


@pytest.mark.parametrize(
    ("provider", "pkg", "model_cls_path"),
    [
        (
            "codex",
            "deep_worker.providers.codex_provider",
            "deep_worker.providers.codex_provider.CodexChatModel",
        ),
        (
            "grok",
            "deep_worker.providers.grok_provider",
            "deep_worker.providers.grok_provider.GrokChatModel",
        ),
    ],
)
def test_create_model_proceeds_when_provider_available(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    pkg: str,
    model_cls_path: str,
) -> None:
    """When the resolution is available the branch MUST proceed to
    construct the model from the canonical ``deep_worker.providers.*``
    import (no sys.path hack)."""
    from deep_worker_cli.config import create_model

    resolution = _available_resolution(pkg)
    fake_model = MagicMock()

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ), patch(model_cls_path, return_value=fake_model) as mock_model:
        result = create_model(provider=provider, model_name="some-model")

    mock_model.assert_called_once()
    assert result is fake_model
