"""Tests for the provider capability guards in ``TaskManager._initialize_manager`` (B2).

Capability-guard contract: the codex / grok native-tool passthrough
wiring in ``task_manager._initialize_manager`` must resolve the vendored
provider through ``provider_paths.ensure_provider_importable``. When the
provider is unavailable the manager must raise ``CapabilityError`` BEFORE
touching ``NATIVE_TOOLS`` / ``build_passthrough_tools`` — a raw
``ModuleNotFoundError`` must never escape into the manager startup path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.provider_paths import ProviderOrigin, ProviderResolution

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_worker_cli.task_manager.task_manager import TaskManager  # noqa: E402


def _stub_config(provider: str) -> MagicMock:
    cfg = MagicMock()
    cfg.provider = provider
    cfg.model = None
    cfg.working_dir = Path("/tmp")
    cfg.web_search_policy = "auto"
    return cfg


def _make_tm(provider: str) -> TaskManager:
    return TaskManager(
        agent_name="t1",
        config=_stub_config(provider),
        session_manager=MagicMock(),
    )


def _unavailable_resolution(pkg: str) -> ProviderResolution:
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="not found",
        unavailable="provider package not installed",
        remedy=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "pkg", "capability"),
    [
        ("codex", "deep_worker.providers.codex_provider", "codex"),
        ("grok", "deep_worker.providers.grok_provider", "grok"),
    ],
)
async def test_initialize_manager_raises_capability_error_when_provider_unavailable(
    provider: str, pkg: str, capability: str
) -> None:
    """Manager startup for an unavailable codex/grok provider MUST raise
    CapabilityError, and must NOT reach ``create_cli_agent``."""
    tm = _make_tm(provider)
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=_unavailable_resolution(pkg),
    ), patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        with pytest.raises(CapabilityError) as exc_info:
            await tm._initialize_manager()

    assert exc_info.value.capability == capability
    mock_create.assert_not_called(), (
        "create_cli_agent must NOT be called when the provider is unavailable"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "pkg"),
    [
        ("codex", "deep_worker.providers.codex_provider"),
        ("grok", "deep_worker.providers.grok_provider"),
    ],
)
async def test_initialize_manager_proceeds_when_provider_available(
    provider: str, pkg: str
) -> None:
    """When the resolution is available the manager MUST proceed to
    ``create_cli_agent`` with the passthrough tools bound."""
    tm = _make_tm(provider)
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    resolution = ProviderResolution(
        pkg=pkg,
        mode="source",
        origin=ProviderOrigin.SOURCE,
        roots=("/tmp/example-tree/deep_worker",),
        reason="source root found",
        unavailable=None,
        remedy=None,
    )

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ), patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    assert mock_create.called, "create_cli_agent must be invoked"
