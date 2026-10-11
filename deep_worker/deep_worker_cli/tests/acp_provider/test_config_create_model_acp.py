"""Unit tests for the ``claude-code`` branch of ``config.create_model``.

Covers the owner-gate Q2/Q5 auto-approve plumbing: the branch must forward
the explicit ``auto_approve`` operator flag into
``ClaudeCodeAcpChatModel(auto_approve=...)`` — fail-closed (False) by
default — and must resolve the vendored provider through
``ensure_provider_importable`` (CapabilityError on absence), exactly like
the other custom providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.provider_paths import ProviderOrigin, ProviderResolution


def _available_resolution(pkg: str) -> ProviderResolution:
    return ProviderResolution(
        pkg=pkg,
        mode="source",
        origin=ProviderOrigin.SOURCE,
        roots=("/path/to/source/deep_worker",),
        reason="source root found",
        unavailable=None,
        remedy=None,
    )


def _unavailable_resolution(pkg: str) -> ProviderResolution:
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="not found",
        unavailable="provider package not installed",
        remedy="pip install deep-worker[claude-code]",
    )


class TestCreateModelClaudeCodeAcp:
    def test_forwards_auto_approve_flag(self) -> None:
        from deep_worker_cli.config import create_model

        model_cls = MagicMock(return_value=MagicMock())
        with patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_available_resolution("deep_worker.providers.claude_code_provider"),
        ), patch(
            "deep_worker.providers.claude_code_provider.ClaudeCodeAcpChatModel",
            model_cls,
        ):
            create_model(provider="claude-code", model_name="sonnet", auto_approve=True)

        assert model_cls.call_count == 1
        _, kwargs = model_cls.call_args
        assert kwargs["auto_approve"] is True
        assert kwargs["model"] == "sonnet"
        assert kwargs["max_tokens"] == 20_000

    def test_default_is_fail_closed(self) -> None:
        from deep_worker_cli.config import create_model

        model_cls = MagicMock(return_value=MagicMock())
        with patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_available_resolution("deep_worker.providers.claude_code_provider"),
        ), patch(
            "deep_worker.providers.claude_code_provider.ClaudeCodeAcpChatModel",
            model_cls,
        ):
            create_model(provider="claude-code", model_name=None)

        _, kwargs = model_cls.call_args
        assert kwargs["auto_approve"] is False
        assert kwargs["model"] is None  # Claude Code picks its own default

    def test_model_name_none_allowed(self) -> None:
        from deep_worker_cli.config import create_model

        model_cls = MagicMock(return_value=MagicMock())
        with patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_available_resolution("deep_worker.providers.claude_code_provider"),
        ), patch(
            "deep_worker.providers.claude_code_provider.ClaudeCodeAcpChatModel",
            model_cls,
        ):
            # model_name=None must NOT trigger provider auto-detection.
            create_model(provider="claude-code", model_name=None)

        _, kwargs = model_cls.call_args
        assert kwargs["model"] is None

    def test_unavailable_provider_raises_capability_error(self) -> None:
        from deep_worker_cli.config import create_model

        with patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_unavailable_resolution(
                "deep_worker.providers.claude_code_provider"
            ),
        ):
            with pytest.raises(CapabilityError) as exc_info:
                create_model(provider="claude-code", model_name="sonnet")

        assert exc_info.value.capability == "claude-code"
        assert "pip install" in str(exc_info.value)
