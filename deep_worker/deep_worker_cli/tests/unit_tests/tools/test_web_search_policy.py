"""Unit tests for the tri-state universal web-search policy (DESIGN §3.3).

Covers ``resolve_web_search_tool`` in ``deep_worker_cli.tools``:

- ``disabled`` never touches the claude provider or the SDK (no import);
- ``auto`` returns the tool when the SDK fork is available, and skips with
  a single log otherwise (the agent still starts);
- ``enabled`` returns the tool when available and raises ``CapabilityError``
  with a remedy otherwise;
- SDK-error classification follows the STRICT order (rev 4): a bare
  ``ImportError`` for the root ``claude_agent_sdk`` module is ``sdk_missing``;
  a missing fork-private ``claude_agent_sdk._mcp_context`` is
  ``sdk_incompatible``; anything else re-raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("claude_agent_sdk")  # optional private fork; skip when absent

from deep_worker_cli.config import WebSearchPolicy
from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.tools import (
    WebSearchAvailability,
    _sdk_fork_status,
    resolve_web_search_tool,
)


def _fake_import_error(name: str | None) -> ImportError:
    exc = ImportError(f"cannot import name {name!r}")
    exc.name = name
    return exc


# ---------------------------------------------------------------------------
# disabled: never resolve/import
# ---------------------------------------------------------------------------


def test_disabled_never_imports_sdk_or_provider() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status",
        side_effect=AssertionError("disabled MUST NOT probe the SDK"),
    ):
        avail = resolve_web_search_tool(WebSearchPolicy.DISABLED)
    assert avail.tool is None
    assert avail.reason == "disabled"


# ---------------------------------------------------------------------------
# sdk_missing classification
# ---------------------------------------------------------------------------


def test_auto_sdk_missing_skips_tool() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status",
        return_value=("sdk_missing", "install the SDK fork"),
    ):
        avail = resolve_web_search_tool(WebSearchPolicy.AUTO)
    assert avail.tool is None
    assert avail.reason == "sdk_missing"
    assert avail.remedy is not None


def test_enabled_sdk_missing_raises_capability_error() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status",
        return_value=("sdk_missing", "install the SDK fork"),
    ):
        with pytest.raises(CapabilityError) as exc_info:
            resolve_web_search_tool(WebSearchPolicy.ENABLED)
    assert exc_info.value.capability == "web_search"
    assert "SDK fork" in (exc_info.value.remedy or "")


# ---------------------------------------------------------------------------
# sdk_incompatible classification
# ---------------------------------------------------------------------------


def test_auto_sdk_incompatible_skips_tool() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status",
        return_value=("sdk_incompatible", "official SDK, not the fork"),
    ):
        avail = resolve_web_search_tool(WebSearchPolicy.AUTO)
    assert avail.tool is None
    assert avail.reason == "sdk_incompatible"


def test_enabled_sdk_incompatible_raises_capability_error() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status",
        return_value=("sdk_incompatible", "official SDK, not the fork"),
    ):
        with pytest.raises(CapabilityError) as exc_info:
            resolve_web_search_tool(WebSearchPolicy.ENABLED)
    assert exc_info.value.capability == "web_search"
    assert "fork" in (exc_info.value.remedy or "")


# ---------------------------------------------------------------------------
# available path
# ---------------------------------------------------------------------------


def test_auto_available_builds_tool() -> None:
    fake_tool = MagicMock()
    with patch(
        "deep_worker_cli.tools._sdk_fork_status", return_value=("ok", None),
    ), patch(
        "deep_worker.providers.claude_code_provider.web_search.WebSearchDelegationTool",
        return_value=fake_tool,
    ):
        avail = resolve_web_search_tool(WebSearchPolicy.AUTO)
    assert avail.tool is fake_tool
    assert avail.reason == "available"
    assert avail.remedy is None


def test_enabled_available_builds_tool() -> None:
    fake_tool = MagicMock()
    with patch(
        "deep_worker_cli.tools._sdk_fork_status", return_value=("ok", None),
    ), patch(
        "deep_worker.providers.claude_code_provider.web_search.WebSearchDelegationTool",
        return_value=fake_tool,
    ):
        avail = resolve_web_search_tool(WebSearchPolicy.ENABLED)
    assert avail.tool is fake_tool
    assert avail.reason == "available"


# ---------------------------------------------------------------------------
# string coercion + unknown values
# ---------------------------------------------------------------------------


def test_string_policy_coercion() -> None:
    with patch(
        "deep_worker_cli.tools._sdk_fork_status", return_value=("ok", None),
    ), patch(
        "deep_worker.providers.claude_code_provider.web_search.WebSearchDelegationTool",
        return_value=MagicMock(),
    ):
        avail = resolve_web_search_tool("enabled")
    assert avail.reason == "available"


def test_unknown_policy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        resolve_web_search_tool("bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# strict error classification in _sdk_fork_status (rev 4)
# ---------------------------------------------------------------------------


def test_sdk_missing_classification_strict_root_name() -> None:
    with patch(
        "builtins.__import__",
        side_effect=_fake_import_error("claude_agent_sdk"),
    ):
        status, remedy = _sdk_fork_status()
    assert status == "sdk_missing"
    assert remedy is not None


def test_sdk_incompatible_when_fork_module_missing() -> None:
    # Root import succeeds, fork-private module fails -> incompatible.
    root_mod = MagicMock()
    root_mod.__name__ = "claude_agent_sdk"
    root_mod.__path__ = ["/fake/claude_agent_sdk"]

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "claude_agent_sdk._mcp_context":
            raise _fake_import_error("claude_agent_sdk._mcp_context")
        if name == "claude_agent_sdk":
            return root_mod
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        status, remedy = _sdk_fork_status()
    assert status == "sdk_incompatible"
    assert remedy is not None


def test_sdk_internal_defect_reraises() -> None:
    # A non-ImportError (e.g. AttributeError inside the module) must re-raise.
    with patch(
        "builtins.__import__",
        side_effect=AttributeError("boom"),
    ):
        with pytest.raises(AttributeError):
            _sdk_fork_status()


def test_web_search_availability_shape() -> None:
    avail = WebSearchAvailability(tool=None, reason="disabled", remedy=None)
    assert avail.tool is None
    assert avail.reason == "disabled"
