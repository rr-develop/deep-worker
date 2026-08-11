"""Guards for the optional claude_code_provider import (agent_command.py).

``claude_code_provider`` is a monorepo-only package (lives at
<repo-root>/claude_code_provider/) and is NOT vendored into the deep-worker
wheel. ``deep_worker_cli/cli/agent_command.py`` is imported by the
``deep-worker`` entry point (via cli/router.py), so a hard top-level
``from deep_worker.providers.claude_code_provider...`` import would break EVERY ``deep-worker
--help`` invocation on an installed wheel.

The import goes through ``_load_claude_session_imports()`` which:

- returns ``(None, None)`` when the provider is genuinely absent
  (``ModuleNotFoundError`` on ``claude_code_provider`` itself);
- re-raises ANY other import defect of an installed/source provider
  (including a missing/renamed ``claude_code_provider.session_import``
  submodule) so it is not masked as "package absent";
- ``handle_import_claude`` (``--import-claude``) reports a clear error
  (rc=2) when the provider is missing.

Behavioral coverage below; the ordinary absent-provider entry-point path
(``deep-worker --help`` → exit 0) is covered by the clean-wheel smoke test
in tests/integration_tests/test_pypi_build.py.
"""

from __future__ import annotations

import argparse
import builtins
import re
from pathlib import Path
from unittest.mock import patch

import pytest

import deep_worker_cli.cli.agent_command as agent_command

AGENT_COMMAND = Path(agent_command.__file__)


# ---------------------------------------------------------------------------
# Static guards (textual)
# ---------------------------------------------------------------------------


def test_claude_code_provider_import_is_optional() -> None:
    src = AGENT_COMMAND.read_text(encoding="utf-8")
    assert "try:" in src, "claude_code_provider import must be in try/except"
    assert "except ModuleNotFoundError as exc:" in src, (
        "claude_code_provider import must catch ModuleNotFoundError"
    )
    assert re.search(
        r"from deep_worker.providers.claude_code_provider\.session_import import", src
    ), "missing claude_code_provider.session_import import"
    assert "exc.name" in src, (
        "except clause must validate exc.name (narrow missing-provider case)"
    )
    assert "parse_claude_transcript, resolve_transcript_path = " in src, (
        "module globals must be assigned from the guarded loader"
    )
    assert "parse_claude_transcript is None" in src, (
        "--import-claude must guard against missing claude_code_provider"
    )


# ---------------------------------------------------------------------------
# Behavioral guards
# ---------------------------------------------------------------------------


def _patch_provider_import(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make ``from deep_worker.providers.claude_code_provider...`` raise ``exc`` during import."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "deep_worker.providers.claude_code_provider" or name.startswith(
            "deep_worker.providers.claude_code_provider."
        ):
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_loader_returns_none_when_provider_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider genuinely absent → loader returns (None, None)."""
    _patch_provider_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'claude_code_provider'", name="claude_code_provider"
        ),
    )
    loaded = agent_command._load_claude_session_imports()
    assert loaded == (None, None)


def test_loader_reraises_internal_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider present but its import raises an internal ImportError → NOT swallowed."""
    _patch_provider_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'some_internal_dep'", name="some_internal_dep"
        ),
    )
    with pytest.raises(ModuleNotFoundError, match="some_internal_dep"):
        agent_command._load_claude_session_imports()


def test_loader_reraises_missing_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider present but its session_import submodule is missing → NOT swallowed.

    Python raises ``ModuleNotFoundError(name="claude_code_provider.session_import")``
    when the package exists but the required submodule is absent/renamed. That is
    an internal provider defect and must be surfaced, not converted to
    "provider absent".
    """
    _patch_provider_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'claude_code_provider.session_import'",
            name="claude_code_provider.session_import",
        ),
    )
    with pytest.raises(ModuleNotFoundError, match="claude_code_provider.session_import"):
        agent_command._load_claude_session_imports()


def test_loader_reraises_non_missing_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-ModuleNotFoundError ImportError (e.g. broken installed provider) → re-raised."""
    _patch_provider_import(monkeypatch, ImportError("broken provider internals"))
    with pytest.raises(ImportError, match="broken provider internals"):
        agent_command._load_claude_session_imports()


@pytest.mark.asyncio
async def test_import_claude_reports_missing_provider(capsys: pytest.CaptureFixture[str]) -> None:
    """With the provider absent, --import-claude returns rc=2 with a clear message."""
    with patch.object(agent_command, "parse_claude_transcript", None), patch.object(
        agent_command, "resolve_transcript_path", None
    ):
        args = argparse.Namespace(
            import_claude="some-uuid",
            dir=None,
            name="test-agent",
            clean=False,
        )
        rc = await agent_command.handle_import_claude(args)

    assert rc == 2
    captured = capsys.readouterr()
    assert "claude_code_provider" in captured.err
