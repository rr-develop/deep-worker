"""TDD for the --import-claude / --import-only CLI handler
(DESIGN_IMPORT_CLAUDE_SESSION.md §5).

We test the async handler ``handle_import_claude`` in isolation: it resolves
the transcript, parses it, writes the history into the agent's SQLite DB, and
(claude-code only) optionally seeds the session_uuid hint. The
daemonize/launch decision is driven by ``--import-only`` but is exercised at
the main() level; here we assert the import side-effects and return value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deep_worker_cli.cli.agent_command import handle_import_claude


def _args(**kw: object) -> argparse.Namespace:
    base = {
        "name": "imp_agent",
        "import_claude": None,
        "import_only": False,
        "provider": "deepseek",
        "model": None,
        "dir": None,
        "clean": False,
        "restart": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _msgs() -> list[BaseMessage]:
    return [HumanMessage(content="hi"), AIMessage(content="hello")]


async def test_resolves_parses_and_imports(tmp_path: Path) -> None:
    f = tmp_path / "sess.jsonl"
    f.write_text("{}\n")
    args = _args(import_claude=str(f))

    imported: dict[str, object] = {}

    async def fake_import(agent: str, messages: list[BaseMessage], *, replace: bool) -> int:
        imported["agent"] = agent
        imported["n"] = len(messages)
        imported["replace"] = replace
        return len(messages)

    with patch(
        "deep_worker_cli.cli.agent_command.parse_claude_transcript",
        return_value=_msgs(),
    ), patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        return_value=f,
    ), patch(
        "deep_worker_cli.sessions.import_history", side_effect=fake_import
    ):
        rc = await handle_import_claude(args)

    assert rc == 0
    assert imported["agent"] == "imp_agent"
    assert imported["n"] == 2
    assert imported["replace"] is False


async def test_clean_flag_passes_replace_true(tmp_path: Path) -> None:
    f = tmp_path / "sess.jsonl"
    f.write_text("{}\n")
    args = _args(import_claude=str(f), clean=True)

    captured: dict[str, object] = {}

    async def fake_import(agent: str, messages: list[BaseMessage], *, replace: bool) -> int:
        captured["replace"] = replace
        return len(messages)

    with patch(
        "deep_worker_cli.cli.agent_command.parse_claude_transcript",
        return_value=_msgs(),
    ), patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        return_value=f,
    ), patch(
        "deep_worker_cli.sessions.import_history", side_effect=fake_import
    ):
        rc = await handle_import_claude(args)

    assert rc == 0
    assert captured["replace"] is True


async def test_not_found_returns_error(tmp_path: Path) -> None:
    args = _args(import_claude="nonexistent-uuid")
    with patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        side_effect=FileNotFoundError("nope"),
    ):
        rc = await handle_import_claude(args)
    assert rc != 0


async def test_empty_parse_returns_error(tmp_path: Path) -> None:
    f = tmp_path / "sess.jsonl"
    f.write_text("{}\n")
    args = _args(import_claude=str(f))
    with patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        return_value=f,
    ), patch(
        "deep_worker_cli.cli.agent_command.parse_claude_transcript",
        return_value=[],
    ):
        rc = await handle_import_claude(args)
    assert rc != 0


async def test_claude_code_does_not_seed_session_uuid_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider=claude-code + a real <uuid>.jsonl in the project dir → the
    session_uuid hint must NOT be written (observed 2026-05-24): resuming an
    external ordinary-claude session loses the deepworker MCP tools."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    from deep_worker.providers.claude_code_provider.session_synthesis import session_project_dir

    uuid = "55555555-5555-5555-5555-555555555555"
    cwd = "/path/to/project"
    proj = session_project_dir(cwd)
    proj.mkdir(parents=True, exist_ok=True)
    f = proj / f"{uuid}.jsonl"
    f.write_text("{}\n")
    args = _args(import_claude=uuid, provider="claude-code", dir=Path(cwd))

    save_mock = MagicMock()

    async def fake_import(agent: str, messages: list[BaseMessage], *, replace: bool) -> int:
        return len(messages)

    with patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        return_value=f,
    ), patch(
        "deep_worker_cli.cli.agent_command.parse_claude_transcript",
        return_value=_msgs(),
    ), patch(
        "deep_worker_cli.sessions.import_history", side_effect=fake_import
    ), patch(
        "deep_worker.providers.claude_code_provider.session_state.save_session_state",
        save_mock,
    ):
        rc = await handle_import_claude(args)

    # REGRESSION GUARD (observed 2026-05-24): import must NEVER seed a
    # session_uuid hint pointing at the imported ordinary-claude transcript.
    # Doing so makes the claude-code provider RESUME that external session,
    # whose restored tool/permission config has no deepworker MCP tools — the
    # agent then ends up with ~1 tool instead of its full set. With no hint,
    # claude-code starts a fresh session (full MCP tools) and carries the
    # imported memory via the normal send loop / synthesis of OUR transcript.
    assert rc == 0
    save_mock.assert_not_called()


async def test_non_claude_provider_does_not_seed_hint(tmp_path: Path) -> None:
    uuid = "66666666-6666-6666-6666-666666666666"
    f = tmp_path / f"{uuid}.jsonl"
    f.write_text("{}\n")
    args = _args(import_claude=uuid, provider="deepseek", dir=Path("/path/to/project"))

    save_mock = MagicMock()

    async def fake_import(agent: str, messages: list[BaseMessage], *, replace: bool) -> int:
        return len(messages)

    with patch(
        "deep_worker_cli.cli.agent_command.resolve_transcript_path",
        return_value=f,
    ), patch(
        "deep_worker_cli.cli.agent_command.parse_claude_transcript",
        return_value=_msgs(),
    ), patch(
        "deep_worker_cli.sessions.import_history", side_effect=fake_import
    ), patch(
        "deep_worker.providers.claude_code_provider.session_state.save_session_state",
        save_mock,
    ):
        rc = await handle_import_claude(args)

    assert rc == 0
    save_mock.assert_not_called()


class TestMainGuards:
    """main()-level combination guards (no daemonize on these paths)."""

    def _run_main(self, argv: list[str]) -> int | None:
        import sys
        from deep_worker_cli.cli.agent_command import main
        with patch.object(sys, "argv", ["deep-worker"] + argv):
            try:
                return main()
            except SystemExit as e:  # argparse may exit
                return int(e.code) if e.code is not None else 0

    def test_import_only_without_import_claude_rejected(self) -> None:
        assert self._run_main(["foo", "--import-only"]) == 2

    def test_import_claude_with_restart_rejected(self) -> None:
        assert self._run_main(["foo", "--import-claude", "x", "--restart"]) == 2

    def test_import_missing_file_returns_error(self) -> None:
        rc = self._run_main(
            ["foo", "--import-claude", "/nonexistent/none.jsonl", "--import-only"]
        )
        assert rc == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
