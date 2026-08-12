"""Unit tests for the future-cutoff safety guard on `--remove-before`.

Design: ``deep_worker/docs/DESIGN_REMOVE_BEFORE_FUTURE_CUTOFF_GUARD.md``.

The guard blocks the CLI command when the parsed cutoff resolves to a
future timestamp (``cutoff > now``), since that is semantically "delete
everything" and is almost always a bug. The opt-out is the explicit
``--force`` flag.

These tests are pure-sync over ``monkeypatched HOME`` in ``tmp_path`` —
no real ``~/.deepagents`` is touched, no subprocess, no RPyC, no LLM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home
@pytest.mark.asyncio
class TestFutureCutoffGuard:
    async def test_future_cutoff_without_force_is_blocked(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        """`remove_before_command("2030-01-01", force=False)` must return 2."""
        monkeypatch.setenv("HOME", str(tmp_path))
        # Seed a fake summary file so we can assert it was NOT touched.
        (deep_worker_home(tmp_path) / "a" / "summaries").mkdir(parents=True)
        survivor = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        survivor.write_text(
            "## Summarized at 2026-01-01T00:00:00+00:00\n\nbody\n\n"
        )
        original = survivor.read_text()

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command(
            "2030-01-01", dry_run=False, force=False
        )
        assert rc == 2
        out = capsys.readouterr().out.lower()
        assert "future" in out
        assert "--force" in out
        # Survivor file untouched.
        assert survivor.read_text() == original

    async def test_future_cutoff_without_force_blocked_even_in_dry_run(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        """Dry-run is NOT an opt-out; future cutoff must still abort."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command(
            "2030-01-01", dry_run=True, force=False
        )
        assert rc == 2
        out = capsys.readouterr().out.lower()
        assert "future" in out
        assert "--force" in out

    async def test_future_cutoff_with_force_proceeds(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """`force=True` bypasses the guard (exit 0 on empty home)."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command(
            "2030-01-01", dry_run=True, force=True
        )
        assert rc == 0

    async def test_past_cutoff_no_guard(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Past absolute cutoff proceeds without `--force`."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command(
            "2020-01-01", dry_run=True, force=False
        )
        assert rc == 0

    async def test_relative_past_cutoff_no_guard(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Relative past cutoff (`30d`) proceeds without `--force`."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command("30d", dry_run=True, force=False)
        assert rc == 0

    async def test_future_cutoff_error_message_mentions_parsed_timestamp(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        """Error surfaces the parsed cutoff so the user can verify a misparse."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command(
            "2030-01-01", dry_run=False, force=False
        )
        assert rc == 2
        out = capsys.readouterr().out
        assert "2030-01-01" in out  # parsed cutoff echoed back


@pytest.mark.asyncio
class TestForceFlagPlumbing:
    """Verify the CLI argparse surfaces `--force`."""

    async def test_argparse_accepts_force_flag(self) -> None:
        """`create_parser().parse_args(["--remove-before","X","--force"])`
        must populate a `force` attribute (True)."""
        from deep_worker_cli.cli.agent_command import create_parser

        args = create_parser().parse_args(
            ["--remove-before", "2030-01-01", "--force"]
        )
        assert getattr(args, "force", None) is True

    async def test_argparse_force_defaults_to_false(self) -> None:
        """Without `--force`, `args.force` is False."""
        from deep_worker_cli.cli.agent_command import create_parser

        args = create_parser().parse_args(["--remove-before", "2030-01-01"])
        assert getattr(args, "force", None) is False
