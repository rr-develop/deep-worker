"""Unit tests for the --remove-before extension covering summaries files.

These tests exercise only the summaries-pruning half of remove_before_command.
The checkpoint-DB-pruning half is covered by the existing
tests/integration_ai/test_thread_management_cli.py suite and is not touched
here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home
def _mk(path: Path, timestamps: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"## Summarized at {ts}\n\nbody\n\n" for ts in timestamps]
    path.write_text("".join(parts))


@pytest.mark.asyncio
class TestRemoveBeforeSummariesExtension:
    async def test_dry_run_does_not_write(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _mk(f, ["2026-01-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00"])
        original = f.read_text()
        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command("2026-03-01", dry_run=True)
        assert rc == 0
        assert f.read_text() == original
        out = capsys.readouterr().out.lower()
        assert "summ" in out  # the command mentions summaries in its preview

    async def test_file_deleted_when_all_older(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _mk(f, ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"])
        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command("2026-03-01", dry_run=False)
        assert rc == 0
        assert not f.exists()

    async def test_file_rewritten_when_partial(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _mk(
            f,
            [
                "2026-01-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ],
        )
        from deep_worker_cli.cli.agent_command import remove_before_command

        rc = await remove_before_command("2026-03-01", dry_run=False)
        assert rc == 0
        assert f.exists()
        body = f.read_text()
        assert "2026-01-01" not in body
        assert "2026-05-01" in body
