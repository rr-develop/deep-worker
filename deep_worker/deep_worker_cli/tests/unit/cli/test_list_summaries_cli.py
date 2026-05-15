"""Unit tests for `deep-worker --list-summaries`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _mk(path: Path, timestamps: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"## Summarized at {ts}\n\nbody\n\n" for ts in timestamps]
    path.write_text("".join(parts))


@pytest.mark.asyncio
class TestListSummariesCommand:
    async def test_empty_home_prints_no_summaries_message(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        from deep_worker_cli.cli.agent_command import list_summaries_command

        rc = await list_summaries_command(agent_name=None, fmt="text")
        assert rc == 0
        out = capsys.readouterr().out.lower()
        assert "no summaries" in out or "0 summaries" in out

    async def test_lists_multiple_agents_in_text(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        _mk(tmp_path / Path("~/.deep_worker").name / "a1" / "summaries" / "a1.md", ["2026-04-01T00:00:00+00:00"])
        _mk(tmp_path / Path("~/.deep_worker").name / "b2" / "summaries" / "b2.md", ["2026-04-02T00:00:00+00:00"])

        from deep_worker_cli.cli.agent_command import list_summaries_command

        rc = await list_summaries_command(agent_name=None, fmt="text")
        assert rc == 0
        out = capsys.readouterr().out
        assert "a1" in out
        assert "b2" in out
        assert "2026-04-01" in out
        assert "2026-04-02" in out

    async def test_filter_by_agent(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        _mk(tmp_path / Path("~/.deep_worker").name / "a1" / "summaries" / "a1.md", ["2026-04-01T00:00:00+00:00"])
        _mk(tmp_path / Path("~/.deep_worker").name / "b2" / "summaries" / "b2.md", ["2026-04-02T00:00:00+00:00"])

        from deep_worker_cli.cli.agent_command import list_summaries_command

        rc = await list_summaries_command(agent_name="a1", fmt="text")
        assert rc == 0
        out = capsys.readouterr().out
        assert "a1" in out
        assert "b2" not in out

    async def test_json_format_returns_valid_array(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        _mk(
            tmp_path / Path("~/.deep_worker").name / "a1" / "summaries" / "a1.md",
            ["2026-04-01T00:00:00+00:00", "2026-04-02T00:00:00+00:00"],
        )
        from deep_worker_cli.cli.agent_command import list_summaries_command

        rc = await list_summaries_command(agent_name=None, fmt="json")
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 1
        row = data[0]
        assert row["agent"] == "a1"
        assert row["thread"] == "a1"
        assert row["events"] == 2
        assert row["latest_ts"] == "2026-04-02T00:00:00+00:00"
        assert "path" in row
        assert row["path"].endswith("/a1/summaries/a1.md")
        assert row["size"] > 0
