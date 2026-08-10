import argparse
import json
import stat
from pathlib import Path

from deep_worker_cli.cli.agent_command import create_parser, merge_restart_config
from deep_worker_cli.cli.mcp_command import _cmd_add, _cmd_remove


def test_agent_parser_accepts_repeatable_mcp_overrides() -> None:
    args = create_parser().parse_args(
        [
            "agent",
            "--mcp-config",
            "one.json",
            "--mcp-config",
            "two.json",
            "--mcp-server",
            'demo={"command":"demo"}',
        ]
    )

    assert args.mcp_config_paths == ["one.json", "two.json"]
    assert args.mcp_server_inline == ['demo={"command":"demo"}']


def test_restart_preserves_and_allows_mcp_override() -> None:
    parser = create_parser()
    defaults = parser.parse_args(["agent"])
    saved = {
        "mcp_config_paths": ["saved.json"],
        "mcp_server_inline": ['saved={"command":"saved"}'],
    }

    inherited = merge_restart_config(saved, defaults, explicit=set())
    assert inherited.mcp_config_paths == ["saved.json"]
    assert inherited.mcp_server_inline == ['saved={"command":"saved"}']

    override_args = parser.parse_args(
        ["agent", "--mcp-config", "new.json", "--mcp-server", 'new={"command":"new"}']
    )
    overridden = merge_restart_config(
        saved,
        override_args,
        explicit={"mcp_config_paths", "mcp_server_inline"},
    )
    assert overridden.mcp_config_paths == ["new.json"]
    assert overridden.mcp_server_inline == ['new={"command":"new"}']


def test_mcp_add_and_remove_write_private_project_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # project-scope resolves from the LAUNCH directory (cli/cwd.py), not from
    # the process cwd. A DEEP_WORKER_ORIGINAL_CWD inherited from the parent
    # would override the chdir above and divert the write into the real repo.
    monkeypatch.delenv("DEEP_WORKER_ORIGINAL_CWD", raising=False)
    add_args = argparse.Namespace(
        command="python3 server.py",
        url=None,
        env=["TOKEN=${env:TEST_TOKEN}"],
        cwd=None,
        auto_approve=True,
        disabled=False,
        scope="project",
        name="demo",
    )

    assert _cmd_add(add_args) == 0
    path = tmp_path / ".deepagents" / "mcp.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["demo"] == {
        "command": "python3",
        "args": ["server.py"],
        "env": {"TOKEN": "${env:TEST_TOKEN}"},
        "auto_approve": True,
    }

    assert _cmd_remove(argparse.Namespace(scope="project", name="demo")) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"] == {}
