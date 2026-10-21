"""``do_fork`` must clean up a partial copy when ``fork_agent_data`` fails.

The cleanup branch calls ``shutil.rmtree``.  ``shutil`` was not imported at
module scope in ``agent_command`` (the only ``import shutil`` was function-local,
inside a different function), so the branch raised ``NameError`` instead of
removing the half-written directory.  Nothing exercised it: it runs only when
``fork_agent_data`` raises, and no test drove that path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli import agent_command


def _make_agent(home: Path, name: str) -> Path:
    """Create a minimal source agent under ``home`` and return its dir."""
    data_dir = home / ".deep_worker" / "agents" / name
    data_dir.mkdir(parents=True)
    (data_dir / "session_metadata.json").write_text(
        json.dumps(
            {
                "status": "idle",
                "launch_config": {
                    "name": name,
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "working_dir": str(home),
                    "storage_version": 5,
                },
            }
        )
    )
    return data_dir


def _fork_args(src: str, dst: str) -> argparse.Namespace:
    """A Namespace shaped like a real parse, seeded from the parser defaults.

    ``do_fork`` funnels through ``merge_restart_config``, which reads every
    option the parser defines; building from ``_PARSER_DEFAULTS`` keeps this
    in step with the parser instead of hard-coding a subset.
    """
    ns = argparse.Namespace(**dict(agent_command._PARSER_DEFAULTS))
    ns.name = dst
    ns.fork = src
    ns.fork_from = None
    ns.clean = False
    return ns


@pytest.mark.asyncio
async def test_fork_failure_removes_the_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing fork removes the partially-copied destination directory.

    The oracle is the destination directory: ``fork_agent_data`` is replaced by
    a stub that creates it (the partial copy) and then raises, so the directory
    exists when the cleanup branch is entered.  If that branch raises instead of
    running, the directory survives and ``do_fork`` never returns its exit code.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    _make_agent(tmp_path, "src_agent")
    dst_dir = tmp_path / ".deep_worker" / "agents" / "dst_agent"

    def _failing_fork(old: str, new: str, message_id: object = None) -> None:
        # Leave a partial copy behind, exactly as a mid-copy failure would.
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "half_written.json").write_text("{}")
        raise RuntimeError("copy aborted midway")

    monkeypatch.setattr(
        "deep_worker_cli.fork_agent.fork_agent_data", _failing_fork
    )

    rc = await agent_command.do_fork(_fork_args("src_agent", "dst_agent"))

    assert rc == 1
    assert not dst_dir.exists(), (
        f"partial copy left behind at {dst_dir}; the cleanup branch did not run"
    )


@pytest.mark.asyncio
async def test_agent_command_imports_shutil_at_module_scope() -> None:
    """``shutil`` resolves as a module global of ``agent_command``.

    The cleanup branch above references it as a global.  A function-local
    ``import shutil`` elsewhere in the module does not bind it here.
    """
    assert hasattr(agent_command, "shutil"), (
        "agent_command.shutil is unbound; shutil.rmtree in do_fork would "
        "raise NameError"
    )
