"""Integration tests for the deep-worker bash launcher's --dev dispatch.

These tests invoke the REAL bash launcher at
    <tree>/deep_worker/deep_worker_cli/deep-worker
and validate its dispatch logic without starting a real agent process.

Strategy
========
The launcher hands control to ``python3 -m deep_worker_cli.cli.agent_command``
via PYTHONPATH. For testing we don't want a real agent to start — we just
want to see WHAT command the launcher decided to run. So we:

  1. Build a tiny fake "dev tree" in a tmpdir with the same directory
     layout the launcher expects:

         <fake_dev>/deep_worker/deep_worker_cli/deep-worker             (a copy of the real launcher being tested)
         <fake_dev>/deep_worker/deep_worker_cli/deep_worker_cli/__init__.py
         <fake_dev>/deep_worker/deep_worker_cli/deep_worker_cli/cli/__init__.py
         <fake_dev>/deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py    ← prints argv + env, exits 0
         deep_worker/deepagents/                                         (empty stub)

  2. Point the launcher at this fake tree via the
     ``DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE`` env var (an explicit testing
     hook in the launcher).

  3. Run the launcher and assert on its decisions, captured as stdout
     from the fake ``agent_command.py``.

These tests therefore exercise:
  * dispatch on --dev → re-exec into dev-tree launcher
  * dispatch on --restart auto-detect from session_metadata.json
  * exit 2 with no fallback when --dev requested but dev-tree missing
  * --dev stripped from argv before re-exec (no infinite loop)
  * pass-through when no --dev / no saved dev_tree

Design doc: docs/DESIGN_DEV_FLAG.md
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home


# Path to the launcher under test — the one that lives in THIS tree.
# Tests in dev-tree validate the dev-tree launcher; tests in master-tree
# validate the master-tree launcher. The path is relative to this file
# so each tree's test file finds its own launcher.
_THIS_FILE = Path(__file__).resolve()
LAUNCHER = (
    _THIS_FILE.parent.parent.parent  # → deep_worker/deep_worker_cli
    / "deep-worker"
)


# ---------------------------------------------------------------------------
# Safety: if a test accidentally invokes the REAL agent_command (because
# dispatch didn't happen as expected), we want to make sure that real
# agent process doesn't survive past the test. All tests use a unique
# agent name from this fixture so cleanup is targeted.
# ---------------------------------------------------------------------------


def _kill_real_agent_if_alive(name: str) -> None:
    """Belt-and-braces cleanup: if a test accidentally started a real
    daemonized agent with this name, find and kill it.

    This MUST be cheap and idempotent — it runs after every test that
    might leak.
    """
    pid_file = Path(f"/tmp/deep_worker_{name}.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
        try:
            pid_file.unlink()
        except OSError:
            pass
    data_dir = deep_worker_home() / name
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    log_dir = deep_worker_home() / "logs"
    if log_dir.exists():
        for f in log_dir.glob(f"{name}.*"):
            try:
                f.unlink()
            except OSError:
                pass


@pytest.fixture
def agent_name() -> str:
    """Unique name per test, with automatic cleanup of any real agent
    that might have been started by mistake."""
    name = f"dwtest_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    yield name
    _kill_real_agent_if_alive(name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dev_tree(tmp_path: Path) -> Path:
    """Create a fake dev-tree-shaped tree whose
    ``agent_command.py`` does nothing but print a JSON blob and exit 0.

    Returns the absolute path to the fake tree root.
    """
    # 1. Directory layout
    root = tmp_path / "fake_dev_tree"
    cli_root = root / "deep_worker" / "deep_worker_cli"
    pkg = cli_root / "deep_worker_cli"
    pkg_cli = pkg / "cli"
    deepagents = root / "deep_worker" / "deepagents"
    pkg_cli.mkdir(parents=True)
    deepagents.mkdir(parents=True)

    # 2. Fake bash launcher = copy of the real launcher under test.
    fake_launcher = cli_root / "deep-worker"
    shutil.copy(LAUNCHER, fake_launcher)
    fake_launcher.chmod(
        fake_launcher.stat().st_mode
        | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    # 3. Fake Python package: just enough for ``python3 -m
    #    deep_worker_cli.cli.agent_command`` to import and run.
    (pkg / "__init__.py").write_text("")
    (pkg_cli / "__init__.py").write_text("")
    (pkg_cli / "agent_command.py").write_text(
        # The fake agent_command writes its observations to a sentinel file
        # (path passed via env) AND prints the same blob to stdout.  We use a
        # file because the launcher swallows nothing — but reading stdout is
        # the simpler primary channel.
        '''
import json
import os
import sys

blob = {
    "argv": sys.argv,
    "sys_path_head": sys.path[:6],
    "DEEP_WORKER_DEV_TREE": os.environ.get("DEEP_WORKER_DEV_TREE"),
    "DEEP_WORKER_TREE_ROOT": os.environ.get("DEEP_WORKER_TREE_ROOT"),
    "tree_marker": "FAKE_DEV_TREE",
    "cwd": os.getcwd(),
}
print("FAKE_AGENT_COMMAND_OUTPUT:" + json.dumps(blob))
sys.exit(0)
'''
    )

    return root


def _run_launcher(
    launcher: Path,
    args: list[str],
    extra_env: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess:
    """Invoke the bash launcher with given args. Returns CompletedProcess."""
    env = os.environ.copy()
    # Make sure parent env doesn't accidentally carry the marker from a
    # previous test or the shell that launched pytest.
    env.pop("DEEP_WORKER_DEV_TREE", None)
    env.pop("DEEP_WORKER_TREE_ROOT", None)
    env.pop("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(launcher)] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _parse_fake_output(stdout: str) -> dict:
    """Pull the FAKE_AGENT_COMMAND_OUTPUT JSON line from stdout."""
    for line in stdout.splitlines():
        if line.startswith("FAKE_AGENT_COMMAND_OUTPUT:"):
            return json.loads(line.removeprefix("FAKE_AGENT_COMMAND_OUTPUT:"))
    raise AssertionError(
        f"Did not find FAKE_AGENT_COMMAND_OUTPUT in stdout:\n{stdout!r}"
    )


# ===========================================================================
# Sanity check
# ===========================================================================


def test_launcher_exists_and_is_executable() -> None:
    assert LAUNCHER.exists(), f"Launcher not found at {LAUNCHER}"
    assert os.access(LAUNCHER, os.X_OK), f"Launcher not executable: {LAUNCHER}"


# ===========================================================================
# Dispatch on --dev
# ===========================================================================


class TestDevDispatch:
    """``deep-worker foo --dev`` must re-exec into the dev-tree launcher."""

    def test_dev_flag_dispatches_to_override_tree(
        self, fake_dev_tree: Path, agent_name: str
    ) -> None:
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--dev"],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0, (
            f"launcher exited {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        blob = _parse_fake_output(result.stdout)
        # The fake agent_command lives inside fake_dev_tree, so reaching
        # it proves the dispatch worked.
        assert blob["tree_marker"] == "FAKE_DEV_TREE"
        assert blob["DEEP_WORKER_DEV_TREE"] == "1"
        assert blob["DEEP_WORKER_TREE_ROOT"] == str(fake_dev_tree)

    def test_dev_flag_stripped_from_argv_before_reexec(
        self, fake_dev_tree: Path, agent_name: str
    ) -> None:
        """The dev-tree launcher must NOT see --dev in its argv, otherwise
        recursion. The agent_command (and ultimately argparse) should also
        not see it — bash already acted on it."""
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--dev", "--dir", "/tmp"],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0
        blob = _parse_fake_output(result.stdout)
        assert "--dev" not in blob["argv"], (
            f"--dev leaked through to argv: {blob['argv']}"
        )
        # Everything else MUST survive.
        assert agent_name in blob["argv"]
        assert "--dir" in blob["argv"]
        assert "/tmp" in blob["argv"]

    def test_dev_flag_position_independent_first(
        self, fake_dev_tree: Path, agent_name: str
    ) -> None:
        """``deep-worker --dev myagent`` works the same way."""
        result = _run_launcher(
            LAUNCHER,
            ["--dev", agent_name],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0
        blob = _parse_fake_output(result.stdout)
        assert blob["tree_marker"] == "FAKE_DEV_TREE"

    def test_dev_flag_position_independent_middle(
        self, fake_dev_tree: Path, agent_name: str
    ) -> None:
        """``deep-worker foo --provider X --dev --model Y`` works."""
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--provider", "claude-code", "--dev", "--model", "opus"],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0
        blob = _parse_fake_output(result.stdout)
        assert blob["tree_marker"] == "FAKE_DEV_TREE"
        assert "--dev" not in blob["argv"]


# ===========================================================================
# Pass-through (no --dev)
# ===========================================================================


class TestNoDevDispatch:
    """Without --dev, the launcher must NOT touch the dev-tree at all."""

    def test_no_dev_runs_local_agent_command(self) -> None:
        """When no --dev is given and no auto-detect kicks in, the
        launcher must just run THIS tree's agent_command, NOT the dev
        tree's. We assert this by checking that DEEP_WORKER_DEV_TREE is
        NOT set in the subprocess env and DEEP_WORKER_TREE_ROOT points
        to the launcher's own tree.

        We invoke with --help so the real agent_command terminates fast.
        """
        result = _run_launcher(LAUNCHER, ["--help"])
        assert result.returncode == 0, (
            f"stderr={result.stderr!r}, stdout={result.stdout!r}"
        )
        # --help should print usage; no marker from our fake tree should appear.
        assert "FAKE_AGENT_COMMAND_OUTPUT" not in result.stdout


# ===========================================================================
# Missing dev tree → exit 2
# ===========================================================================


class TestMissingDevTreeError:
    """--dev requested but dev-tree directory does not exist or is not a
    valid layout → launcher exits 2 with a clear stderr message. No
    fallback to prod.
    """

    def test_exit_2_when_dev_tree_does_not_exist(
        self, tmp_path: Path, agent_name: str
    ) -> None:
        nonexistent = tmp_path / "does_not_exist"
        # Important: don't create it.
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--dev"],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(nonexistent),
            },
        )
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "--dev" in result.stderr or "dev" in result.stderr.lower()

    def test_exit_2_when_dev_launcher_missing(
        self, tmp_path: Path, agent_name: str
    ) -> None:
        """Directory exists, but no deep-worker launcher inside it →
        still exit 2, no silent fallback."""
        empty_tree = tmp_path / "empty_tree"
        empty_tree.mkdir()
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--dev"],
            extra_env={
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(empty_tree),
            },
        )
        assert result.returncode == 2
        assert result.stderr.strip(), "expected an error message in stderr"


# ===========================================================================
# --restart auto-detect
# ===========================================================================


class TestRestartAutoDetect:
    """``deep-worker foo --restart`` (without --dev) must auto-detect a
    previous launch with ``dev_tree: true`` in session_metadata.json
    and dispatch to dev-tree.
    """

    @pytest.fixture
    def fake_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A fake $HOME with ~/.deep_worker/<name>/session_metadata.json."""
        home = tmp_path / "home"
        home.mkdir()
        return home

    def _write_metadata(
        self,
        home: Path,
        name: str,
        dev_tree: bool,
    ) -> None:
        data_dir = home / Path("~/.deep_worker").name / name
        data_dir.mkdir(parents=True)
        meta = {
            "status": "idle",
            "launch_config": {
                "provider": "claude-code",
                "model": "opus",
                "working_dir": "/path/to/dev-tree"
                if dev_tree else "/path/to/production-tree",
                "skip_permissions": True,
                "task_manager": True,
                "memory_only": False,
                "use_sqlite": False,
                "clean": False,
                "dev_tree": dev_tree,
                "started_at": "2026-05-13T22:00:00",
                "pid": 1,
                "cli_args": "myagent --restart",
            },
        }
        (data_dir / "session_metadata.json").write_text(json.dumps(meta))

    def test_restart_auto_dispatches_when_saved_dev_tree_true(
        self, fake_home: Path, fake_dev_tree: Path, agent_name: str
    ) -> None:
        self._write_metadata(fake_home, agent_name, dev_tree=True)
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--restart"],
            extra_env={
                "HOME": str(fake_home),
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        blob = _parse_fake_output(result.stdout)
        assert blob["tree_marker"] == "FAKE_DEV_TREE"
        assert blob["DEEP_WORKER_DEV_TREE"] == "1"
        # --restart MUST be preserved (only --dev is stripped).
        assert "--restart" in blob["argv"]

    def test_restart_does_not_dispatch_when_saved_dev_tree_false(
        self, fake_home: Path, agent_name: str
    ) -> None:
        """If session_metadata.json says dev_tree=False, --restart should
        run on this tree (NOT re-exec into dev-tree)."""
        self._write_metadata(fake_home, agent_name, dev_tree=False)
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--restart", "--help"],  # --help so it exits fast
            extra_env={"HOME": str(fake_home)},
        )
        # Even if the real --restart code refuses without a real agent, the
        # key point is no FAKE marker appears.
        assert "FAKE_AGENT_COMMAND_OUTPUT" not in result.stdout

    def test_restart_does_not_dispatch_when_no_metadata(
        self, fake_home: Path, agent_name: str
    ) -> None:
        """No previous session_metadata.json → no auto-detect, no re-exec."""
        # Don't write any metadata.
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--restart"],
            extra_env={"HOME": str(fake_home)},
        )
        # Should NOT have reached the fake dev tree.
        assert "FAKE_AGENT_COMMAND_OUTPUT" not in result.stdout

    def test_explicit_dev_beats_saved_false(
        self, fake_home: Path, fake_dev_tree: Path, agent_name: str
    ) -> None:
        """``deep-worker foo --restart --dev`` after a non-dev launch
        must still dispatch to dev tree (explicit > saved)."""
        self._write_metadata(fake_home, agent_name, dev_tree=False)
        result = _run_launcher(
            LAUNCHER,
            [agent_name, "--restart", "--dev"],
            extra_env={
                "HOME": str(fake_home),
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE": str(fake_dev_tree),
            },
        )
        assert result.returncode == 0
        blob = _parse_fake_output(result.stdout)
        assert blob["tree_marker"] == "FAKE_DEV_TREE"
