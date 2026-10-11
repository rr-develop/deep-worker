"""Integration test for the wiring of the ``lsp`` CLI command — on the REAL binary.

Why this file exists
--------------------
The unit tests (`tests/unit_tests/lsp/test_lsp_command.py`) call
``setup_lsp_parser``/``handle_lsp_command`` DIRECTLY. They honestly passed,
yet the real ``deep-worker lsp status`` command failed with
``unrecognized arguments``: the group was registered in
``deep_worker_cli/main.py``, which is a DEAD entry point — the launcher
runs ``python3 -m deep_worker_cli.cli.agent_command``.

The classic gap "code is correct, wiring is not". It is only caught by
running the real binary, so the tests below are subprocess, no mocks.

Policy "integration tests must not hang": every run has a hard timeout,
and the process is guaranteed to be killed in ``finally``.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# tests/integration_tests/lsp/ → tests/integration_tests → tests → package
_PKG_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _PKG_ROOT / "deep-worker"

_CMD_BUDGET_SEC = 60.0


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    gradle_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run this tree's launcher with a hard time budget.

    No mocks: this is exactly the path by which the user runs the
    command (bash launcher → python3 -m …agent_command).
    """
    assert _LAUNCHER.is_file(), f"launcher not found: {_LAUNCHER}"

    env = {**os.environ}
    # Don't let the test accidentally go to the prod tree via auto-dispatch.
    env.pop("DEEP_WORKER_DEV_TREE", None)
    env.pop("DEEP_WORKER_TREE_ROOT", None)
    # An inherited ORIGINAL_CWD would make the `cwd=` check unreliable:
    # the launcher will set it itself from the actual cwd of the subprocess.
    env.pop("DEEP_WORKER_ORIGINAL_CWD", None)
    # ``lsp enable``/``disable`` touch $GRADLE_USER_HOME/init.d. The test must
    # work in its own sandbox, not in the developer's ~/.gradle, so the
    # variable is ALWAYS set — the caller passes a tmp directory.
    assert gradle_home is not None, (
        "pass gradle_home: otherwise the test may write into the real ~/.gradle"
    )
    env["GRADLE_USER_HOME"] = str(gradle_home)

    proc = subprocess.Popen(
        ["bash", str(_LAUNCHER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=_CMD_BUDGET_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=10)
        pytest.fail(
            f"'deep-worker {' '.join(args)}' exceeded {_CMD_BUDGET_SEC}s — "
            f"treat as a hang, not a flake.\nstdout={out}\nstderr={err}"
        )
    finally:
        if proc.poll() is None:  # pragma: no cover — defensive
            proc.kill()
            proc.wait(timeout=10)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_lsp_status_runs_via_real_binary(tmp_path: Path) -> None:
    """``deep-worker lsp status`` must work, not fail with
    ``unrecognized arguments``. This is exactly that wiring defect."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"

    res = _run(["lsp", "status", "--config", str(cfg)], gradle_home=tmp_path / "gh")

    combined = res.stdout + res.stderr
    assert "unrecognized arguments" not in combined, combined
    assert "invalid choice" not in combined, combined
    assert res.returncode == 0, combined
    assert "status:" in res.stdout, res.stdout
    assert "disabled" in res.stdout.lower(), res.stdout


def test_lsp_enable_then_status_via_real_binary(tmp_path: Path) -> None:
    """Full user path through the real binary."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"

    enabled = _run(
        ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)],
        gradle_home=tmp_path / "gh",
    )
    assert enabled.returncode == 0, enabled.stdout + enabled.stderr
    assert cfg.exists(), "config was not written by the real CLI"

    status = _run(["lsp", "status", "--config", str(cfg)], gradle_home=tmp_path / "gh")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "enabled" in status.stdout.lower(), status.stdout


def test_lsp_appears_in_help(tmp_path: Path) -> None:
    """The user must see the command in the help — that is where the
    report started: ``deep-worker --help | grep lsp`` found nothing."""
    res = _run(["--help"], gradle_home=tmp_path / "gh")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "lsp" in res.stdout, res.stdout


def test_normal_agent_launch_still_parses(tmp_path: Path) -> None:
    """REGRESSION LOCK.

    A naive addition of subparsers to the main parser breaks the positional
    ``name`` (``nargs='?'``): ``deep-worker myagent`` starts failing with
    ``invalid choice: 'myagent'``. We verify that this does NOT happen.

    We do not actually start an agent — we use a deliberately conflicting
    flag combination that reaches validation AFTER argument parsing: all
    that matters is that argparse does not reject the name as a subcommand.
    """
    res = _run(["--list", "some_agent_name"], gradle_home=tmp_path / "gh")

    combined = res.stdout + res.stderr
    assert "invalid choice" not in combined, combined
    assert "unrecognized arguments" not in combined, combined
    # Expect a domain error ("name cannot be combined with --list"), NOT
    # an argparse subcommand-parsing error.
    assert "Cannot specify agent name" in combined, combined


def test_unknown_subcommand_is_not_swallowed(tmp_path: Path) -> None:
    """An agent name resembling a subcommand must not be intercepted: the
    dispatcher must react ONLY to 'lsp'."""
    res = _run(["--list", "lspagent"], gradle_home=tmp_path / "gh")
    combined = res.stdout + res.stderr
    assert "Cannot specify agent name" in combined, combined


def _make_worktree(root: Path) -> Path:
    """Directory where ``.git`` is a FILE (as in a git worktree)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text(
        "gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8"
    )
    return root


def test_enable_without_flags_targets_the_invocation_dir(tmp_path: Path) -> None:
    """REGRESSION on a user-reported bug.

    ``deep-worker lsp enable`` WITHOUT flags, run from a worktree, wrote
    the config into the INSTALLATION directory (the launcher does
    ``cd "$SCRIPT_DIR"``, and the code read ``Path.cwd()``).

    The key difference from the other tests: NO ``--config`` /
    ``--project-root`` — we test exactly the auto-detection path a human
    uses. Its absence is what let the bug through.
    """
    user_dir = _make_worktree(tmp_path / "user_worktree")

    res = _run(["lsp", "enable"], cwd=user_dir, gradle_home=tmp_path / "gh")
    assert res.returncode == 0, res.stdout + res.stderr

    cfg = user_dir / ".deep_worker" / "mcp.json"
    assert cfg.exists(), (
        f"config must land in the invocation dir; stdout={res.stdout}"
    )

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["serena"]
    assert str(user_dir) in entry["args"], entry["args"]
    # The installation directory must not leak into the path or --project.
    assert str(_PKG_ROOT) not in entry["args"], entry["args"]


def test_status_without_flags_reports_invocation_dir(tmp_path: Path) -> None:
    """``lsp status`` must show the invocation directory root — otherwise a
    silent root substitution would go unnoticed again."""
    user_dir = _make_worktree(tmp_path / "user_worktree")

    res = _run(["lsp", "status"], cwd=user_dir, gradle_home=tmp_path / "gh")

    assert res.returncode == 0, res.stdout + res.stderr
    assert str(user_dir) in res.stdout, res.stdout


# --------------------------------------------------------------------------
# Gradle init script (Kotlin/Android cross-file navigation)
#
# No mocks: real launcher, real filesystem, isolated $GRADLE_USER_HOME.
# Unit tests call functions directly and would not catch a situation like
# "resource did not make it into the distribution" or "flag did not reach
# the parser".
# --------------------------------------------------------------------------


def test_enable_installs_gradle_init_script_via_real_binary(tmp_path: Path) -> None:
    gradle_home = tmp_path / "gradle-home"
    cfg = tmp_path / ".deep_worker" / "mcp.json"

    res = _run(
        ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)],
        gradle_home=gradle_home,
    )
    assert res.returncode == 0, res.stdout + res.stderr

    script = gradle_home / "init.d" / "deep-worker-kotlin-android-source-sets.gradle"
    assert script.exists(), (
        f"init script must be installed by a real 'lsp enable'; stdout={res.stdout}"
    )
    # The resource must be NON-EMPTY and carry a version marker: if it did
    # not make it into the distribution, the file will be missing or garbage.
    first_line = script.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("// deep-worker:kotlin-android-source-sets v"), first_line
    assert stat.S_IMODE(script.stat().st_mode) == 0o644
    assert stat.S_IMODE(script.parent.stat().st_mode) == 0o755


def test_enable_no_gradle_init_flag_via_real_binary(tmp_path: Path) -> None:
    gradle_home = tmp_path / "gradle-home"
    cfg = tmp_path / ".deep_worker" / "mcp.json"

    res = _run(
        [
            "lsp",
            "enable",
            "--config",
            str(cfg),
            "--project-root",
            str(tmp_path),
            "--no-gradle-init",
        ],
        gradle_home=gradle_home,
    )
    combined = res.stdout + res.stderr
    assert "unrecognized arguments" not in combined, combined
    assert res.returncode == 0, combined
    assert not (gradle_home / "init.d").exists()


def test_disable_removes_gradle_init_script_via_real_binary(tmp_path: Path) -> None:
    gradle_home = tmp_path / "gradle-home"
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    script = gradle_home / "init.d" / "deep-worker-kotlin-android-source-sets.gradle"

    _run(
        ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)],
        gradle_home=gradle_home,
    )
    assert script.exists()

    res = _run(["lsp", "disable", "--config", str(cfg)], gradle_home=gradle_home)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not script.exists(), res.stdout
