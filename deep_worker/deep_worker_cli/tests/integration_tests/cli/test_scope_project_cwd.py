"""``--scope project`` must resolve from the INVOCATION directory — on the REAL binary.

Why this file exists
--------------------
The ``deep-worker`` launcher does ``cd "$SCRIPT_DIR"`` before starting
Python, so ``Path.cwd()`` inside the process points at the INSTALLATION
directory, not the user's directory. A command that resolves project-scope
from ``Path.cwd()`` reads and WRITES another repository's config.

The defect was reproduced live: ``deep-worker mcp list --scope project``
run from an arbitrary directory showed a config path in the installation
tree — i.e. ``mcp add`` would have written there too. Unit tests did not
catch this: they call functions directly, without the launcher, where the
cwd substitution simply does not exist.

Hence the rule (see also ``cli/cwd.py``): every scope-by-cwd command must
have a subprocess test on the real launcher WITHOUT flags that point the
path explicitly — otherwise the test checks the wrong thing.

Policy "integration tests must not hang": a hard budget on each run, and
the process is guaranteed to be killed in ``finally``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# tests/integration_tests/cli/ → tests/integration_tests → tests → package
_PKG_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _PKG_ROOT / "deep-worker"

_CMD_BUDGET_SEC = 60.0


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run this tree's launcher from ``cwd`` with a hard budget."""
    assert _LAUNCHER.is_file(), f"launcher not found: {_LAUNCHER}"

    env = {**os.environ}
    env.pop("DEEP_WORKER_DEV_TREE", None)
    env.pop("DEEP_WORKER_TREE_ROOT", None)
    # Key point: an inherited ORIGINAL_CWD would make the check
    # unreliable — the launcher must set it itself from the actual
    # cwd of the subprocess.
    env.pop("DEEP_WORKER_ORIGINAL_CWD", None)

    proc = subprocess.Popen(
        ["bash", str(_LAUNCHER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Isolated git repository with a nested subdirectory."""
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, timeout=30
    )
    (tmp_path / "sub").mkdir()
    return tmp_path


def test_mcp_list_project_scope_uses_invocation_dir(repo: Path) -> None:
    """``mcp list --scope project`` looks into the repository of the INVOCATION."""
    res = _run(["mcp", "list", "--scope", "project"], cwd=repo / "sub")

    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined

    # Positive check: the command prints the config path as its first line.
    # Checking "absence of some foreign path" is not enough — under the
    # regression the path moves to the installation directory, which
    # differs between trees, and a negative assert silently passes.
    expected = repo.resolve() / ".deepagents" / "mcp.json"
    assert str(expected) in combined, (
        f"project-scope does not resolve from the invocation directory.\n"
        f"expected: {expected}\noutput:\n{combined}"
    )


def test_mcp_add_project_scope_writes_into_invocation_repo(repo: Path) -> None:
    """``mcp add --scope project`` WRITES into the invocation repository.

    The most destructive case: before the fix, the write went into a
    foreign tree.
    """
    res = _run(
        [
            "mcp", "add", "probe",
            "--command", "echo hi",
            "--scope", "project",
        ],
        cwd=repo / "sub",
    )

    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined

    written = repo / ".deepagents" / "mcp.json"
    assert written.is_file(), (
        f"config not written to the invocation repository ({written}).\n{combined}"
    )
    data = json.loads(written.read_text(encoding="utf-8"))
    assert "probe" in data.get("mcpServers", {}), data

    # Permissions: the file may contain tokens.
    assert oct(written.stat().st_mode & 0o777) == "0o600"


def test_lsp_enable_project_scope_writes_into_invocation_repo(repo: Path) -> None:
    """Same contract for ``lsp enable`` (the regression already happened)."""
    res = _run(["lsp", "enable"], cwd=repo / "sub")

    combined = res.stdout + res.stderr
    # Serena may not be installed in the CI environment — that is not our defect.
    if res.returncode != 0 and "serena" in combined.lower():
        pytest.skip(f"serena unavailable in this environment: {combined[:200]}")

    assert res.returncode == 0, combined
    written = repo / ".deepagents" / "mcp.json"
    assert written.is_file(), (
        f"lsp enable wrote the config outside the invocation repository.\n{combined}"
    )
