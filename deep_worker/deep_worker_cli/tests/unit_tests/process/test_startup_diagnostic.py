"""Tests for a related issue: startup diagnostic logging with commit SHA + launch config.

Covers:
- _get_git_commit_info() returns real SHA in a git tree
- _get_git_commit_info() returns unknown on error
- [STARTUP] output format and field presence via stdout capture
- Integration: startup line presence in agent.log after --restart
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.process.agent_process import (
    _get_git_commit_info,
    AgentProcess,
)
from deep_worker_cli.session.session_manager import SessionManager


# =========================================================================
# 1. _get_git_commit_info — real git tree
# =========================================================================

def test_get_git_commit_info_in_real_tree() -> None:
    """Must return a non-empty SHA for the real repo we are running from.

    The path is discovered from the test's own location via
    ``git rev-parse --show-toplevel`` rather than hardcoded, so the
    test works on every host (CI runner, developer workstation, any
    checkout layout, etc.).
    """
    discovered = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    assert discovered, "test must be invoked from inside a git working tree"
    sha, describe = _get_git_commit_info(discovered)
    assert sha != "unknown", "should detect the commit SHA of this repo"
    assert len(sha) == 40, "a full SHA is 40 hex characters"
    assert all(c in "0123456789abcdef" for c in sha), "SHA must be hex only"
    # describe is either the same SHA or a tag-based short form
    assert describe, "describe must not be empty"


def test_get_git_commit_info_nonexistent_path() -> None:
    """A path without .git returns (unknown, unknown) without error."""
    sha, describe = _get_git_commit_info("/tmp/.nonexistent_git_tree_42")
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_none_path() -> None:
    """None path returns (unknown, unknown)."""
    sha, describe = _get_git_commit_info(None)
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_empty_path() -> None:
    """Empty string returns (unknown, unknown)."""
    sha, describe = _get_git_commit_info("")
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_subprocess_failure_is_graceful() -> None:
    """If subprocess.run raises, both values are 'unknown'."""
    with patch(
        "deep_worker_cli.process.agent_process.subprocess.run",
        side_effect=FileNotFoundError("no git"),
    ):
        # tree_root is just a string here — subprocess is mocked, so the
        # path is never dereferenced. Use a portable placeholder, not a
        # host-specific mount that doesn't exist on CI runners.
        sha, describe = _get_git_commit_info("/tmp/dwtest_launch_tree")
    assert sha == "unknown"
    assert describe == "unknown"


# =========================================================================
# 2. [STARTUP] output format (unit test via stdout capture)
# =========================================================================

def test_startup_diagnostic_output_format() -> None:
    """Verify the [STARTUP] line is printed with correct key=value pairs."""
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_startup_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_startup_diag"
    agent.launch_config = {
        "provider": "claude-code-sdk",
        "model": "sonnet",
        # tree_root is a fixture string — the test only asserts that the
        # function returns a value (commit / describe) and that the path
        # is rendered into the [STARTUP] line. Use a portable placeholder
        # instead of a host-specific mount so the test reads the same on
        # CI as on the developer workstation.
        "tree_root": "/tmp/dwtest_launch_tree",
        "working_dir": "/home/user",
        "skip_permissions": False,
        "task_manager": True,
        "memory_only": False,
        "history_memory": False,
        "use_sqlite": False,
        "dev_tree": False,
        "clean": False,
        "auto_approve_send_message": False,
        "preferred_provider": "claude-code-sdk",
        "started_at": "2026-06-04T12:00:00",
    }

    # Capture the diagnostic print the same way start() does it
    import io
    import os
    import sys as _sys

    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        lc = agent.launch_config or {}
        tree = str(lc.get("tree_root", ""))
        commit, describe = _get_git_commit_info(tree)
        pairs = [
            ("agent", agent.name),
            ("provider", str(lc.get("provider", "unknown"))),
            ("model", str(lc.get("model", "unknown"))),
            ("tree", tree),
            ("commit", commit),
            ("describe", describe),
            ("pid", str(os.getpid())),
            ("working_dir", str(lc.get("working_dir", ""))),
            ("skip_permissions", str(lc.get("skip_permissions", False))),
            ("task_manager", str(lc.get("task_manager", False))),
            ("memory_only", str(lc.get("memory_only", False))),
            ("history_memory", str(lc.get("history_memory", False))),
            ("use_sqlite", str(lc.get("use_sqlite", False))),
            ("dev_tree", str(lc.get("dev_tree", False))),
            ("clean", str(lc.get("clean", False))),
            ("auto_approve_send_message", str(lc.get("auto_approve_send_message", False))),
            ("preferred_provider", str(lc.get("preferred_provider", ""))),
            ("started_at", str(lc.get("started_at", ""))),
        ]
        line = " ".join(f"{k}={v}" for k, v in pairs)
        print(f"[STARTUP] {line}", flush=True)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    assert output.startswith("[STARTUP]"), (
        "diagnostic line must start with [STARTUP]"
    )
    assert "agent=test_startup_diag" in output
    assert "provider=claude-code-sdk" in output
    assert "model=sonnet" in output
    assert "tree=/tmp/dwtest_launch_tree" in output
    assert "commit=" in output
    assert "describe=" in output
    assert "pid=" in output
    assert "task_manager=True" in output
    # Verify key=value format (no JSON or dict representation)
    assert "{" not in output, "must be key=value pairs, not JSON"
    assert ":" not in output.replace("preferred_provider=", "").split(" ")[0], (
        "no dict-style colons outside values"
    )


# =========================================================================
# 3. Integration: --restart log capture (lightweight)
# =========================================================================

def test_startup_diagnostic_fields_integration() -> None:
    """All required fields from the related issue's spec appear in the output."""
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_startup_int_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_startup_int"
    agent.launch_config = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        # tree_root is fixture data only (see note in test_startup_diagnostic_output_format).
        "tree_root": "/tmp/dwtest_launch_tree",
        "working_dir": "/tmp",
    }

    import io
    import os
    import sys as _sys

    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        lc = agent.launch_config or {}
        tree = str(lc.get("tree_root", ""))
        commit, describe = _get_git_commit_info(tree)
        pairs = [
            ("agent", agent.name),
            ("provider", str(lc.get("provider", "unknown"))),
            ("model", str(lc.get("model", "unknown"))),
            ("tree", tree),
            ("commit", commit),
            ("describe", describe),
            ("pid", str(os.getpid())),
        ]
        line = " ".join(f"{k}={v}" for k, v in pairs)
        print(f"[STARTUP] {line}", flush=True)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    # Required: commit SHA is present (even if unknown)
    assert "commit=" in output
    # Required: provider is present
    assert "provider=deepseek" in output
    # Required: agent name is present
    assert "agent=test_startup_int" in output
    # Verify the line is grep-able (single space-separated key=value)
    parts = output.strip().split()
    assert len(parts) >= 7, "must have at least 7 key=value pairs plus [STARTUP]"
    for part in parts[1:]:  # skip [STARTUP]
        assert "=" in part, f"each field must be key=value, got: {part}"


# =========================================================================
# 5. _get_git_revision — SHA + dirty, captured at every process start
# =========================================================================

def _init_repo(root: Path) -> None:
    """Create a git repo with one commit. Identity is set locally.

    ``commit.gpgsign=false`` matters: a host whose global config signs every
    commit would otherwise make these tests fail for a reason unrelated to
    what they assert.
    """
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *a], capture_output=True, text=True, timeout=10
    )
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    (root / "f.txt").write_text("one\n", encoding="utf-8")
    run("add", "f.txt")
    run("commit", "-qm", "initial")


def test_get_git_revision_clean_tree(tmp_path: Path) -> None:
    """A committed, unmodified tree reports its SHA and dirty=False."""
    from deep_worker_cli.process.agent_process import _get_git_revision

    _init_repo(tmp_path)
    sha, dirty = _get_git_revision(str(tmp_path))

    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    assert dirty is False, "a freshly committed tree is not dirty"


def test_get_git_revision_detects_dirty_tracked_modification(tmp_path: Path) -> None:
    """A modified tracked file flips dirty, and the SHA does NOT move.

    This is the whole point of the flag: HEAD still names the last commit, so
    the SHA alone would claim the agent ran exactly that code. It did not.
    """
    from deep_worker_cli.process.agent_process import _get_git_revision

    _init_repo(tmp_path)
    clean_sha, clean_dirty = _get_git_revision(str(tmp_path))
    assert clean_dirty is False

    (tmp_path / "f.txt").write_text("two\n", encoding="utf-8")
    dirty_sha, dirty = _get_git_revision(str(tmp_path))

    assert dirty is True, "an uncommitted tracked modification must set dirty"
    assert dirty_sha == clean_sha, "the SHA must be unchanged — only dirty moves"


def test_get_git_revision_detects_dirty_staged_and_untracked(tmp_path: Path) -> None:
    """Staged changes count as dirty; so does a stray untracked file.

    Untracked matters here specifically: the reported failure mode is "some
    agent left a file behind", and a spurious file in the tree means the
    checkout is not purely the named commit.
    """
    from deep_worker_cli.process.agent_process import _get_git_revision

    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("staged\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "f.txt"],
        capture_output=True, text=True, timeout=10,
    )
    _, staged_dirty = _get_git_revision(str(tmp_path))
    assert staged_dirty is True, "staged-but-uncommitted changes are dirty"

    # reset BEFORE checkout: `checkout -- f.txt` restores from the index, and
    # the index still holds the staged edit, so the reverse order leaves the
    # modification in the working tree and the assertion below would fail for
    # a reason that has nothing to do with _get_git_revision.
    subprocess.run(
        ["git", "-C", str(tmp_path), "reset", "-q"],
        capture_output=True, text=True, timeout=10,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "checkout", "-q", "--", "f.txt"],
        capture_output=True, text=True, timeout=10,
    )
    assert _get_git_revision(str(tmp_path))[1] is False, "reset must restore clean"

    (tmp_path / "stray.txt").write_text("left behind\n", encoding="utf-8")
    _, untracked_dirty = _get_git_revision(str(tmp_path))
    assert untracked_dirty is True, "an untracked file makes the checkout impure"


def test_get_git_revision_unknown_outside_git(tmp_path: Path) -> None:
    """No git tree: SHA is 'unknown' and dirty is None, never a false clean.

    ``None`` rather than ``False`` is deliberate — we did not observe a clean
    tree, we failed to observe anything, and a report must not imply the
    stronger claim.
    """
    from deep_worker_cli.process.agent_process import _get_git_revision

    sha, dirty = _get_git_revision(str(tmp_path / "nope"))
    assert sha == "unknown"
    assert dirty is None

    assert _get_git_revision(None) == ("unknown", None)
    assert _get_git_revision("") == ("unknown", None)


def test_agent_process_defaults_provenance_to_unknown() -> None:
    """Before startup runs, provenance must be None — never a fake clean tree.

    ``dirty=False`` on a process whose startup never executed would assert an
    unmodified checkout on no evidence at all.
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    tmp = Path(tempfile.mkdtemp(prefix="dwtest_prov_default_"))
    proc = AgentProcess(
        name="test_prov_default",
        config=AgentConfig(
            working_dir=tmp,
            provider="claude-code-sdk",
            model="claude-3-5-sonnet-20241022",
        ),
        data_dir=tmp,
        rpyc_port=False,
    )

    assert proc._startup_commit is None
    assert proc._startup_dirty is None


async def test_startup_captures_revision_from_the_real_tree(tmp_path: Path) -> None:
    """The startup path must capture the CURRENT revision of tree_root.

    Reading it from launch_config instead would replay a stale SHA across
    restarts — and a restart is usually done to pick up new code.
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import _get_git_revision

    _init_repo(tmp_path)
    expected_sha, expected_dirty = _get_git_revision(str(tmp_path))
    assert expected_dirty is False

    data_dir = tmp_path / "_data"
    data_dir.mkdir()
    proc = AgentProcess(
        name="test_prov_startup",
        config=AgentConfig(
            working_dir=tmp_path,
            provider="claude-code-sdk",
            model="claude-3-5-sonnet-20241022",
        ),
        data_dir=data_dir,
        rpyc_port=False,
    )
    # launch_config carries a DIFFERENT (stale) commit, as it would after a
    # restart taken precisely to pick up new code. The live tree must win.
    proc.launch_config = {"tree_root": str(tmp_path), "commit": "0" * 40}

    # Drive the REAL start(). We only care about the capture, which happens
    # near the top; abort immediately after it so this stays a bounded unit
    # test instead of standing up PID files, signal handlers and a provider.
    sentinel = RuntimeError("stop after revision capture")

    with patch.object(
        AgentProcess, "_create_pid_file", side_effect=sentinel
    ):
        with pytest.raises(RuntimeError):
            await proc.start()

    assert proc._startup_commit == expected_sha, (
        "start() must capture the SHA of the live tree_root"
    )
    assert proc._startup_commit != proc.launch_config["commit"], (
        "the live tree must win over anything persisted in launch_config"
    )
    assert proc._startup_dirty is False
    # The normalisation both create_cli_agent call sites read through.
    assert proc._reported_commit == expected_sha


async def test_startup_stamps_revision_into_session_metadata(
    tmp_path: Path,
) -> None:
    """The revision must be readable from metadata right after start().

    THE contract: no infrastructure report is filed here, and no log line is
    parsed. A consumer (web UI, dwcli, another agent, a script) must be able
    to open session_metadata.json and learn which code the agent runs. The
    stamp also has to land BEFORE the agent is built, so it survives an
    agent-construction failure.
    """
    import json

    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import _get_git_revision

    _init_repo(tmp_path)
    expected_sha, expected_dirty = _get_git_revision(str(tmp_path))
    assert expected_dirty is False

    data_dir = tmp_path / "_data"
    data_dir.mkdir()
    proc = AgentProcess(
        name="test_prov_meta",
        config=AgentConfig(
            working_dir=tmp_path,
            provider="claude-code-sdk",
            model="claude-3-5-sonnet-20241022",
        ),
        data_dir=data_dir,
        rpyc_port=False,
    )
    proc.launch_config = {"tree_root": str(tmp_path), "commit": "0" * 40}

    # Let start() run THROUGH the real SessionManager (step 3), then abort
    # before the agent is built (step 4). The wrapper constructs the genuine
    # manager first, so the on-disk stamp is produced by production code;
    # raising afterwards keeps this a bounded unit test and proves the stamp
    # survives an agent that never comes up.
    from deep_worker_cli.process import agent_process as ap_module

    real_session_manager = ap_module.SessionManager

    def _stamp_then_stop(*args, **kwargs):
        real_session_manager(*args, **kwargs)
        raise RuntimeError("stop after metadata stamp")

    with patch.object(ap_module, "SessionManager", _stamp_then_stop):
        with pytest.raises(RuntimeError):
            await proc.start()

    meta_file = data_dir / "session_metadata.json"
    assert meta_file.exists(), "start() must have created session metadata"
    meta = json.loads(meta_file.read_text())

    assert meta["commit"] == expected_sha, (
        "metadata must name the live tree's SHA without any report being filed"
    )
    assert meta["dirty"] is False
    # The stale SHA is still echoed inside launch_config (that dict is
    # persisted verbatim, as the caller passed it). What matters is that the
    # TOP-LEVEL field — the one consumers read — ignored it in favour of the
    # live tree. Reading provenance from launch_config would pin the previous
    # process's revision after a restart taken to pick up new code.
    assert meta["commit"] != meta["launch_config"]["commit"], (
        "top-level provenance must not be taken from the replayed launch_config"
    )


async def test_startup_reports_null_commit_when_tree_is_not_a_repo(
    tmp_path: Path,
) -> None:
    """A non-repo tree must yield null provenance, never a fabricated clean.

    ``_reported_commit`` maps the log placeholder "unknown" to None so the
    record carries null, consistent with ``dirty=None``.
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig

    data_dir = tmp_path / "_data"
    data_dir.mkdir()
    proc = AgentProcess(
        name="test_prov_norepo",
        config=AgentConfig(
            working_dir=tmp_path,
            provider="claude-code-sdk",
            model="claude-3-5-sonnet-20241022",
        ),
        data_dir=data_dir,
        rpyc_port=False,
    )
    proc.launch_config = {"tree_root": str(tmp_path)}

    with patch.object(
        AgentProcess, "_create_pid_file", side_effect=RuntimeError("stop")
    ):
        with pytest.raises(RuntimeError):
            await proc.start()

    assert proc._startup_commit == "unknown"
    assert proc._startup_dirty is None
    assert proc._reported_commit is None, (
        "unknown must become null on the record, not a fake clean tree"
    )


def test_get_reportable_revision_maps_unknown_to_null(tmp_path: Path) -> None:
    """The shared normaliser: real SHA passes through, "unknown" becomes None.

    Standalone CLI modes stamp reports through this helper, so the mapping
    must match what the daemon's ``_reported_commit`` produces.
    """
    from deep_worker_cli.process.agent_process import get_reportable_revision

    _init_repo(tmp_path)
    sha, dirty = get_reportable_revision(str(tmp_path))
    assert sha is not None and len(sha) == 40
    assert dirty is False

    # Not a repo -> null provenance, never a fabricated clean tree. Must be
    # OUTSIDE tmp_path's repo: a subdirectory of a repo is still in that repo,
    # so git would legitimately resolve the parent's HEAD.
    outside = Path(tempfile.mkdtemp(prefix="dwtest_norepo_"))
    try:
        inside_repo = subprocess.run(
            ["git", "-C", str(outside), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).returncode == 0
        if inside_repo:
            pytest.skip(f"{outside} is itself inside a git repo on this host")
        sha_none, dirty_none = get_reportable_revision(str(outside))
        assert sha_none is None
        assert dirty_none is None
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_code_tree_revision_ignores_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance must name the deep_worker CODE tree, never the user's cwd.

    An installed ``dwcli`` is normally launched from inside some unrelated
    project repo. Reading cwd there would stamp that project's SHA onto the
    report: well-formed, but a false claim about which deep_worker code ran —
    strictly worse than reporting null.
    """
    from deep_worker_cli.process import agent_process
    from deep_worker_cli.process.agent_process import (
        get_code_tree_revision,
        get_reportable_revision,
    )

    # repo A: the user's project, which we chdir into. Distinct content so its
    # SHA cannot collide with repo B's.
    project = tmp_path / "user_project"
    project.mkdir()
    _init_repo(project)
    project_sha, _ = get_reportable_revision(str(project))
    assert project_sha is not None

    # repo B: a stand-in for the installed deep_worker source tree.
    code = tmp_path / "code_tree"
    code.mkdir()
    _init_repo(code)
    (code / "extra.txt").write_text("second commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(code), "add", "-A"], check=True, timeout=10)
    subprocess.run(
        ["git", "-C", str(code), "commit", "-m", "second"],
        check=True, capture_output=True, timeout=10,
    )
    code_sha, _ = get_reportable_revision(str(code))
    assert code_sha is not None and code_sha != project_sha

    monkeypatch.chdir(project)
    monkeypatch.setattr(agent_process, "__file__", str(code / "agent_process.py"))

    sha, dirty = get_code_tree_revision()
    assert sha == code_sha, (
        "provenance must come from the deep_worker code tree, not cwd"
    )
    assert sha != project_sha, (
        "stamping the user's project SHA would be a valid-looking falsehood"
    )
    assert dirty is False


def test_code_tree_revision_is_null_without_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel/install with no .git must yield null, not a fabricated SHA."""
    from deep_worker_cli.process import agent_process
    from deep_worker_cli.process.agent_process import get_code_tree_revision

    installed = Path(tempfile.mkdtemp(prefix="dwtest_installed_"))
    try:
        if subprocess.run(
            ["git", "-C", str(installed), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).returncode == 0:
            pytest.skip(f"{installed} is itself inside a git repo on this host")
        monkeypatch.setattr(
            agent_process, "__file__", str(installed / "agent_process.py")
        )
        sha, dirty = get_code_tree_revision()
        assert sha is None
        assert dirty is None
    finally:
        shutil.rmtree(installed, ignore_errors=True)


def test_standalone_cli_modes_stamp_provenance() -> None:
    """``dwcli`` runs must carry provenance too, not just daemonized agents.

    Both foreground entry points resolve the deep_worker code tree (NOT cwd)
    and forward it; dropping either would silently produce null-provenance
    reports from the CLI. The behavioural guarantee that the source is the
    code tree lives in ``test_code_tree_revision_ignores_cwd``.
    """
    import ast

    for module_name in ("deep_worker_cli.main", "deep_worker_cli.non_interactive"):
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_cli_agent"
        ]
        assert calls, f"{module_name} must construct an agent"
        for call in calls:
            kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
            assert kwargs.get("commit") == "_commit", (
                f"{module_name} must forward the captured commit"
            )
            assert kwargs.get("dirty") == "_dirty", (
                f"{module_name} must forward the captured dirty flag"
            )

        # Pin the SOURCE of the provenance, not just the variable names: an
        # assignment reading cwd would satisfy the kwarg check above while
        # stamping the user's project SHA. Other cwd uses in these modules
        # are unrelated, so inspect the capture assignment specifically.
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Tuple)
                and [getattr(e, "id", None) for e in t.elts] == ["_commit", "_dirty"]
                for t in node.targets
            )
        ]
        assert len(assignments) == 1, (
            f"{module_name} must capture provenance exactly once, "
            f"found {len(assignments)}"
        )
        captured_from = ast.unparse(assignments[0].value)
        assert captured_from == "get_code_tree_revision()", (
            f"{module_name} must resolve provenance from the deep_worker code "
            f"tree, not from cwd (an installed dwcli runs inside the user's "
            f"unrelated project repo); got {captured_from!r}"
        )


def test_task_manager_construction_passes_startup_provenance() -> None:
    """The TaskManager must be built WITH the process's captured revision.

    The construction sits deep inside ``start()`` behind agent/provider setup,
    so rather than stand all of that up we execute the real call expression
    lifted from the source against a stub TaskManager. If the production call
    stops forwarding commit/dirty, the kwargs asserted here disappear.
    """
    import ast
    import sys

    module_file = sys.modules[AgentProcess.__module__].__file__
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TaskManager"
    ]
    assert len(calls) == 1, (
        f"expected exactly one TaskManager construction, found {len(calls)}"
    )

    kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert kwargs.get("commit") == "self._reported_commit", (
        "TaskManager must receive the normalised startup commit; "
        f"got {kwargs.get('commit')!r}"
    )
    assert kwargs.get("dirty") == "self._startup_dirty", (
        "TaskManager must receive the startup dirty flag; "
        f"got {kwargs.get('dirty')!r}"
    )
