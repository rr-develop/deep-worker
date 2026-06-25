"""Global pytest configuration for deep_worker_cli tests.

CRITICAL REQUIREMENT: ALWAYS TEST LOCAL REPOSITORY CODE
========================================================

This conftest.py ensures that tests ALWAYS use the local repository code
from this repository tree, NOT installed packages from site-packages.

WHY THIS IS CRITICAL:
---------------------
- Tests MUST validate the actual code in THIS repository
- Testing installed packages is COMPLETELY UNACCEPTABLE
- Using site-packages versions leads to:
  * Testing outdated/different code
  * False test results
  * Wasted debugging time
  * Production bugs that tests don't catch

NEVER REMOVE OR DISABLE THIS PATH MANIPULATION!
------------------------------------------------
If you think you need to remove this, you are WRONG.
The correct solution is to fix the code, not the test configuration.

This module is executed BEFORE any test code runs, ensuring correct
import paths from the very beginning.
"""

import sys
from pathlib import Path

# ============================================================================
# CRITICAL: /path/to/rpyc_async on sys.path BEFORE any RPyC import
# ============================================================================
# Production code does this inside `deep_worker_cli.process.agent_process`
# (see `agent_process.py:16-22`). Tests that import
# `deep_worker_cli.rpyc_service.agent_service` directly never go through
# `agent_process`, so `import rpyc` inside `agent_service.py` fails with
# ModuleNotFoundError. We mirror the production injection here so tests
# see the same `rpyc` the production process does.
LOCAL_RPYC_ASYNC = Path("/path/to/rpyc_async")
if LOCAL_RPYC_ASYNC.exists():
    rpyc_async_str = str(LOCAL_RPYC_ASYNC)
    if rpyc_async_str in sys.path:
        sys.path.remove(rpyc_async_str)
    sys.path.insert(0, rpyc_async_str)

# ============================================================================
# CRITICAL: Force local deepagents to be first in sys.path
# ============================================================================
# This MUST happen before any imports from deepagents.
# DO NOT use editable install or pip install - those are NOT the repository code.

# Path to local deepagents (sibling to deep_worker_cli)
LOCAL_DEEPAGENTS = Path(__file__).parent.parent.parent / "deepagents"

if not LOCAL_DEEPAGENTS.exists():
    msg = (
        f"CRITICAL ERROR: Local deepagents not found at {LOCAL_DEEPAGENTS}\n"
        "Tests REQUIRE local repository code, not installed packages.\n"
        "Check your repository structure."
    )
    raise RuntimeError(msg)

# FORCE local deepagents to be FIRST (before site-packages)
deepagents_str = str(LOCAL_DEEPAGENTS)
if deepagents_str in sys.path:
    sys.path.remove(deepagents_str)
sys.path.insert(0, deepagents_str)

# ============================================================================
# CRITICAL: local deepworker_a2a package importable from THIS tree
# ============================================================================
# The A2A protocol package lives at the deep_worker package root, a sibling of
# deep_worker_cli / deepagents. Production puts that root on PYTHONPATH via the
# deep-worker launcher; mirror that here so `import deepworker_a2a` resolves to
# the LOCAL repository package, never site-packages. It is named
# `deepworker_a2a` (NOT `a2a`) so it never shadows the installed `a2a-sdk`.
DEEP_WORKER_DIR = Path(__file__).parent.parent.parent  # the deep_worker package root
deep_worker_dir_str = str(DEEP_WORKER_DIR)
if deep_worker_dir_str in sys.path:
    sys.path.remove(deep_worker_dir_str)
sys.path.insert(0, deep_worker_dir_str)

# Verify that we're using the correct deepagents
try:
    import deepagents
    actual_path = Path(deepagents.__file__).parent.parent
    if actual_path != LOCAL_DEEPAGENTS:
        msg = (
            f"CRITICAL ERROR: Wrong deepagents is being imported!\n"
            f"Expected: {LOCAL_DEEPAGENTS}\n"
            f"Actual:   {actual_path}\n"
            f"This means tests are running against WRONG CODE.\n"
            f"Check sys.path configuration."
        )
        raise RuntimeError(msg)
except ImportError as e:
    msg = (
        f"CRITICAL ERROR: Cannot import deepagents from {LOCAL_DEEPAGENTS}\n"
        f"Error: {e}\n"
        f"Tests CANNOT run without local deepagents."
    )
    raise RuntimeError(msg) from e

# Verify ProjectMemoryMiddleware is available
try:
    from deepagents.middleware.project_memory import ProjectMemoryMiddleware
    # Success - we have the real local version
except ImportError:
    # This should NOT happen if we're using local code
    msg = (
        "CRITICAL ERROR: ProjectMemoryMiddleware not found in local deepagents.\n"
        f"Path: {LOCAL_DEEPAGENTS / 'deepagents' / 'middleware' / 'project_memory.py'}\n"
        "This indicates the local deepagents is incomplete or corrupted."
    )
    raise RuntimeError(msg)

print(f"✓ Tests will use LOCAL deepagents from: {LOCAL_DEEPAGENTS}")
print(f"✓ ProjectMemoryMiddleware available from local code")


def _snapshot_test_agent_artefacts() -> dict[str, set[str]]:
    """Snapshot the ``test_*`` dirs under the agent data directory + ``/tmp/deep_worker_test_*.pid``
    files + ``test_*`` entries in the registry that exist BEFORE the
    pytest session starts.

    The diff against the post-session state tells us which artefacts
    THIS session leaked. Items in the snapshot are preserved (they
    predate this session); anything not in the snapshot but present
    afterwards is a leak we created and must clean up.
    """
    home_dw = Path("~/.deep_worker").expanduser()
    dirs: set[str] = set()
    if home_dw.exists():
        for child in home_dw.iterdir():
            if child.is_dir() and child.name.startswith("test_"):
                dirs.add(child.name)

    pid_files: set[str] = set()
    tmp = Path("/tmp")
    if tmp.exists():
        for f in tmp.glob("deep_worker_test_*.pid"):
            pid_files.add(f.name)

    registry_names: set[str] = set()
    reg_file = Path("/tmp/deep_worker_registry.json")
    if reg_file.exists():
        try:
            import json as _json
            data = _json.loads(reg_file.read_text())
            for name in data:
                if name.startswith("test_"):
                    registry_names.add(name)
        except (OSError, ValueError):
            pass

    return {
        "dirs": dirs,
        "pid_files": pid_files,
        "registry_names": registry_names,
    }


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    """Record the pre-session artefact snapshot. Anything not in this
    snapshot but present at sessionfinish is a leak we created."""
    session.config._dwcli_test_artefact_snapshot = (  # type: ignore[attr-defined]
        _snapshot_test_agent_artefacts()
    )


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """Sweep ``test_*`` artefacts THIS session leaked.

    Why this exists: per-fixture cleanup races with deep-worker's
    daemonize sequence (the child registers itself AFTER the parent
    ``subprocess.run`` returns), and some unit tests instantiate an
    ``AgentProcess`` directly — that bypasses ``tmp_path`` for
    persistence because ``SessionManager`` always writes to
    the agent data directory (``<name>/``) based on the agent's name, not its
    working_dir. Without this sweep, every pytest run leaves stale
    ``test_*`` dirs, ``/tmp/deep_worker_test_*.pid`` files, and
    registry entries on the host — the operator sees them as live
    test agents in ``dwcli --list``.

    We diff against the pre-session snapshot so this fixture NEVER
    deletes artefacts that predate the session.
    """
    before = getattr(
        session.config, "_dwcli_test_artefact_snapshot",
        {"dirs": set(), "pid_files": set(), "registry_names": set()},
    )

    import os as _os
    import signal as _signal
    import shutil as _shutil
    import json as _json
    import time as _time

    # 1. Registry: kill PIDs we own, then unregister.
    reg_file = Path("/tmp/deep_worker_registry.json")
    new_registry_names: set[str] = set()
    if reg_file.exists():
        try:
            data = _json.loads(reg_file.read_text())
        except (OSError, ValueError):
            data = {}
        for name, entry in list(data.items()):
            if not name.startswith("test_"):
                continue
            if name in before["registry_names"]:
                continue  # predates this session — leave it alone
            new_registry_names.add(name)
            pid = entry.get("pid") if isinstance(entry, dict) else None
            if isinstance(pid, int):
                try:
                    _os.kill(pid, _signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        # Give SIGTERM a brief moment, then force-kill survivors.
        _time.sleep(0.5)
        for name in new_registry_names:
            entry = data.get(name, {})
            pid = entry.get("pid") if isinstance(entry, dict) else None
            if isinstance(pid, int):
                try:
                    _os.kill(pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            from deep_worker_cli.process.registry import ProcessRegistry  # noqa: PLC0415
            registry = ProcessRegistry()
            for name in new_registry_names:
                try:
                    registry.unregister(name)
                except Exception:
                    pass
        except Exception:
            pass

    # 2. ``test_*`` dirs under the agent data directory created during this session.
    home_dw = Path("~/.deep_worker").expanduser()
    if home_dw.exists():
        for child in home_dw.iterdir():
            if not (child.is_dir() and child.name.startswith("test_")):
                continue
            if child.name in before["dirs"]:
                continue
            _shutil.rmtree(child, ignore_errors=True)

    # 3. ``/tmp/deep_worker_test_*.pid`` files created during this session.
    tmp = Path("/tmp")
    if tmp.exists():
        for f in tmp.glob("deep_worker_test_*.pid"):
            if f.name in before["pid_files"]:
                continue
            try:
                f.unlink()
            except OSError:
                pass


def pytest_collection_modifyitems(config, items):
    """Auto-mark everything under `tests/integration/` and
    `tests/integration_ai/` as `@pytest.mark.integration`.

    The default `-m 'not integration'` filter in pyproject.toml
    deselects those suites because they either spin up a real
    deep-worker agent or run `AsyncioServer + rpyc client` in the
    same process (forbidden by rpyc_async policy — see
    `/path/to/rpyc_async/docs/DESIGN_NO_SAME_PROCESS_TESTS.md`).
    Running them deliberately with `pytest -m integration` or
    pointing at the directory still works.
    """
    import pytest as _pytest
    tests_dir = Path(__file__).parent.resolve()
    integration_dirs = {
        tests_dir / "integration",
        tests_dir / "integration_ai",
    }
    marker = _pytest.mark.integration
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        for d in integration_dirs:
            try:
                item_path.relative_to(d)
            except ValueError:
                continue
            else:
                item.add_marker(marker)
                break
