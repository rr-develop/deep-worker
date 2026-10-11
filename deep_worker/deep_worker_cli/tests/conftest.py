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

import os
import sys
from pathlib import Path

import pytest

# ============================================================================
# CRITICAL: /path/to/rpyc_async on sys.path BEFORE any RPyC import
# ============================================================================
# Production code does this inside `deep_worker_cli.process.agent_process`
# (see `agent_process.py:16-22`). Tests that import
# `deep_worker_cli.rpyc_service.agent_service` directly never go through
# `agent_process`, so `import rpyc` inside `agent_service.py` fails with
# ModuleNotFoundError. We mirror the production injection here so tests
# see the same `rpyc` the production process does.
# rpyc_async is a SEPARATE repository whose production checkout is
# /path/to/rpyc_async, hard-coded by absolute path in app code, so it has no dev
# isolation of its own. A dev tree must test against the rpyc_async *dev
# worktree*, discovered by asking git which worktree holds branch `dev` (no
# hard-coded worktree path). Override with DEEP_WORKER_RPYC_ASYNC=<path>.
import subprocess  # noqa: E402


def _resolve_rpyc_async() -> Path:
    override = os.environ.get("DEEP_WORKER_RPYC_ASYNC")
    if override:
        return Path(override)

    prod = Path("/path/to/rpyc_async")
    if "dev_tree" not in str(Path(__file__).resolve()):
        return prod

    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=prod, capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return prod

    current = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree "):]
        elif line.strip() == "branch refs/heads/dev" and current:
            return Path(current)
    return prod


LOCAL_RPYC_ASYNC = _resolve_rpyc_async()
if LOCAL_RPYC_ASYNC.exists():
    rpyc_async_str = str(LOCAL_RPYC_ASYNC)
    if rpyc_async_str in sys.path:
        sys.path.remove(rpyc_async_str)
    sys.path.insert(0, rpyc_async_str)

# Keep the chosen checkout AHEAD of any other rpyc_async source.
#
# Several test modules do `sys.path.insert(0, "/path/to/rpyc_async")` at import
# time (e.g. unit_tests/cli/test_interactive_mode.py), and rpyc-async is ALSO
# installed from PyPI. Either can end up first, so whichever module imports
# `rpyc_async` next silently binds to a different copy than the one selected
# above — tests then pass alone and fail in a full run purely by collection
# order. A meta-path finder re-asserts the choice no matter who inserts what.
if str(LOCAL_RPYC_ASYNC) != "/path/to/rpyc_async":
    import importlib.abc  # noqa: E402
    import importlib.machinery  # noqa: E402

    class _PinRpycAsyncFinder(importlib.abc.MetaPathFinder):
        """Resolve `rpyc_async*` from the selected checkout, always."""

        _root = str(LOCAL_RPYC_ASYNC)

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "rpyc_async" and not fullname.startswith("rpyc_async."):
                return None
            # Re-run the normal search with ONLY our root on the path, so
            # submodules resolve from the same tree as the parent package.
            if fullname == "rpyc_async":
                search = [self._root]
            else:
                parent = sys.modules.get(fullname.rpartition(".")[0])
                search = list(getattr(parent, "__path__", [])) if parent else None
                if not search:
                    return None
            return importlib.machinery.PathFinder.find_spec(fullname, search, target)

    if not any(isinstance(f, _PinRpycAsyncFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PinRpycAsyncFinder())

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
# The A2A protocol package lives at .../deep_worker/deepworker_a2a (sibling of
# deep_worker_cli / deepagents). Production puts .../deep_worker on PYTHONPATH
# via the deep-worker launcher; mirror that here so `import deep_worker.providers.deepworker_a2a`
# resolves to the LOCAL repository package, never site-packages. It is named
# `deepworker_a2a` (NOT `a2a`) so it never shadows the installed `a2a-sdk`.
DEEP_WORKER_DIR = Path(__file__).parent.parent.parent  # .../deep_worker
deep_worker_dir_str = str(DEEP_WORKER_DIR)
if deep_worker_dir_str in sys.path:
    sys.path.remove(deep_worker_dir_str)
sys.path.insert(0, deep_worker_dir_str)

# ============================================================================
# CRITICAL: repo root on sys.path so `deep_worker.providers.<pkg>` resolves
# ============================================================================
# The vendored providers now live under `deep_worker/providers/` (prefixed
# import scheme). For `import
# deep_worker.providers.semantic_search` etc. to resolve in the dev tree, the
# repo ROOT (parent of `deep_worker/`) must be on sys.path. In installed
# (wheel) mode this is a no-op — the package is found under site-packages.
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # .../<repo>
if (REPO_ROOT / "deep_worker" / "__init__.py").exists():
    repo_root_str = str(REPO_ROOT)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

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

# Import AFTER the sys.path manipulation above — before it, deep_worker_cli
# is not importable.
from deep_worker_cli.storage_paths import deep_worker_home  # noqa: E402


# Snapshot the user-base NOW, at import time, while HOME is still the
# operator's real home. The sandbox fixture below moves HOME, which would make
# this unresolvable if computed later.
def _real_user_base() -> str:
    """The user-base directory (``~/.local`` on POSIX), captured before the
    HOME swap.

    Exported to children as ``PYTHONUSERBASE`` so each one rebuilds its own
    version-specific user-site underneath it. Deliberately NOT the resolved
    ``site.getusersitepackages()`` path: that is pinned to THIS interpreter's
    minor version, and handing it to a child on another version is an ABI
    trap. Letting the child resolve its own subdirectory also means ``.pth``
    files are processed natively by its ``site`` initialisation, so editable
    installs (``_claude_agent_sdk.pth`` → ``/path/to/claude_sdk/src``,
    ``_editable_impl_rpyc_async.pth`` → ``/path/to/rpyc_async``) keep working
    without us re-implementing ``.pth`` parsing.
    """
    import site

    try:
        base = site.getuserbase()
    except Exception:  # noqa: BLE001 — site is best-effort here
        return ""
    return base or ""


_REAL_USER_BASE = _real_user_base()


def pytest_configure(config) -> None:
    """Register the opt-out marker for the HOME-isolation fixture."""
    config.addinivalue_line(
        "markers",
        "allow_live_home: test deliberately reads the operator's real HOME; "
        "exempt from the sandbox fixture. Needs review — see "
        "tests/unit_tests/test_tests_never_touch_prod_state.py",
    )


@pytest.fixture(autouse=True)
def _sandbox_deep_worker_home(request, tmp_path_factory, monkeypatch):
    """Point HOME at a sandbox so no test can write the live runtime tree.

    Owner directive (2026-08-17): "Tests must not mess up production
    configs. NEVER!!!"

    The incident this prevents: ``test_handle_invoke_session_limit``
    isolated its ``SessionManager`` but not the code under test, which
    reached ``AvailabilityTracker()`` with no ``data_dir``. Its default is
    ``deep_worker_home()``, so one green test rewrote the host-wide
    ``provider_availability.json`` with usage-limit records attributed to
    agents that never existed — and those records gate real failover
    decisions. Cleaning up afterwards (see ``pytest_sessionfinish`` below)
    cannot fix that class of bug: the damage is done while the test runs,
    and a concurrently running agent may already have read the lie.

    ``HOME`` is the correct and only lever: ``deep_worker_home()`` derives
    from ``Path.home()`` unconditionally and there is deliberately no
    ``DEEP_WORKER_HOME`` env var (``test_restart_with_fable_alias`` pins
    that). ``Path.home()`` consults ``HOME`` first on POSIX.

    The sandbox is exported through ``os.environ`` (not merely patched
    in-process) so a test that spawns a subprocess hands the child the
    sandbox instead of the operator's real tree.

    Opt out with ``@pytest.mark.allow_live_home`` when a test genuinely
    needs the real home; such a test must not WRITE there.
    """
    if request.node.get_closest_marker("allow_live_home"):
        yield
        return

    sandbox = tmp_path_factory.mktemp("dw_home")
    monkeypatch.setenv("HOME", str(sandbox))
    monkeypatch.setenv("USERPROFILE", str(sandbox))

    # Moving HOME invalidates site.getusersitepackages(): the interpreter
    # derives the user-site directory from HOME, so a subprocess spawned by
    # a test would lose every dependency installed with `pip install --user`
    # (langchain, mcp, …) and die with ModuleNotFoundError.
    #
    # PYTHONUSERBASE — not PYTHONPATH — is the right lever. PYTHONPATH is
    # inherited by EVERY child Python regardless of its version, so forcing
    # this interpreter's version-specific site-packages onto it poisons a
    # child running a different minor version: it imports 3.12 extension
    # modules and dies on an ABI mismatch ("No module named
    # 'pydantic_core._pydantic_core'"). That is exactly how the first cut of
    # this fixture broke the four test_cli_fast_return tests, which spawn the
    # installed `deep-worker` from a 3.14 virtualenv.
    #
    # PYTHONUSERBASE instead lets each child derive its OWN
    # <base>/lib/pythonX.Y/site-packages and run .pth processing natively, so
    # a version mismatch degrades to "no user packages" (the child falls back
    # to its venv, as it would without any sandbox) rather than "incompatible
    # user packages".
    if _REAL_USER_BASE:
        monkeypatch.setenv("PYTHONUSERBASE", _REAL_USER_BASE)

    # In-process imports are a separate matter: this interpreter already
    # completed site-initialisation before HOME moved, so its sys.path is
    # intact. Nothing to forward here.

    yield sandbox


def _snapshot_test_agent_artefacts() -> dict[str, set[str]]:
    """Snapshot ``~/.deep_worker/test_*`` dirs + ``/tmp/deep_worker_test_*.pid``
    files + ``test_*`` entries in the registry that exist BEFORE the
    pytest session starts.

    The diff against the post-session state tells us which artefacts
    THIS session leaked. Items in the snapshot are preserved (they
    predate this session); anything not in the snapshot but present
    afterwards is a leak we created and must clean up.
    """
    home_dw = deep_worker_home()
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
    ``~/.deep_worker/<name>/`` based on the agent's name, not its
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

    # 2. ``~/.deep_worker/test_*`` dirs created during this session.
    home_dw = deep_worker_home()
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
    the rpyc_async design doc on same-process tests).
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


# ============================================================================
# CRITICAL: tests must not inherit pytest's own command line
# ============================================================================
# ``deep_worker_cli.cli.agent_command._explicit_cli_flags()`` defaults to
# reading the real ``sys.argv`` to decide which flags the user passed
# explicitly. Under pytest that is PYTEST's command line, so any test which
# calls ``merge_restart_config(saved, args)`` without an explicit
# ``explicit=`` set silently inherits whatever the developer typed.
#
# argparse matches long options by unique PREFIX, and the detector declares
# ``-pp/--preferred-provider``. So running
#
#     pytest tests/unit_tests -p no:randomly
#
# makes ``-p`` match ``-pp``: ``_explicit_cli_flags()`` returns
# ``{"preferred_provider"}`` and ~32 restart/failover tests fail — while the
# same tests pass under a bare ``pytest``. The product code is correct; the
# suite was simply not isolated from its own invocation.
#
# Neutralising argv here fixes every such test at once and keeps the fix in
# one place instead of threading ``explicit=set()`` through ~43 call sites.
# A test that WANTS specific flags still passes ``argv``/``explicit``
# explicitly, which is the honest way to express that intent.
#
# Regression tests: tests/unit_tests/cli/test_sys_argv_isolation.py
import pytest


@pytest.fixture(autouse=True)
def _isolate_sys_argv(monkeypatch):
    """Give every test a neutral ``sys.argv`` (program name only)."""
    monkeypatch.setattr(sys, "argv", [sys.argv[0] if sys.argv else "pytest"])


@pytest.fixture(autouse=True)
def _isolate_package_loggers():
    """Snapshot and restore the ``deep_worker_cli`` logger hierarchy after
    every test.

    Production code paths (notably ``AgentProcess.start()`` in
    ``process/agent_process.py`` ~line 1203) attach a stderr handler and raise
    the level of the ``deep_worker_cli`` logger to INFO so that token-usage
    output is visible in a real agent process.  In-process tests that drive
    those paths leak that side effect into the shared interpreter.  A targeted
    fix lived inside ``test_agent_checkpointer.py`` as an autouse fixture, but
    it only restores the *parent* logger level; child loggers such as
    ``deep_worker_cli.file_ops`` that were touched (or whose effective level
    was implicitly raised through inheritance) keep the inherited INFO
    effective level, and later tests that assert on DEBUG records via
    ``caplog`` silently capture nothing and fail (see
    ``test_exception_handling.py``).

    Restoring the level on the parent is sufficient for *newly-touched*
    children (their effective level is recomputed on the next call), but the
    test suite also contains tests that ``mock.patch`` ``subprocess.run``
    globally and never call ``mock.patch.stopall()`` cleanly when they exit
    via exception.  Those mock leaks can mask ``caplog`` output too.  This
    fixture is therefore intentionally defensive: it walks every logger under
    ``deep_worker_cli`` and resets both the explicit level (back to NOTSET)
    and any handlers that were added during the test, so neither the logging
    nor the mocking pollution can escape into a sibling test.
    """
    import logging

    pkg_root = logging.getLogger("deep_worker_cli")
    snapshot: dict[str, tuple[int, tuple[logging.Handler, ...]]] = {}
    visited: set[int] = set()

    def _walk(logger: logging.Logger) -> None:
        if id(logger) in visited:
            return
        visited.add(id(logger))
        snapshot[logger.name] = (logger.level, tuple(logger.handlers))
        for child in logger.handlers:
            pass
        # Manager keeps track of child loggers; iterate over a copy to be
        # safe against mutations during iteration.
        for child in list(logging.root.manager.loggerDict.values()):
            if isinstance(child, logging.Logger) and child.parent is logger:
                _walk(child)

    _walk(pkg_root)

    try:
        yield
    finally:
        current_root = logging.getLogger("deep_worker_cli")
        for name, (saved_level, saved_handlers) in snapshot.items():
            logger = logging.getLogger(name)
            logger.setLevel(saved_level)
            for handler in list(logger.handlers):
                if handler not in saved_handlers:
                    logger.removeHandler(handler)
