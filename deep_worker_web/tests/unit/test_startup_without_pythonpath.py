"""Regression guard: the web server must start without an ambient PYTHONPATH.

Background
==========
The web server would not start. ``run.py`` had no ``sys.path`` bootstrap, but
``app/routers/agents.py`` imports ``deep_worker_cli.model_predicate`` at module
scope. Launching ``run.py`` without a hand-crafted ``PYTHONPATH`` therefore died
at import time::

    File "app/routers/agents.py", line 48, in <module>
        from deep_worker_cli.model_predicate import (
    ModuleNotFoundError: No module named 'deep_worker_cli'

That made startup depend on the caller's environment, which nothing guarantees.
The in-process restart helper (``app/routers/server.py``) is precisely such a
caller: it re-spawns ``run.py`` with no ``env=`` and with stdout/stderr at
DEVNULL, *after* the old process has already called ``os._exit(0)``. So a
restart could leave no server running at all and no trace of why.

A server that cannot find its own modules is unacceptable, so these tests pin
the invariant from both ends:

  * ``run.py`` bootstraps ``sys.path`` from its own location, and that bootstrap
    happens BEFORE any ``app.*`` import.
  * With ``PYTHONPATH`` scrubbed from the environment, a real subprocess can
    import ``run.py`` and then ``app.main`` — the actual failing scenario.
  * The restart helper neither drops the environment nor sends startup output
    to DEVNULL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_THIS_FILE = Path(__file__).resolve()
_WEB_ROOT = _THIS_FILE.parent.parent.parent  # → deep_worker_web/

RUN_PY = _WEB_ROOT / "run.py"
SERVER_ROUTER = _WEB_ROOT / "app" / "routers" / "server.py"

# Import of app.main pulls in fastapi/uvicorn and agent-framework deps. If the
# test environment lacks them the subprocess checks cannot distinguish "our
# bootstrap is broken" from "deps absent", so they are skipped instead.
_DEPS = ("fastapi", "uvicorn", "pydantic_settings")


def _deps_available() -> bool:
    """Check deps in a subprocess under the SAME scrubbed env the tests use.

    Checking in-process would consult the parent's PYTHONPATH and could report
    deps as present when they are unreachable in the scrubbed child.
    """
    probe = (
        "import importlib.util, sys\n"
        f"names = {list(_DEPS)!r}\n"
        "missing = [n for n in names if importlib.util.find_spec(n) is None]\n"
        "print('MISSING:' + ','.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp",
            env=_scrubbed_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "MISSING:\n" in result.stdout or result.stdout.strip() == "MISSING:"


def _scrubbed_env() -> dict[str, str]:
    """Environment with the inherited import path removed.

    This is the whole point: the server must not rely on the launcher.

    Only import-path variables are removed. User site-packages stays enabled:
    third-party deps (fastapi, uvicorn) legitimately live there, and the bug
    was specifically a PYTHONPATH dependency for the project's OWN packages.
    """
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
    }


def test_run_py_exists() -> None:
    assert RUN_PY.exists(), f"not found: {RUN_PY}"


def test_run_py_has_sys_path_bootstrap() -> None:
    """``run.py`` must contain an explicit sys.path bootstrap."""
    text = RUN_PY.read_text()
    assert "_bootstrap_sys_path" in text, (
        "run.py has no _bootstrap_sys_path(). Without it the server can only "
        "start when the caller supplies a correct PYTHONPATH, and any "
        "env-dropping launcher (notably the restart helper) kills the server "
        "with ModuleNotFoundError: deep_worker_cli."
    )


def test_bootstrap_runs_before_app_imports() -> None:
    """Ordering is load-bearing.

    ``app.routers.agents`` imports ``deep_worker_cli`` at module scope, so the
    bootstrap is useless if it runs after ``from app.config import settings``.
    """
    lines = RUN_PY.read_text().splitlines()

    bootstrap_call = next(
        (
            i
            for i, line in enumerate(lines)
            if "_bootstrap_sys_path()" in line and not line.lstrip().startswith("def ")
        ),
        None,
    )
    app_import = next(
        (i for i, line in enumerate(lines) if line.startswith("from app.")),
        None,
    )

    assert bootstrap_call is not None, "no call to _bootstrap_sys_path() in run.py"
    assert app_import is not None, "expected a top-level 'from app.' import in run.py"
    assert bootstrap_call < app_import, (
        f"_bootstrap_sys_path() is called on line {bootstrap_call + 1} but "
        f"'from app.' import is on line {app_import + 1}. The bootstrap MUST "
        f"precede any app import, since app.routers.agents needs "
        f"deep_worker_cli at import time."
    )


def test_bootstrap_resolves_paths_relative_to_own_tree() -> None:
    """Paths must derive from ``__file__``, never a hard-coded checkout.

    Mirrors test_tree_path_resolution.py: a dev server must load dev packages.
    """
    text = RUN_PY.read_text()
    for bad in (
        "/prod-tree/deep_worker/deep_worker_cli",
        "/dev-tree/deep_worker/deep_worker_cli",
    ):
        assert bad not in text, (
            f"run.py hard-codes {bad!r}. It must resolve sibling packages "
            f"relative to __file__ so each checkout loads its own tree."
        )


@pytest.mark.skipif(not _deps_available(), reason="web deps not installed")
def test_run_py_bootstrap_works_with_pythonpath_scrubbed() -> None:
    """The real regression: import run.py with NO PYTHONPATH, then app.main.

    Runs in a genuine subprocess with a scrubbed environment — no mocks, no
    monkeypatching of sys.path — because the bug was an environment dependency.
    """
    probe = (
        "import importlib.util, sys\n"
        "assert 'PYTHONPATH' not in __import__('os').environ\n"
        f"spec = importlib.util.spec_from_file_location('runmod', r'{RUN_PY}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        # run.py must have made the tree importable purely on its own.
        "importlib.import_module('deep_worker_cli.model_predicate')\n"
        "importlib.import_module('app.main')\n"
        "print('STARTUP_IMPORTS_OK')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        # cwd deliberately NOT the web root: a correct bootstrap must not
        # depend on being launched from its own directory either.
        cwd="/tmp",
        env=_scrubbed_env(),
    )

    assert "STARTUP_IMPORTS_OK" in result.stdout, (
        "run.py could not bootstrap its own imports without PYTHONPATH.\n"
        f"exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(not _deps_available(), reason="web deps not installed")
def test_app_main_import_fails_without_bootstrap_control() -> None:
    """Control case proving the test above is meaningful.

    Importing ``app.main`` with a scrubbed env and WITHOUT executing run.py's
    bootstrap must still fail. If this ever passes, the ambient environment is
    supplying the path and the positive test proves nothing.
    """
    probe = (
        "import importlib, sys\n"
        f"sys.path.insert(0, r'{_WEB_ROOT}')\n"
        "try:\n"
        "    importlib.import_module('app.routers.agents')\n"
        "    print('UNEXPECTEDLY_IMPORTED')\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('EXPECTED_FAILURE:' + exc.name)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/tmp",
        env=_scrubbed_env(),
    )

    if "UNEXPECTEDLY_IMPORTED" in result.stdout:
        pytest.skip(
            "deep_worker_cli is importable from the ambient environment "
            "(site-packages or .pth), so the scrubbed-env control cannot run."
        )

    assert "EXPECTED_FAILURE:deep_worker_cli" in result.stdout, (
        "Control case did not reproduce the original failure mode.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_restart_helper_does_not_discard_environment() -> None:
    """The restart helper must not spawn the replacement with a dropped env."""
    text = SERVER_ROUTER.read_text()
    assert "env={}" not in text and "env=None" not in text, (
        "The restart helper explicitly clears the environment. Even though "
        "run.py now self-bootstraps, wiping the env is what made a failed "
        "restart invisible."
    )


def test_restart_helper_records_startup_failures() -> None:
    """A failed restart must leave a trace.

    The old helper sent the replacement's stdout AND stderr to DEVNULL, so a
    startup crash produced no diagnostic at all — and the old server had
    already exited, leaving nothing serving the port.
    """
    text = SERVER_ROUTER.read_text()

    assert "restart_startup.log" in text, (
        "The restart helper does not capture the replacement server's startup "
        "output to a log file. A restart that dies during import must not "
        "vanish silently."
    )
    assert "proc.poll()" in text, (
        "The restart helper does not check whether the replacement server "
        "survived. Without this a restart can silently leave no server."
    )
