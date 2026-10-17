"""``deep-worker web`` must start the web UI from an installed distribution.

Before this entry existed the web server was reachable only as ``python
run.py`` from a git checkout, so a user who did ``pip install
deep-worker`` had no way to start it at all — ``run.py`` is not part of
the distribution.

**These tests drive the REAL entry points**, both of them, because there
are two and they are different programs:

* the ``deep-worker`` **shell wrapper** — what a bare ``deep-worker`` on
  this host resolves to. It execs ``python3 -m
  deep_worker_cli.cli.agent_command`` directly and NEVER touches
  ``router.py``, so a test that drove the router function alone would
  grade an entry this host does not execute. That exact mistake is
  documented in ``lsp/test_bridge_thin_entry.py``: a guard on the
  declared console script passed while the live wrapper still loaded
  torch.
* the ``[project.scripts]`` **console script** — what a pip install
  gives, where the wrapper does not exist.

The ``app`` package name is the load-bearing detail
---------------------------------------------------
``uvicorn`` is handed the import string ``"app.main:app"`` and ~20
modules under ``web/app`` import one another absolutely, so ``app`` must
resolve as a TOP-LEVEL name in both layouts. The subcommand achieves
that by putting the ``web/`` directory on ``sys.path`` — the same thing
``run.py:_bootstrap_sys_path`` does from source. ``test_mutation_*``
below is the proof that this line is what makes it work rather than an
ambient ``PYTHONPATH``: with the line removed, the import fails with
``ModuleNotFoundError: No module named 'app'``.
"""
from __future__ import annotations

import os
import shutil
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The physical wrapper inside THIS tree — the real entry on this host.
_TREE_WRAPPER = _REPO_ROOT / "deep-worker"

#: Source layout of the web application, a sibling of ``deep_worker/``.
_WEB_SOURCE = _REPO_ROOT.parents[1] / "deep_worker_web"

_RUN_TIMEOUT_S = 300


def _probe_env() -> dict[str, str]:
    """Environment for an entry-point launch.

    ``PYTHONPATH`` is cleared deliberately: the wrapper is responsible
    for setting it, and inheriting the ambient one would mask an entry
    that fails to make ``app`` importable — which is the whole claim
    under test. ``DEEP_WORKER_*`` selectors are cleared so the layout is
    chosen by the test and not by the session that runs it.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "DEEP_WORKER_DEV_TREE", "DEEP_WORKER_TREE_ROOT"}
    }
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _resolve_wrapper() -> str:
    """This tree's wrapper, falling back to PATH (e.g. on CI)."""
    if _TREE_WRAPPER.is_file():
        return str(_TREE_WRAPPER)
    found = shutil.which("deep-worker")
    if found:
        return found
    pytest.skip("no deep-worker wrapper available to exercise")


def test_wrapper_routes_web_to_the_subcommand_parser():
    """``deep-worker web --help`` must reach the web parser, not the agent one.

    This is the narrowest observable proof that the wrapper's branch
    fires: the help text comes from ``_dispatch_web``'s own
    ``ArgumentParser(prog="deep-worker web")``. Before the branch
    existed, ``web`` was an agent NAME and the invocation went to
    ``agent_command``, whose parser prints a different usage line.

    ``--help`` is used rather than a real start so the test binds no
    port and cannot collide with the long-lived servers on this host.
    """
    proc = subprocess.run(
        [_resolve_wrapper(), "web", "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=_probe_env(),
        cwd=str(_REPO_ROOT),
        timeout=_RUN_TIMEOUT_S,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"rc={proc.returncode}: {combined[-600:]!r}"
    assert "usage: deep-worker web" in combined, (
        "the wrapper did not route `web` to the web subcommand parser; "
        f"got: {combined[-600:]!r}"
    )
    # The flags are part of the contract a user sees.
    assert "--port" in combined and "--reload" in combined, combined[-400:]


def test_wrapper_web_branch_sits_after_dev_and_pythonpath():
    """Placement inside the wrapper is load-bearing, so pin it.

    Two orderings must hold, and neither is cosmetic:

    * AFTER the ``--dev`` dispatch — otherwise ``deep-worker --dev web``
      would serve the PRODUCTION tree's web code while the developer
      believes they are testing the dev tree;
    * AFTER ``export PYTHONPATH`` — otherwise ``python3 -m
      deep_worker_cli.cli.router`` cannot resolve ``deep_worker_cli`` at
      all, because the package is not installed on this host
      (``pip install -e .`` is forbidden; see ``deep_worker/CLAUDE.md``
      §1).

    Asserted on the wrapper's text because the ordering is a property of
    the shell script, not of any Python object it later runs.
    """
    text = _TREE_WRAPPER.read_text() if _TREE_WRAPPER.is_file() else pytest.skip(
        "tree wrapper not present"
    )
    dev_branch = text.index('if [[ $dev_requested -eq 1 ]]')
    pythonpath = text.index('export PYTHONPATH=')
    web_branch = text.index('"${filtered_args[0]}" == "web"')

    assert dev_branch < web_branch, (
        "the web branch precedes the --dev dispatch, so `--dev web` would "
        "serve the production tree's code"
    )
    assert pythonpath < web_branch, (
        "the web branch precedes `export PYTHONPATH`, so the router module "
        "would not be importable on a host where the package is not installed"
    )


def test_wrapper_does_not_divert_other_subcommands():
    """Intercepting ``web`` must not change where anything else goes.

    The wrapper launches every agent on this host, so a branch that is
    keyed too loosely would be a far worse defect than the one it fixes.
    An unrecognised flag must still be rejected by ``agent_command``'s
    own parser, which proves the arguments reached it untouched.
    """
    proc = subprocess.run(
        [_resolve_wrapper(), "--definitely-not-a-flag"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=_probe_env(),
        cwd=str(_REPO_ROOT),
        timeout=_RUN_TIMEOUT_S,
    )
    combined = proc.stdout + proc.stderr
    assert "unrecognized arguments" in combined, (
        f"unknown flag did not reach the normal parser: {combined[-400:]!r}"
    )
    assert proc.returncode == 2, (
        f"argparse exit code changed: expected 2, got {proc.returncode}"
    )


def _import_app_probe(web_root: Path, *, add_to_path: bool) -> subprocess.CompletedProcess:
    """Resolve ``app.main`` the way the subcommand does, in a clean process.

    ``cwd`` is pinned OUTSIDE any source tree and ``PYTHONPATH`` is
    absent, so the only way ``app`` can resolve is the ``sys.path`` entry
    under test. Pinning cwd is not pedantry: ``python -c`` puts the
    working directory on ``sys.path[0]``, which twice made an earlier
    probe pass by importing the real checkout it was supposed to be
    isolated from.
    """
    # The sibling packages are added in BOTH arms. ``app.services``
    # imports ``deep_worker_cli.storage_paths`` at module scope, so
    # without them the import dies on the sibling — and an earlier
    # version of this probe skipped on that, meaning the positive test
    # never actually asserted that ``app`` resolves. The variable under
    # test is only the ``web/`` entry, so everything else must be held
    # equal between the two arms.
    siblings = [
        str(_REPO_ROOT),
        str(_REPO_ROOT.parent / "deepagents"),
        str(_REPO_ROOT.parent),
    ]
    src = (
        "import sys\n"
        f"for _p in {siblings!r}:\n"
        "    sys.path.insert(0, _p)\n"
        f"if {add_to_path!r}:\n"
        f"    sys.path.insert(0, {str(web_root)!r})\n"
        "import app.main\n"
        "print('OK', app.main.__file__)\n"
    )
    env = _probe_env()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        env=env,
        cwd="/tmp",
        timeout=_RUN_TIMEOUT_S,
    )


def test_app_resolves_top_level_once_web_root_is_on_sys_path():
    """``app.main`` must import once ``web/`` is on ``sys.path``.

    This is the positive half of the pair. It is what lets ~20 absolute
    ``from app.services ... import`` statements and uvicorn's
    ``"app.main:app"`` string keep working unchanged in both layouts,
    which is why the task did not require rewriting them to relative
    imports.
    """
    if not (_WEB_SOURCE / "app" / "main.py").is_file():
        pytest.skip(f"web source tree not present at {_WEB_SOURCE}")
    # No skip-on-failure branch here. An earlier version skipped when the
    # import died on the sibling ``deep_worker_cli`` — but ``app`` had in
    # fact resolved by then (the traceback reached
    # ``app/services/__init__.py``), so the skip hid the fact that this
    # test asserted nothing at all. The probe now supplies the siblings
    # in both arms instead.
    env_ok = _import_app_probe(_WEB_SOURCE, add_to_path=True)
    assert env_ok.returncode == 0, (
        f"app.main did not import with web/ on sys.path: {env_ok.stderr[-600:]!r}"
    )
    assert "OK" in env_ok.stdout, env_ok.stdout[-300:]


def test_mutation_app_does_not_resolve_without_the_sys_path_entry():
    """Negative control: without the entry, the import MUST fail.

    Without this half the positive test proves nothing — it would pass
    just as happily if ``app`` were resolving from the ambient
    environment or from the working directory. The failure is asserted
    BY CONTENT (``No module named 'app'``) rather than by a non-zero
    exit code, so an unrelated crash cannot be mistaken for the
    mechanism under test.
    """
    if not (_WEB_SOURCE / "app" / "main.py").is_file():
        pytest.skip(f"web source tree not present at {_WEB_SOURCE}")
    result = _import_app_probe(_WEB_SOURCE, add_to_path=False)
    assert result.returncode != 0, (
        "app.main imported with NO sys.path entry — something else is "
        f"making it resolvable, so the positive test proves nothing: {result.stdout!r}"
    )
    assert "No module named 'app'" in result.stderr, (
        "the import failed for a different reason than the missing path "
        f"entry, so this is not a control for that mechanism: {result.stderr[-600:]!r}"
    )


def _free_port() -> int:
    """A port the kernel just confirmed is free.

    Bound and released rather than picked from a range, so this cannot
    collide with the long-lived servers on this host (7887, 7888) nor
    with a concurrent run of this same test.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def test_real_wrapper_start_makes_app_importable_in_the_server_process():
    wrapper = _resolve_wrapper()
    if not (_WEB_SOURCE / "app" / "main.py").is_file():
        pytest.skip(f"web source tree not present at {_WEB_SOURCE}")
    port = _free_port()
    proc = subprocess.Popen(
        [wrapper, "web", "--foreground", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=_probe_env(), cwd=str(_REPO_ROOT),
    )
    output = b""
    try:
        assert proc.stdout is not None
        deadline = time.monotonic() + 90.0
        while b"Uvicorn running on" not in output:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                pytest.fail(f"server startup timed out: {output[-1500:]!r}")
            data = os.read(proc.stdout.fileno(), 65536)
            if not data:
                text = output.decode(errors="replace")
                assert "No module named 'app'" not in text, (
                    "the real entry could not import `app`: " + text[-1500:]
                )
                for dep in ("uvicorn", "fastapi", "pydantic_settings", "anyio"):
                    if f"No module named '{dep}'" in text:
                        pytest.skip(f"server process has no {dep!r}")
                pytest.fail(f"server exited before binding: {text[-1500:]}")
            output += data
        assert _port_is_open(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)



def test_router_dispatches_web_before_importing_the_agent_stack():
    """``_dispatch_web`` must be consulted before the agent stack import.

    This is the console-script half: a pip install has no wrapper, so
    ``deep_worker_main`` is the real entry there. Ordering is asserted on
    the function's own source — the ``agent_command`` import must not
    appear before the web dispatch — because importing that module is
    irreversible once done, so a runtime check cannot observe "it was
    not imported yet".
    """
    import inspect

    from deep_worker_cli.cli import router

    src = inspect.getsource(router.deep_worker_main)
    web_call = src.index("_dispatch_web")
    agent_import = src.index("agent_command")
    assert web_call < agent_import, (
        "deep_worker_main imports the agent stack before dispatching `web`, "
        "which pays the full langchain/torch cost for a server that uses "
        "none of it"
    )


def test_web_dispatch_declines_every_other_argv():
    """Only a leading ``web`` token may be captured.

    ``_dispatch_web`` returning anything but ``None`` for these argvs
    would divert real invocations: an agent start, the bridge, a bare
    call. Checked directly on the predicate because the consequence of
    getting it wrong is silent misrouting rather than an error.
    """
    from deep_worker_cli.cli.router import _dispatch_web

    for argv in ([], ["lsp", "bridge"], ["myagent"], ["--restart", "web"], ["webhook"]):
        assert _dispatch_web(argv) is None, (
            f"_dispatch_web captured argv it must decline: {argv!r}"
        )


def test_locate_web_root_finds_the_source_layout():
    """The resolver must find ``<repo>/deep_worker_web`` from a checkout.

    ``_locate_web_root`` is what the subcommand uses before ``app`` is
    importable, so it cannot use ``app.paths`` and is the one place that
    still reasons about layout directly. Its answer must carry both the
    code and the assets, since the server mounts ``static/`` from it.
    """
    from deep_worker_cli.cli.router import _locate_web_root

    found = _locate_web_root()
    assert found is not None, "web root not located from a source checkout"
    found_path = Path(str(found))
    assert (found_path / "app" / "main.py").is_file(), found_path
    assert (found_path / "static" / "index.html").is_file(), found_path
