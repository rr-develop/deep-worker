"""What a user gets from ``pip install deep-worker`` — nothing else.

Every other entry test on this tree can pass while the shipped package is
broken, because they all reach code through this checkout: the wrapper
``deep-worker`` script, a ``PYTHONPATH`` that already contains the source,
or an in-process import of ``router``. A pip user has none of those. The
defects this file pins were reported from a host with 0.0.21 installed
from PyPI and no sources at all:

* ``deep-worker --version`` exited 2 — the flag existed nowhere, so it
  fell through to the agent parser.
* ``deep-worker --help`` never mentioned the web UI or the version: the
  epilog in ``agent_command.create_parser`` lists entries by hand and the
  web server is dispatched a layer above it, in ``router``.
* the web UI had no flag at all; it was reachable only as the positional
  ``deep-worker web``, which collides with an agent of that name.

The interface is FLAGS (``--web``, ``--version``), decided by the owner
after the first round: the first positional of ``deep-worker`` is an agent
name, so a positional subcommand can never be disambiguated from an agent
called ``web`` or ``version``. Those agent names must keep working.

So these tests build the wheel and run the installed console script in a
throwaway venv under ``/tmp`` with its own ``HOME``. That is the only
configuration that grades the artifact a user receives. Repository policy
forbids installing this project on the host and forbids a ``.venv``
inside the tree (``deep_worker/CLAUDE.md`` §1, §2); a venv in a temporary
directory purely to verify the distribution is explicitly allowed there,
and every fixture below removes its own.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

_CLI_ROOT = Path(__file__).resolve().parents[2]

#: Building a wheel and installing it costs ~1-2 min; the whole module
#: shares one venv.
_BUILD_TIMEOUT_S = 900
_RUN_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _is_reaper(pid: int) -> bool:
    """True when ``pid`` adopts orphans rather than merely being an ancestor.

    pid 1 always qualifies. A systemd user session also qualifies: it sets
    ``PR_SET_CHILD_SUBREAPER``, so a daemon that correctly detaches lands on
    ``systemd --user`` instead of pid 1. ``/proc/<pid>/status`` exposes that
    bit via ``ChildSubreaper`` on kernels that report it; where it is absent
    we fall back to "a direct child of pid 1 whose own session leader it is",
    which is what a session manager looks like.

    Asserting the parent is a reaper — not "some pid outside our ancestry" —
    is the portable form: the launching shell is never a reaper, while every
    host's adopter is one.
    """
    if pid == 1:
        return True
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:  # pragma: no cover - race with a dying process
        return False
    for line in status.splitlines():
        if line.startswith("ChildSubreaper:"):
            return line.split()[1] != "0"
    return int(stat[1]) == 1 and int(stat[3]) == pid


@pytest.fixture(scope="module")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Build the wheel, install it into a clean venv, yield its launcher.

    ``--no-deps`` keeps this hermetic and fast: the commands under test
    (``--help``, ``version``) must not need the runtime stack, and a test
    that silently downloaded all of langchain would be measuring the
    network. The web acceptance test installs what it needs on top.
    """
    if shutil.which("git") is None:  # pragma: no cover - defensive
        pytest.skip("git unavailable")
    root = tmp_path_factory.mktemp("installed-cli")
    home = root / "home"
    home.mkdir()
    dist = root / "dist"

    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=str(_CLI_ROOT),
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT_S,
    )
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable: {build.stderr[-600:]}")
    wheels = sorted(dist.glob("deep_worker-*.whl"))
    assert wheels, f"no wheel produced in {dist}"

    env_dir = root / "venv"
    venv.create(env_dir, with_pip=True, clear=True)
    python = env_dir / "bin" / "python"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "-q", str(wheels[-1])],
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT_S,
    )
    assert install.returncode == 0, install.stderr[-2000:]

    launcher = env_dir / "bin" / "deep-worker"
    assert launcher.is_file(), (
        "the wheel declares no `deep-worker` console script — a pip user "
        "would have no command at all"
    )
    return {
        "launcher": str(launcher),
        "python": str(python),
        "home": str(home),
        "wheel": str(wheels[-1]),
        "venv": str(env_dir),
    }


def _run(installed_cli: dict[str, str], *argv: str) -> subprocess.CompletedProcess[str]:
    """Invoke the installed console script the way a shell would.

    The environment is built from scratch rather than inherited: with the
    session's ``PYTHONPATH`` leaking in, the installed package would
    import this checkout and the test would grade the sources again.
    """
    return subprocess.run(
        [installed_cli["launcher"], *argv],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        cwd=installed_cli["home"],
        env={
            "HOME": installed_cli["home"],
            "PATH": f"{Path(installed_cli['launcher']).parent}:/usr/bin:/bin",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_the_version(
    installed_cli: dict[str, str], flag: str
) -> None:
    """``deep-worker --version`` must print the version and exit 0.

    It exited 2 on the installed 0.0.21 — the flag existed nowhere, so it
    reached the agent parser as an unknown option.

    The venv is installed with ``--no-deps`` on purpose, which makes the
    second assertion meaningful: if the flag fell through to
    ``agent_command``, the module-level yaml and langchain imports would raise
    instead of a version being printed.
    """
    proc = _run(installed_cli, flag)
    combined = proc.stdout + proc.stderr
    assert "ModuleNotFoundError" not in combined, (
        f"`deep-worker {flag}` fell through to the agent stack, which a "
        f"version report must never import:\n{combined[-800:]}"
    )
    assert proc.returncode == 0, combined[:800]
    from deep_worker_cli._version import __version__

    assert __version__ in proc.stdout, (
        f"expected version {__version__!r} in stdout, got:\n{proc.stdout[:400]}"
    )


def test_positional_version_is_still_an_agent_name(
    installed_cli: dict[str, str],
) -> None:
    """``deep-worker version`` must keep meaning "the agent named version".

    This is the owner's constraint on the whole design: the first
    positional is an agent name, always. An earlier round made ``version``
    a subcommand, which silently stole that name from anyone who had an
    agent called it. The flag form carries the new behaviour instead, so
    this asserts the POSITIONAL did not change meaning.

    With ``--no-deps`` the agent cannot actually boot, so the proof is
    that it tried: either the launch notice or the agent stack's own
    import error. What must NOT appear is a version number.
    """
    proc = _run(installed_cli, "version")
    combined = proc.stdout + proc.stderr
    from deep_worker_cli._version import __version__

    assert not proc.stdout.strip().startswith(f"deep-worker {__version__}"), (
        "`deep-worker version` printed the version, so the positional was "
        "captured as a subcommand — an agent named 'version' is now "
        f"unreachable:\n{combined[:600]}"
    )
    assert "starting in background" in combined or "ModuleNotFoundError" in combined, (
        "`deep-worker version` neither started the agent nor tried to: the "
        f"positional no longer routes to the agent launcher:\n{combined[:800]}"
    )


def test_web_flag_is_documented_in_help(installed_cli: dict[str, str]) -> None:
    """``--web`` must be a real, self-describing flag.

    Named for what it checks. It previously claimed to prove that the
    POSITIONAL ``web`` still named an agent while only ever invoking
    ``--web --help`` — a promise in the name that the body never made,
    which is why it stayed green through the very behaviour it was
    supposed to catch. The positional's actual behaviour is pinned by
    ``test_bare_positional_web_currently_serves_the_ui`` instead.
    """
    proc = _run(installed_cli, "--web", "--help")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[:600]
    assert "--web" in proc.stdout, (
        f"`--web --help` does not describe the flag:\n{proc.stdout[:600]}"
    )


def test_positional_web_is_an_agent_name(installed_cli: dict[str, str]) -> None:
    """``deep-worker web`` must start the AGENT named "web".

    The pre-flag spelling made this positional serve the UI, so a user
    whose agent was called "web" silently got a server on the default
    port — 7887 — instead of their agent. Removing that branch is what
    this pins; ``--web`` is the only spelling that reaches the server.

    The oracle is WHICH dispatcher the call lands in, read off the
    traceback frame rather than the printed text. Under ``--no-deps``
    neither route can complete — the web path dies on ``import
    uvicorn``, the agent path on ``import yaml`` — so both surface a
    ``ModuleNotFoundError`` and the message alone cannot tell them
    apart. ``_dispatch_web`` appears in the frames only if the
    positional was diverted to the web dispatcher.
    """
    proc = _run(installed_cli, "web")
    combined = proc.stdout + proc.stderr

    assert "_dispatch_web" not in combined, (
        "`deep-worker web` was diverted to the web dispatcher: the "
        "positional is a subcommand again, so an agent named 'web' is "
        f"unreachable:\n{combined[:600]}"
    )
    assert "starting in background" in combined or "ModuleNotFoundError" in combined, (
        "`deep-worker web` neither started the agent nor tried to: the "
        f"positional no longer routes to the agent launcher:\n{combined[:800]}"
    )


def test_web_flag_keeps_its_usage_line(installed_cli: dict[str, str]) -> None:
    """``--web --help`` must answer from the web parser, with its own prog.

    Guards the diagnostic surface: the usage line is what tells a user
    which spelling they are being helped with.
    """
    flag = _run(installed_cli, "--web", "--help")
    assert flag.returncode == 0, (flag.stdout + flag.stderr)[:600]
    assert "usage: deep-worker --web" in flag.stdout, (
        f"flag form lost its usage line:\n{flag.stdout[:600]}"
    )


def test_version_comes_from_installed_metadata(installed_cli: dict[str, str]) -> None:
    """The printed version must match the distribution's own metadata.

    Reading ``_version.py`` would still pass if the wheel were built from
    a stale tree, so the oracle here is ``importlib.metadata``, i.e. what
    ``pip show deep-worker`` reports for the thing actually installed.
    """
    meta = subprocess.run(
        [
            installed_cli["python"],
            "-c",
            "import importlib.metadata as m; print(m.version('deep-worker'))",
        ],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
    )
    assert meta.returncode == 0, meta.stderr[:400]
    installed_version = meta.stdout.strip()
    proc = _run(installed_cli, "--version")
    assert installed_version in proc.stdout, (
        f"`--version` printed {proc.stdout.strip()!r}, but the installed "
        f"distribution is {installed_version!r}"
    )


def test_help_advertises_web_and_version(installed_cli: dict[str, str]) -> None:
    """``--help`` must name ``--web`` and ``--version``.

    Undocumented functionality does not exist for the user, which is the
    whole reason this is a test and not a docs note. The tokens searched
    for are the FLAGS, so ``--enable-web-search`` and ``--to-version``
    cannot satisfy it — on the broken build a naive grep for "web" found
    seven such lines and no way to start the server.
    """
    proc = _run(installed_cli, "--help")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[:800]
    for token in ("--web", "--version"):
        assert token in proc.stdout, (
            f"`{token}` is absent from --help, so a user cannot discover "
            f"it:\n{proc.stdout[:1200]}"
        )
    assert "--web --port" in proc.stdout or "--web --status" in proc.stdout, (
        "--help shows no example of combining --web with its options:\n"
        f"{proc.stdout[-1200:]}"
    )


def test_help_does_not_need_the_agent_runtime(installed_cli: dict[str, str]) -> None:
    """Help must work in a minimal install.

    The venv here deliberately has ``--no-deps``, so reaching
    ``agent_command`` (which imports yaml and langchain at module level) dies
    with ``ModuleNotFoundError``. Help is exactly what a user runs when
    something is already broken, so it must not depend on the runtime.
    """
    proc = _run(installed_cli, "--help")
    combined = proc.stdout + proc.stderr
    # A runtime complaint on stderr is fine and even useful; what must not
    # happen is help FAILING. So the contract is: exit 0, and a usable
    # listing on stdout — `deep-worker --help | grep` has to keep working.
    assert proc.returncode == 0, (
        "`--help` failed in an install whose runtime is incomplete; help is "
        f"what a user runs when things are already broken:\n{combined[-900:]}"
    )
    assert "usage: deep-worker" in proc.stdout, (
        f"no usage line on stdout:\n{proc.stdout[:600]}"
    )
    assert "Traceback" not in proc.stdout, (
        f"a traceback leaked into the help listing:\n{proc.stdout[:600]}"
    )
    for token in ("--web", "--version"):
        assert token in proc.stdout, (
            f"degraded help omits `{token}`, which needs no runtime:\n"
            f"{proc.stdout[:900]}"
        )


def test_positional_help_is_still_an_agent_name(
    installed_cli: dict[str, str],
) -> None:
    """``deep-worker help`` must stay an agent name too.

    Same rule as ``web`` and ``version``: positionals are agent names.
    ``--help`` is the documented spelling and is covered above.
    """
    proc = _run(installed_cli, "help")
    combined = proc.stdout + proc.stderr
    assert "--web" not in proc.stdout, (
        "`deep-worker help` printed the help listing, so the positional was "
        f"captured as a subcommand:\n{combined[:600]}"
    )
    assert "starting in background" in combined or "ModuleNotFoundError" in combined, (
        f"`deep-worker help` no longer routes to the agent launcher:\n{combined[:800]}"
    )


def test_web_help_is_reachable_and_documents_the_controls(
    installed_cli: dict[str, str],
) -> None:
    """``deep-worker --web --help`` must describe port/stop/status."""
    proc = _run(installed_cli, "--web", "--help")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[:800]
    assert "deep-worker --web" in proc.stdout, proc.stdout[:400]
    for token in ("--port", "--stop", "--status", "--foreground"):
        assert token in proc.stdout, (
            f"`{token}` missing from `--web --help`:\n{proc.stdout[:800]}"
        )


def test_help_explains_that_positionals_are_agent_names(
    installed_cli: dict[str, str],
) -> None:
    """Help must make the positional-vs-flag split readable.

    The flags exist precisely so that an agent called ``web`` keeps
    working, and a user needs to be able to learn that from ``--help``
    rather than by experiment.
    """
    proc = _run(installed_cli, "--help")
    assert "agent NAME" in proc.stdout or "agent name" in proc.stdout.lower(), (
        "help never says a positional is an agent name:\n"
        f"{proc.stdout[-1500:]}"
    )


def test_web_serves_health_detached_from_the_terminal(
    installed_cli: dict[str, str],
) -> None:
    """Acceptance: the server a pip user starts answers and outlives the shell.

    No ``@pytest.mark.timeout``: ``pytest-timeout`` is a dev extra that
    CI does not install, and ``--strict-markers`` turns the unknown
    marker into a COLLECTION error that takes the whole file down. The
    other users of that marker all live under ``integration_tests``,
    which CI deselects, so this file was the first to hit it. Every
    subprocess below is bounded by its own ``timeout=`` instead, which
    needs no plugin.

    This installs the real runtime dependencies into the same venv — the
    point of the check is that ``pip install deep-worker`` alone carries
    the web stack, with no extras — then starts the server through the
    console script and asserts three things the owner asked for: it
    binds, ``/health`` answers, and the process is detached (reparented
    to init, no controlling TTY) so closing the terminal cannot kill it.
    """
    deps = subprocess.run(
        [
            installed_cli["python"],
            "-m",
            "pip",
            "install",
            "-q",
            installed_cli["wheel"],
        ],
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT_S,
    )
    if deps.returncode != 0:
        pytest.skip(f"runtime deps unavailable offline: {deps.stderr[-500:]}")

    port = _free_port()
    started = _run(installed_cli, "--web", "--port", str(port))
    combined = started.stdout + started.stderr
    assert started.returncode == 0, f"web failed to start:\n{combined[:1500]}"

    deadline = time.monotonic() + 120
    body = ""
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(
                    b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    b"Connection: close\r\n\r\n"
                )
                chunks = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                body = b"".join(chunks).decode("utf-8", "replace")
            if "200" in body.split("\r\n", 1)[0]:
                break
        except OSError:
            time.sleep(1)
    else:
        pytest.fail(f"/health never answered on port {port}\nstart said:\n{combined}")

    state = Path(installed_cli["home"]) / ".deep_worker" / "services_shared" / "web"
    record = json.loads((state / f"{port}.json").read_text())
    pid = int(record["pid"])
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        # Reparented away from the launching shell. The reaper is USUALLY pid 1,
        # but under a systemd user session it is ``systemd --user``, which sets
        # PR_SET_CHILD_SUBREAPER and adopts the session's orphans itself — so
        # asserting literally ``PPID == 1`` fails on a correctly detached
        # server (measured: PPID 1111 = ``systemd --user``, itself PPid 1).
        # What must hold is that the parent is not this test's process tree:
        # anything rooted at pid 1 without passing through us survives the
        # terminal, which is the property the owner asked for.
        ppid = int(stat[1])
        assert _is_reaper(ppid), (
            f"web server pid {pid} has PPID {ppid}, which is not a reaper, "
            "so it is still a child of the launching shell and dies with "
            "the terminal"
        )
        assert stat[4] == "0", (
            f"web server pid {pid} kept a controlling TTY ({stat[4]}), so a "
            "terminal hangup would reach it"
        )
        assert "200" in body.split("\r\n", 1)[0], body[:300]
    finally:
        stopped = _run(installed_cli, "--web", "--stop", "--port", str(port))
        if Path(f"/proc/{pid}").exists():  # pragma: no cover - cleanup guard
            os.kill(pid, 15)
        assert stopped.returncode == 0, (stopped.stdout + stopped.stderr)[:600]
