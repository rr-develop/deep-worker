"""Regression tests: repository-provided commands must not leak ``.venv``.

Background
----------
``.githooks/pre-commit`` hard-blocks any ``.venv`` in the working tree. The
rationale is an earlier incident: a stale ``rpyc`` installed inside a
``.venv`` shadowed a sibling async library project for every running agent,
leaking 21 GB RSS and taking a day to diagnose.

The hook's own type-check script cleans up after itself (``trap cleanup_venvs
EXIT INT TERM``). Nothing else did. Both package ``Makefile``s ran bare
``uv run`` in ``test``/``type``/``lint``/... targets, and the hook's own failure
message told developers to run a bare ``uv run`` by hand. Each of those creates
a ``.venv`` and leaves it, which then blocks the *next* commit on the venv check
rather than on the actual problem.

The fix routes every such command through ``deep_worker/scripts/uv-run.sh``,
which owns the cleanup. These tests pin that behaviour and — more importantly —
pin that no *new* bare invocation gets added anywhere (``test_no_bare_uv_run_*``).

Design: deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md

Stand-alone by design: stdlib + git only, no fixtures from the packages, so it
runs anywhere pytest does. ``uv`` itself is stubbed via a ``PATH`` seam wherever
a deterministic pass/fail/interrupt is needed, so the tests are fast and do not
depend on a network or a real dependency resolve.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import select
import signal
import stat
import subprocess
import textwrap
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Bounded budget for any single wrapper invocation in these tests. Nothing here
#: does real work, so anything slower than this means the wrapper is wedged --
#: which is a failure, not a reason to wait longer.
TIMEOUT = 60.0


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return Path(_git("rev-parse", "--show-toplevel", cwd=here))


ROOT = _repo_root()
SCRIPTS = ROOT / "deep_worker" / "scripts"
UV_RUN = SCRIPTS / "uv-run.sh"
TYPE_CHECK = SCRIPTS / "type-check.sh"
HOOK = SCRIPTS / "pre-commit-type-check.sh"
CLI_DIR = ROOT / "deep_worker" / "deep_worker_cli"
AGENTS_DIR = ROOT / "deep_worker" / "deepagents"


def _make_project(tmp_path: Path, name: str = "proj") -> Path:
    """A minimal directory that looks like a uv project root."""
    proj = tmp_path / name
    proj.mkdir(parents=True)
    (proj / "pyproject.toml").write_text(
        '[project]\nname = "stub"\nversion = "0.0.0"\n'
    )
    return proj


def _install_stub_uv(tmp_path: Path, body: str, name: str = "bin") -> Path:
    """Create a directory holding a stub ``uv`` executable, for the PATH seam."""
    bindir = tmp_path / name
    bindir.mkdir(parents=True, exist_ok=True)
    stub = bindir / "uv"
    stub.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    stub.chmod(0o755)
    return bindir


def _env(
    tmp_path: Path,
    bindir: Path | None = None,
    xdg: Path | None = None,
) -> dict[str, str]:
    """Environment with an isolated lock area and (optionally) a stub ``uv``."""
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}" if bindir else env["PATH"]
    # Keep every test's locks out of the real runtime dir so tests can never
    # serialize against a developer's own wrapper run.
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmpdir)
    if xdg is None:
        # MUST be cleared, not merely left alone: the real XDG_RUNTIME_DIR
        # (/run/user/<uid>, mode 0700 and owned by us) takes precedence over
        # TMPDIR, so leaving it set would send every lock there and make the
        # TMPDIR-based fixtures below assert against a path the wrapper never
        # touches -- tests that pass without exercising anything.
        env.pop("XDG_RUNTIME_DIR", None)
    else:
        env["XDG_RUNTIME_DIR"] = str(xdg)
    return env


def _run_wrapper(
    cwd: Path,
    *args: str,
    env: dict[str, str],
    timeout: float = TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(UV_RUN), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _venv(proj: Path) -> Path:
    return proj / ".venv"


def _alive(pid: int) -> bool:
    """True if *pid* still exists (any state, including zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, budget: float = 10.0) -> bool:
    """Block until *pid* exits, event-driven via pidfd.

    Deliberately not a sleep/retry loop: polling is banned in this project, and
    a pidfd becomes readable exactly when the process dies, so ``select`` gives
    us the answer without a timer. Processes we do not own (or that are already
    gone) fall through to a single ``kill(pid, 0)`` probe -- one syscall, not a
    loop.
    """
    if not _alive(pid):
        return True
    try:
        pidfd = os.pidfd_open(pid)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        # No pidfd (old kernel) or the process died between the two calls.
        return not _alive(pid)
    try:
        readable, _, _ = select.select([pidfd], [], [], budget)
        if not readable:
            return not _alive(pid)
        return True
    finally:
        os.close(pidfd)


requires_wrapper = pytest.mark.skipif(
    not UV_RUN.exists(),
    reason="uv-run.sh does not exist yet (RED phase of TDD)",
)


# ---------------------------------------------------------------------------
# 1. The scripts exist
# ---------------------------------------------------------------------------


def test_uv_run_wrapper_exists_and_is_executable() -> None:
    """The sanctioned seam every other command routes through."""
    assert UV_RUN.is_file(), (
        f"Missing {UV_RUN}. This wrapper is what makes `uv run` compliant with "
        f"the repo-wide .venv ban: it owns the cleanup. See "
        f"{SCRIPTS / 'DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md'}."
    )
    assert os.access(UV_RUN, os.X_OK), f"{UV_RUN} is not executable."


def test_type_check_wrapper_exists_and_is_executable() -> None:
    """The human-facing entry point suggested by the hook's failure output."""
    assert TYPE_CHECK.is_file(), (
        f"Missing {TYPE_CHECK}. The pre-commit hook points developers at this "
        f"script instead of a bare `uv run`, which is what used to leak .venv."
    )
    assert os.access(TYPE_CHECK, os.X_OK), f"{TYPE_CHECK} is not executable."


# ---------------------------------------------------------------------------
# 3-6. Wrapper cleanup behaviour
# ---------------------------------------------------------------------------


@requires_wrapper
def test_no_venv_left_when_command_succeeds(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(
        tmp_path,
        """
        mkdir -p .venv && echo stub > .venv/marker
        exit 0
        """,
    )
    result = _run_wrapper(proj, "true", env=_env(tmp_path, bindir))

    assert result.returncode == 0, result.stderr
    assert not _venv(proj).exists(), (
        f".venv survived a SUCCESSFUL run: {_venv(proj)}. The wrapper's whole "
        f"purpose is that it cannot."
    )


@requires_wrapper
def test_no_venv_left_when_command_fails(tmp_path: Path) -> None:
    """Failure is the interesting path: it is when developers rerun by hand."""
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(
        tmp_path,
        """
        mkdir -p .venv && echo stub > .venv/marker
        exit 7
        """,
    )
    result = _run_wrapper(proj, "false", env=_env(tmp_path, bindir))

    assert result.returncode == 7, (
        f"Expected the wrapped command's exit code (7), got "
        f"{result.returncode}. The cleanup must not swallow or rewrite it."
    )
    assert not _venv(proj).exists(), (
        f".venv survived a FAILED run: {_venv(proj)}. Cleanup must run on the "
        f"failure path, not only the happy path."
    )


@requires_wrapper
def test_exit_code_is_forwarded_verbatim(tmp_path: Path) -> None:
    """Exit code is the payload's, not the cleanup's (R3)."""
    proj = _make_project(tmp_path)
    for code in (0, 1, 3, 42):
        bindir = _install_stub_uv(tmp_path, f"exit {code}\n", name=f"bin{code}")
        result = _run_wrapper(proj, "x", env=_env(tmp_path, bindir))
        assert result.returncode == code, (
            f"Wrapper returned {result.returncode} for a command that exited "
            f"{code}."
        )


# ---------------------------------------------------------------------------
# 5, 5a. Signal handling and descendant containment (R8)
# ---------------------------------------------------------------------------


@requires_wrapper
def test_interrupt_leaves_no_venv_and_kills_child(tmp_path: Path) -> None:
    """SIGTERM mid-run: no .venv, recorded child dead, status 128+signo.

    The stub creates ``.venv`` *late*, so an implementation that cleans up
    before its child is dead loses the race deterministically.
    """
    proj = _make_project(tmp_path)
    pidfile = tmp_path / "child.pid"
    bindir = _install_stub_uv(
        tmp_path,
        f"""
        echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > {pidfile}
        sleep 3
        mkdir -p .venv && echo late > .venv/marker
        sleep 5
        """,
    )
    proc = subprocess.Popen(
        [str(UV_RUN), "slow"],
        cwd=str(proj),
        env=_env(tmp_path, bindir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    _await_file(pidfile)
    child_pid = int(pidfile.read_text().split()[0])

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.communicate(timeout=TIMEOUT)

    assert _wait_gone(child_pid), (
        f"Wrapped command (pid {child_pid}) survived the wrapper's exit."
    )
    assert not _venv(proj).exists(), (
        f".venv survived an INTERRUPTED run: {_venv(proj)}."
    )
    # Python reports death-by-signal as a negative returncode; a shell would
    # render the same thing as 128+signo. Either is the conventional status --
    # what must NOT happen is a synthetic code that hides how we died.
    assert proc.returncode in (-signal.SIGTERM, 128 + signal.SIGTERM), (
        f"Expected the wrapper to die of SIGTERM (reported as "
        f"{-signal.SIGTERM} or {128 + signal.SIGTERM}) after re-raising it, "
        f"got {proc.returncode}."
    )


@requires_wrapper
def test_interrupt_kills_grandchild_that_ignores_term(tmp_path: Path) -> None:
    """The case that makes the *unconditional* group KILL load-bearing.

    The stub (group leader) exits IMMEDIATELY, which instantly satisfies any
    "wait for the direct child" step, while a grandchild that ignores SIGTERM
    keeps running and creates ``.venv`` late.

    An implementation that treats leader exit as proof the group is empty, or
    that only escalates to KILL when a grace timer expires, passes the previous
    test and fails this one.
    """
    proj = _make_project(tmp_path)
    pidfile = tmp_path / "grandchild.pid"
    bindir = _install_stub_uv(
        tmp_path,
        f"""
        (
            trap '' TERM INT
            # $BASHPID, NOT $$: inside a subshell bash keeps $$ equal to the
            # PARENT's pid, so recording $$ here would make the assertion below
            # observe the already-exited leader and pass vacuously.
            echo "$BASHPID" > {pidfile}
            sleep 3
            mkdir -p .venv && echo grandchild > .venv/marker
            sleep 10
        ) &
        # Leader exits at once, while the grandchild lives on.
        exit 0
        """,
    )
    proc = subprocess.Popen(
        [str(UV_RUN), "spawner"],
        cwd=str(proj),
        env=_env(tmp_path, bindir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    _await_file(pidfile)
    grandchild_pid = int(pidfile.read_text().strip())

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.communicate(timeout=TIMEOUT)

    assert _wait_gone(grandchild_pid), (
        f"Grandchild (pid {grandchild_pid}) survived. Signalling only the "
        f"direct child cannot contain it -- the wrapper must KILL the whole "
        f"child process group unconditionally before cleaning up."
    )
    assert not _venv(proj).exists(), (
        f".venv survived, recreated by a grandchild after cleanup: "
        f"{_venv(proj)}."
    )


class _InotifyWatch:
    """Minimal event-driven watch on a directory, stdlib only.

    Used instead of a sleep/retry loop: ``wait()`` blocks in ``select`` until
    the kernel reports activity, so the test never re-reads on a timer.
    """

    _IN_MODIFY = 0x00000002
    _IN_CREATE = 0x00000100
    _IN_MOVED_TO = 0x00000080
    _IN_CLOSE_WRITE = 0x00000008

    def __init__(self, directory: Path) -> None:
        self._fd = -1
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            fd = libc.inotify_init1(os.O_CLOEXEC)
            if fd < 0:
                return
            mask = (
                self._IN_MODIFY
                | self._IN_CREATE
                | self._IN_MOVED_TO
                | self._IN_CLOSE_WRITE
            )
            if libc.inotify_add_watch(fd, str(directory).encode(), mask) < 0:
                os.close(fd)
                return
            self._fd = fd
        except (OSError, AttributeError):
            self._fd = -1

    @property
    def supported(self) -> bool:
        return self._fd >= 0

    def wait(self, timeout: float) -> bool:
        readable, _, _ = select.select([self._fd], [], [], timeout)
        if not readable:
            return False
        try:
            os.read(self._fd, 64 * 1024)  # drain; the content is irrelevant
        except OSError:
            return False
        return True

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def _await_file(path: Path, budget: float = 20.0) -> None:
    """Block until a stub announces itself by writing *path*.

    Event-driven via inotify rather than a polling loop. The watch is armed
    BEFORE the first read, so a file written in between is not missed.
    """
    watch = _InotifyWatch(path.parent)
    if not watch.supported:  # pragma: no cover - depends on the kernel
        pytest.skip("inotify unavailable; refusing to fall back to polling")
    try:
        deadline = time.monotonic() + budget
        while True:
            if path.exists() and path.read_text().strip():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"Stub never wrote {path} within {budget}s")
            watch.wait(remaining)
    finally:
        watch.close()


# ---------------------------------------------------------------------------
# 7. Pre-existing .venv must be preserved, never deleted (R6)
# ---------------------------------------------------------------------------


@requires_wrapper
def test_preexisting_venv_is_preserved_not_deleted(tmp_path: Path) -> None:
    """A ``.venv`` the wrapper did not create is a hard stop, not a cleanup job.

    Deleting it would (a) destroy state the wrapper does not own and (b) hide
    exactly the situation the repo-wide ban exists to surface.
    """
    proj = _make_project(tmp_path)
    venv = _venv(proj)
    venv.mkdir()
    sentinel = venv / "sentinel.txt"
    payload = b"do not touch me\n" * 16
    sentinel.write_bytes(payload)

    bindir = _install_stub_uv(tmp_path, "echo 'uv should not have run' >&2\nexit 0\n")
    result = _run_wrapper(proj, "true", env=_env(tmp_path, bindir))

    assert result.returncode != 0, (
        "Wrapper must FAIL loudly when a .venv already exists; it cannot claim "
        "ownership of an environment it did not create."
    )
    assert sentinel.exists(), f"Pre-existing {sentinel} was deleted."
    assert sentinel.read_bytes() == payload, (
        f"Pre-existing {sentinel} was modified. It must be byte-identical."
    )


# ---------------------------------------------------------------------------
# 7a, 7c. Serialization and lock identity (R7)
# ---------------------------------------------------------------------------


@requires_wrapper
def test_concurrent_wrappers_same_root_do_not_overlap(tmp_path: Path) -> None:
    """Two invocations against one root must serialize.

    Without a lock both observe "no .venv", both run, and the first one's
    cleanup deletes the environment out from under the second -- a correctness
    bug for the second command, not merely a cleanup bug.
    """
    proj = _make_project(tmp_path)
    log = tmp_path / "intervals.log"
    bindir = _install_stub_uv(
        tmp_path,
        f"""
        echo "enter $$ $(date +%s%N)" >> {log}
        mkdir -p .venv && echo stub > .venv/marker
        sleep 1
        echo "exit $$ $(date +%s%N)" >> {log}
        exit 0
        """,
    )
    env = _env(tmp_path, bindir)
    procs = [
        subprocess.Popen(
            [str(UV_RUN), "concurrent"],
            cwd=str(proj),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for p in procs:
        p.communicate(timeout=TIMEOUT)
        assert p.returncode == 0, "Both serialized runs should still succeed."

    intervals = _parse_intervals(log)
    assert len(intervals) == 2, f"Expected 2 complete runs, got {intervals}"
    (a_start, a_end), (b_start, b_end) = sorted(intervals)
    assert a_end <= b_start, (
        f"The two runs OVERLAPPED ({a_start}-{a_end} vs {b_start}-{b_end}). "
        f"The second entered `uv` before the first had cleaned up and released "
        f"the lock, so neither can truthfully own the .venv it sees."
    )
    assert not _venv(proj).exists()


@requires_wrapper
def test_concurrent_wrappers_distinct_roots_do_overlap(tmp_path: Path) -> None:
    """Distinct roots must NOT serialize against each other.

    Guards against a truncated or weak digest collapsing unrelated projects
    onto a single lock file, which would silently serialize the whole repo.
    """
    proj_a = _make_project(tmp_path, "alpha")
    proj_b = _make_project(tmp_path, "beta")
    log = tmp_path / "intervals.log"
    bindir = _install_stub_uv(
        tmp_path,
        f"""
        echo "enter $$ $(date +%s%N)" >> {log}
        sleep 1
        echo "exit $$ $(date +%s%N)" >> {log}
        exit 0
        """,
    )
    env = _env(tmp_path, bindir)
    procs = [
        subprocess.Popen(
            [str(UV_RUN), "concurrent"],
            cwd=str(p),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for p in (proj_a, proj_b)
    ]
    for p in procs:
        p.communicate(timeout=TIMEOUT)
        assert p.returncode == 0

    intervals = _parse_intervals(log)
    assert len(intervals) == 2
    (a_start, a_end), (b_start, b_end) = sorted(intervals)
    assert b_start < a_end, (
        f"Runs against DIFFERENT project roots serialized ({a_start}-{a_end} "
        f"then {b_start}-{b_end}). Their lock files must differ -- most likely "
        f"the lock name is a truncated digest and the roots collided."
    )


def _parse_intervals(log: Path) -> list[tuple[int, int]]:
    """Pair up ``enter``/``exit`` markers per pid into (start, end) tuples."""
    starts: dict[str, int] = {}
    out: list[tuple[int, int]] = []
    for line in log.read_text().splitlines():
        kind, pid, stamp = line.split()
        if kind == "enter":
            starts[pid] = int(stamp)
        else:
            out.append((starts[pid], int(stamp)))
    return out


def _lock_path_for(proj: Path, env: dict[str, str]) -> Path:
    digest = hashlib.sha256(str(proj.resolve()).encode()).hexdigest()
    return _uid_dir(env) / "deep-worker-uv-locks" / digest


#: A stub that reports, from inside the payload, whether the lock is genuinely
#: held -- by trying to take it through an INDEPENDENT open. flock is per open
#: file description, so this is the only honest probe.
_CONTENTION_PROBE = """
python3 - <<'PY' > "$LOCK_REPORT" 2>&1
import fcntl, os
path = os.environ["LOCK_REPORT_TARGET"]
try:
    fd = os.open(path, os.O_RDWR)
except OSError as exc:
    print(f"NO_LOCK_FILE {exc}")
else:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("LOCKED")
    else:
        print("NOT_LOCKED")
PY
exit 0
"""


@requires_wrapper
def test_forged_env_marker_cannot_skip_the_lock(tmp_path: Path) -> None:
    """The lock must not be skippable via a caller-controlled env variable.

    ``uv-run.sh`` resumes after the locking helper re-execs it. If that resume
    decision came from an environment marker, anyone could set the marker and
    run the payload unlocked -- and worse, the payload inherits our environment,
    so a NESTED wrapper would silently believe a lock was already held.
    Whether we hold the lock must be established from the kernel, not the env.
    """
    proj = _make_project(tmp_path)
    report = tmp_path / "report.txt"
    bindir = _install_stub_uv(tmp_path, _CONTENTION_PROBE)
    env = _env(tmp_path, bindir)
    env["LOCK_REPORT"] = str(report)
    env["LOCK_REPORT_TARGET"] = str(_lock_path_for(proj, env))
    # The forgery: claim the lock is already held.
    env["_UV_RUN_LOCK_FD"] = "9"

    result = _run_wrapper(proj, "probe", env=env)

    assert result.returncode == 0, result.stderr
    assert report.exists(), (
        "The payload never ran, so this test proved nothing. Check the stub."
    )
    verdict = report.read_text().strip()
    assert verdict == "LOCKED", (
        f"With a forged environment marker the payload observed {verdict!r}. "
        f"The wrapper trusted the marker and skipped locking, so two runs "
        f"could both claim ownership of the same .venv."
    )


@requires_wrapper
def test_nested_wrapper_for_another_root_takes_its_own_lock(
    tmp_path: Path,
) -> None:
    """An inherited fd 9 for project A must not count as a lock for project B.

    The payload inherits our fds, so a wrapper invoked from inside another
    wrapper sees a real, genuinely-flocked fd 9 -- just for the WRONG project.
    Identity has to be checked against this project's lock inode.
    """
    outer = _make_project(tmp_path, "outer")
    inner = _make_project(tmp_path, "inner")
    report = tmp_path / "report.txt"

    env = _env(tmp_path, None)
    inner_lock = _lock_path_for(inner, env)

    # Outer payload = a nested wrapper run against the INNER project.
    inner_stub = _install_stub_uv(tmp_path, _CONTENTION_PROBE, name="bin_inner")
    outer_stub_body = f"""
    cd {inner}
    PATH={inner_stub}:$PATH LOCK_REPORT="$LOCK_REPORT" \\
        LOCK_REPORT_TARGET="$LOCK_REPORT_TARGET" \\
        {UV_RUN} nested
    exit $?
    """
    outer_bindir = _install_stub_uv(tmp_path, outer_stub_body, name="bin_outer")

    env["PATH"] = f"{outer_bindir}:{os.environ['PATH']}"
    env["LOCK_REPORT"] = str(report)
    env["LOCK_REPORT_TARGET"] = str(inner_lock)

    result = _run_wrapper(outer, "outer", env=env)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert report.exists(), "The nested payload never ran; the test proved nothing."
    verdict = report.read_text().strip()
    assert verdict == "LOCKED", (
        f"The nested wrapper for {inner} observed {verdict!r} on its OWN lock. "
        f"It reused the outer project's inherited lock instead of taking its "
        f"own, so two runs against {inner} could proceed concurrently."
    )


@requires_wrapper
def test_fd9_holding_another_projects_lock_is_not_accepted(
    tmp_path: Path,
) -> None:
    """A genuinely-locked fd 9 for project A must not authorise project B.

    ``flock`` on one's own open file description always succeeds, so "locking
    fd 9 works" alone would report "held" for whatever project we happen to be
    asked about. Identity has to be pinned to THIS project's lock inode -- which
    is what makes a nested wrapper take its own lock instead of coasting on an
    inherited one.
    """
    other = _make_project(tmp_path, "other")
    target = _make_project(tmp_path, "target")
    report = tmp_path / "report.txt"
    bindir = _install_stub_uv(tmp_path, _CONTENTION_PROBE)
    env = _env(tmp_path, bindir)

    other_lock = _lock_path_for(other, env)
    target_lock = _lock_path_for(target, env)
    other_lock.parent.mkdir(parents=True, exist_ok=True)
    _uid_dir(env).chmod(0o700)
    other_lock.parent.chmod(0o700)
    for path in (other_lock, target_lock):
        path.touch()
        path.chmod(0o600)

    env["LOCK_REPORT"] = str(report)
    env["LOCK_REPORT_TARGET"] = str(target_lock)

    # fd 9 genuinely holds the OTHER project's lock.
    inherited = os.open(other_lock, os.O_RDWR)
    fcntl.flock(inherited, fcntl.LOCK_EX)
    # Meanwhile a third party owns the lock we actually need.
    contender = os.open(target_lock, os.O_RDWR)
    fcntl.flock(contender, fcntl.LOCK_EX)
    try:

        def _preexec() -> None:  # pragma: no cover - runs in the child
            os.dup2(inherited, 9)

        proc = subprocess.Popen(
            [str(UV_RUN), "probe"],
            cwd=str(target),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_preexec,
            close_fds=False,
        )
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # Correct: it did not accept the foreign lock and is blocking for
            # the right one.
            proc.kill()
            proc.communicate(timeout=TIMEOUT)
            assert not report.exists(), (
                "The payload ran while a third party held the target lock."
            )
            return
    finally:
        os.close(contender)
        os.close(inherited)

    assert not report.exists(), (
        "The wrapper accepted fd 9's lock on a DIFFERENT project as ownership "
        "of this project's lock, and ran the payload while that lock was held "
        "by someone else. Verification must compare the lock file's inode."
    )


@requires_wrapper
def test_open_but_unlocked_fd9_does_not_count_as_holding_the_lock(
    tmp_path: Path,
) -> None:
    """fd 9 pointing at the right file is NOT proof that fd 9 holds the lock.

    The forgery here is subtle and was a real defect: open the correct lock
    file on fd 9 *without* locking it, while another process genuinely holds
    the lock. Every identity check passes (same inode, our uid, mode 0600), and
    a probe through an independent ``open`` returns EAGAIN -- but that EAGAIN
    comes from the OTHER holder, not from fd 9. Concluding "held" there would
    run the payload concurrently with the real lock owner.

    Verification must therefore act on fd 9 itself.
    """
    proj = _make_project(tmp_path)
    report = tmp_path / "report.txt"
    bindir = _install_stub_uv(tmp_path, _CONTENTION_PROBE)
    env = _env(tmp_path, bindir)
    lock_path = _lock_path_for(proj, env)
    env["LOCK_REPORT"] = str(report)
    env["LOCK_REPORT_TARGET"] = str(lock_path)

    # Materialise the lock file exactly as the wrapper would.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _uid_dir(env).chmod(0o700)
    lock_path.parent.chmod(0o700)
    lock_path.touch()
    lock_path.chmod(0o600)

    holder = os.open(lock_path, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)  # a "real wrapper" owns the lock
    forged = os.open(lock_path, os.O_RDWR)  # correct file, NOT locked
    try:
        # Hand the forged, unlocked descriptor to the wrapper as fd 9.
        def _preexec() -> None:  # pragma: no cover - runs in the child
            os.dup2(forged, 9)

        proc = subprocess.Popen(
            [str(UV_RUN), "probe"],
            cwd=str(proj),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_preexec,
            close_fds=False,
        )
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # Correct behaviour: it could not verify ownership, so it went
            # through the helper and is now BLOCKING on the lock we hold.
            proc.kill()
            proc.communicate(timeout=TIMEOUT)
            assert not report.exists(), (
                "The payload ran while another process held the lock."
            )
            return
    finally:
        os.close(forged)
        os.close(holder)

    assert not report.exists(), (
        "The wrapper accepted an open-but-unlocked fd 9 as proof of ownership "
        "and ran the payload while another process held the lock. Verification "
        "must flock fd 9 itself, not probe through a separate open."
    )


@requires_wrapper
def test_lock_file_is_full_sha256_of_resolved_root(tmp_path: Path) -> None:
    """The lock name must be the FULL digest, not a prefix (collision safety)."""
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    env = _env(tmp_path, bindir)
    _run_wrapper(proj, "true", env=env)

    digest = hashlib.sha256(str(proj.resolve()).encode()).hexdigest()
    lock_dir = Path(env["TMPDIR"]) / f"deep-worker-{os.geteuid()}" / "deep-worker-uv-locks"
    assert lock_dir.is_dir(), (
        f"Expected a private lock directory at {lock_dir}; found none. The "
        f"lock must live outside the project tree so it is never scanned by "
        f"the .venv ban."
    )
    names = {p.name for p in lock_dir.iterdir()}
    assert digest in names, (
        f"Lock file for {proj} should be named with the full SHA-256 digest "
        f"{digest}; found {sorted(names)}."
    )


# ---------------------------------------------------------------------------
# 7b. Hostile lock paths must fail closed, at EVERY level
# ---------------------------------------------------------------------------


def _uid_dir(env: dict[str, str]) -> Path:
    return Path(env["TMPDIR"]) / f"deep-worker-{os.geteuid()}"


@requires_wrapper
@pytest.mark.parametrize("hostile", ["symlink", "world_writable", "not_a_dir"])
@pytest.mark.parametrize("level", ["uid_parent", "lock_subdir"])
def test_hostile_lock_path_fails_closed(
    tmp_path: Path, level: str, hostile: str
) -> None:
    """Each created path component is validated, not just the deepest one.

    ``TMPDIR`` is attacker-controllable and the per-UID component under it is
    predictable, so validating only ``deep-worker-uv-locks`` while trusting its
    parent leaves the identical hole one level up.
    """
    proj = _make_project(tmp_path)
    marker = tmp_path / "uv_was_invoked"
    bindir = _install_stub_uv(tmp_path, f"touch {marker}\nexit 0\n")
    env = _env(tmp_path, bindir)

    uid_dir = _uid_dir(env)
    target = uid_dir if level == "uid_parent" else uid_dir / "deep-worker-uv-locks"
    if level == "lock_subdir":
        uid_dir.mkdir(parents=True)
        uid_dir.chmod(0o700)

    elsewhere = tmp_path / "attacker"
    elsewhere.mkdir()
    if hostile == "symlink":
        target.symlink_to(elsewhere, target_is_directory=True)
    elif hostile == "world_writable":
        target.mkdir(parents=True)
        target.chmod(0o777)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not a directory\n")

    result = _run_wrapper(proj, "true", env=env)

    assert result.returncode != 0, (
        f"Wrapper proceeded with a hostile {hostile} at the {level} level "
        f"({target}). It must fail closed."
    )
    assert not marker.exists(), (
        "`uv` was invoked despite an unsafe lock path; validation must happen "
        "before any command runs."
    )
    combined = result.stdout + result.stderr
    assert str(target) in combined, (
        f"The error message must name the offending path {target} so a human "
        f"can act on it. Got:\n{combined}"
    )


@requires_wrapper
def test_hostile_lock_file_symlink_is_not_followed(tmp_path: Path) -> None:
    """A pre-created symlink at the lock path must not be written through."""
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    env = _env(tmp_path, bindir)

    lock_dir = _uid_dir(env) / "deep-worker-uv-locks"
    lock_dir.mkdir(parents=True)
    _uid_dir(env).chmod(0o700)
    lock_dir.chmod(0o700)

    victim = tmp_path / "victim.txt"
    payload = b"precious contents\n"
    victim.write_bytes(payload)
    # 0600 so the victim would satisfy the wrapper's fstat checks if the symlink
    # were followed; the refusal must come from O_NOFOLLOW, not from the mode.
    victim.chmod(0o600)
    digest = hashlib.sha256(str(proj.resolve()).encode()).hexdigest()
    (lock_dir / digest).symlink_to(victim)

    result = _run_wrapper(proj, "true", env=env)

    assert result.returncode != 0, (
        "Wrapper must refuse a lock path that is a symlink (O_NOFOLLOW)."
    )
    assert victim.read_bytes() == payload, (
        f"The symlink target {victim} was written through or truncated."
    )


@requires_wrapper
def test_permissive_existing_lock_file_is_rejected(tmp_path: Path) -> None:
    """An owner-owned but world-writable lock file must NOT be accepted.

    Ownership alone is not enough: a 0666 lock file lets any local user hold or
    corrupt the lock, which defeats the serialization the ``.venv`` ownership
    claim rests on.
    """
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    env = _env(tmp_path, bindir)

    lock_dir = _uid_dir(env) / "deep-worker-uv-locks"
    lock_dir.mkdir(parents=True)
    _uid_dir(env).chmod(0o700)
    lock_dir.chmod(0o700)
    digest = hashlib.sha256(str(proj.resolve()).encode()).hexdigest()
    permissive = lock_dir / digest
    permissive.touch()
    permissive.chmod(0o666)

    result = _run_wrapper(proj, "true", env=env)

    assert result.returncode != 0, (
        "A 0666 lock file was accepted. Mode must be validated from the opened "
        "file description, not just ownership."
    )
    combined = result.stdout + result.stderr
    assert "0600" in combined or "mode" in combined.lower(), (
        f"Error should explain the mode problem. Got:\n{combined}"
    )


@requires_wrapper
def test_lock_open_does_not_follow_symlink_under_churn(tmp_path: Path) -> None:
    """TOCTOU probe: swap the lock path for a symlink while runs are starting.

    A static pre-created symlink only proves the *check* works. This repeatedly
    races the swap against the open, so an implementation that checks the
    pathname and then opens it with plain shell redirection (which follows
    symlinks) eventually writes through to the victim.
    """
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    env = _env(tmp_path, bindir)

    lock_dir = _uid_dir(env) / "deep-worker-uv-locks"
    lock_dir.mkdir(parents=True)
    _uid_dir(env).chmod(0o700)
    lock_dir.chmod(0o700)
    digest = hashlib.sha256(str(proj.resolve()).encode()).hexdigest()
    lock_path = lock_dir / digest

    victim = tmp_path / "victim.txt"
    payload = b"precious contents\n"
    # Written ONCE, before the churn starts. The swapper must never rewrite it:
    # if it did, a read could catch the file mid-write and report a spurious
    # "corruption" that the wrapper had nothing to do with.
    victim.write_bytes(payload)
    # 0600 and owned by us, i.e. the victim passes every check the wrapper makes
    # on an *opened* lock file. Without this the fstat validation would reject
    # the followed symlink for the wrong reason (mode 0644), and the test would
    # still pass with O_NOFOLLOW removed -- proving nothing about following.
    victim.chmod(0o600)

    stop = False

    def swapper() -> None:
        while not stop:
            try:
                if lock_path.exists() or lock_path.is_symlink():
                    lock_path.unlink()
                lock_path.symlink_to(victim)
            except OSError:
                pass

    thread = threading.Thread(target=swapper, daemon=True)
    thread.start()
    # Observing "was the victim opened?" cannot be done by looking at contents
    # or inode: the wrapper opens O_RDWR without truncating, and flock changes
    # neither. So we detect it the only way that is actually observable -- by
    # holding a conflicting lock on the victim ourselves. If the wrapper
    # resolves the symlink, its blocking flock() contends with ours and the run
    # hangs; with O_NOFOLLOW it refuses the path outright and returns promptly.
    victim_fd = os.open(victim, os.O_RDWR)
    fcntl.flock(victim_fd, fcntl.LOCK_EX)
    try:
        for _ in range(25):
            try:
                result = _run_wrapper(proj, "true", env=env, timeout=15)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"The wrapper blocked trying to flock {victim}, so it "
                    f"followed the symlinked lock path. The open must be "
                    f"atomic with O_NOFOLLOW; checking the pathname first and "
                    f"then opening it by name is a TOCTOU."
                )
            # Either it refused the symlink (non-zero) or the swapper happened
            # to leave a real file in place (zero). Both are fine; what must
            # never happen is contending on the victim.
            assert result.returncode in (0, 1), (
                f"Unexpected exit {result.returncode}: {result.stderr}"
            )
            assert victim.read_bytes() == payload, (
                f"The victim file {victim} was modified through the symlinked "
                f"lock path."
            )
    finally:
        stop = True
        thread.join(timeout=5)
        os.close(victim_fd)

    assert victim.read_bytes() == payload


@requires_wrapper
@pytest.mark.parametrize("kind", ["relative", "missing"])
def test_unusable_xdg_runtime_dir_falls_back_rather_than_aborting(
    tmp_path: Path, kind: str
) -> None:
    """An unusable ``XDG_RUNTIME_DIR`` is ordinary, not suspicious.

    Unlike a hostile lock path (fail closed), this must fall back to the
    per-UID directory and carry on.
    """
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    xdg = Path("relative/path") if kind == "relative" else tmp_path / "gone"
    env = _env(tmp_path, bindir, xdg=xdg)

    result = _run_wrapper(proj, "true", env=env)

    assert result.returncode == 0, (
        f"XDG_RUNTIME_DIR={xdg} is merely unusable, not hostile; the wrapper "
        f"should fall back to the per-UID directory. stderr:\n{result.stderr}"
    )
    assert _uid_dir(env).is_dir(), "Expected the per-UID fallback to be used."


@requires_wrapper
def test_lock_directories_are_private(tmp_path: Path) -> None:
    """No group/world bits on anything the wrapper creates."""
    proj = _make_project(tmp_path)
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    env = _env(tmp_path, bindir)
    _run_wrapper(proj, "true", env=env)

    for path in (_uid_dir(env), _uid_dir(env) / "deep-worker-uv-locks"):
        assert path.is_dir(), f"{path} was not created"
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o700, f"{path} has mode {mode:04o}, expected 0700"


# ---------------------------------------------------------------------------
# 8, 9. type-check.sh semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not TYPE_CHECK.exists(), reason="type-check.sh does not exist yet (RED phase)"
)
def test_type_check_runs_both_packages_and_aggregates_failure(
    tmp_path: Path,
) -> None:
    """A failure in the first package must not skip the second (`set -e` trap).

    Otherwise a developer fixes one error, reruns, and discovers a second one
    that was there all along -- one round trip per error.
    """
    seen = tmp_path / "seen.log"
    bindir = _install_stub_uv(
        tmp_path,
        f"""
        echo "$PWD" >> {seen}
        case "$PWD" in
            *deep_worker_cli) exit 1 ;;
            *) exit 0 ;;
        esac
        """,
    )
    result = subprocess.run(
        [str(TYPE_CHECK)],
        cwd=str(ROOT),
        env=_env(tmp_path, bindir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    checked = seen.read_text() if seen.exists() else ""
    assert "deep_worker_cli" in checked, "deep_worker_cli was never checked"
    assert "deepagents" in checked, (
        f"deepagents was NOT checked after deep_worker_cli failed. The script "
        f"must aggregate, not short-circuit. Saw:\n{checked}"
    )
    assert result.returncode != 0, "Aggregate result must be non-zero on failure."


@pytest.mark.skipif(
    not TYPE_CHECK.exists(), reason="type-check.sh does not exist yet (RED phase)"
)
def test_type_check_single_package_argument(tmp_path: Path) -> None:
    seen = tmp_path / "seen.log"
    bindir = _install_stub_uv(tmp_path, f'echo "$PWD" >> {seen}\nexit 0\n')
    result = subprocess.run(
        [str(TYPE_CHECK), "deepagents"],
        cwd=str(ROOT),
        env=_env(tmp_path, bindir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    checked = seen.read_text()
    assert "deepagents" in checked
    assert "deep_worker_cli" not in checked, (
        f"Asked for deepagents only, but deep_worker_cli was checked too:\n"
        f"{checked}"
    )


@pytest.mark.skipif(
    not TYPE_CHECK.exists(), reason="type-check.sh does not exist yet (RED phase)"
)
def test_type_check_rejects_unknown_package(tmp_path: Path) -> None:
    bindir = _install_stub_uv(tmp_path, "exit 0\n")
    result = subprocess.run(
        [str(TYPE_CHECK), "no_such_package"],
        cwd=str(ROOT),
        env=_env(tmp_path, bindir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    assert result.returncode != 0, "Unknown package must be an error, not a no-op."
    combined = (result.stdout + result.stderr).lower()
    assert "no_such_package" in combined, "Error must name the bad argument."
