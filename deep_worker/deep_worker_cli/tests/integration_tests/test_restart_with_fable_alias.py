"""End-to-end guard for the model-alias fix: --restart --model fable.

SCOPE (deliberately bounded, per the rule that integration tests must not
hang):

* Persisted ``launch_config.model == "fable"`` in the merged
  ``session_metadata.json`` — the primary symptom of the original bug.
* The daemon's own argv (``/proc/<daemon_pid>/cmdline``) carries
  ``--model fable`` as an exact ``(--model, fable)`` pair — proves the
  launcher forwarded the alias correctly to the Python invocation.
* The launcher's stderr contains no ``WARNING: model`` line for the
  valid alias.

NOT in scope (and no assertion pretends otherwise):

* Whether the eventual ``ClaudeSDKClient`` child (spawned lazily on the
  first real query, see ``client_manager.py:555-569``) actually receives
  ``--model fable`` in its argv. Verifying that would require driving a
  real query through Anthropic's API with unbounded network latency,
  which the user has explicitly ruled out for integration tests. The
  merge-and-launch path is instead covered end-to-end by:
    - the daemon argv assertion above,
    - unit test ``tests/test_config_create_model_claude_code.py``
      (``ClaudeCodeChatModel(model="fable")`` build path),
    - and the design's §5 unit matrix.

WAITS AND CLEANUP:

* File waits use Linux ``inotify`` with a bounded ``select()`` — the same
  primitive the production ``ProcessRegistry.wait_for_registration`` uses
  (registry.py:397-473). No poll loops.
* Process reaping uses ``os.pidfd_open`` + ``select`` for BOTH the
  launcher and the daemon. This blocks in the kernel — the pidfd becomes
  readable exactly when the process exits — and never busy-loops. We
  deliberately avoid ``subprocess.Popen.wait(timeout=…)``: CPython
  implements that branch as an explicit ``while … waitpid(WNOHANG);
  time.sleep(delay)`` polling loop (see ``inspect.getsource
  (subprocess.Popen._wait)`` — the source literally begins with
  ``# Enter a busy loop``), which the project's no-polling policy
  forbids. After the pidfd signals exit we then call the plain
  ``launcher.wait()`` (no timeout) so ``waitpid`` reaps the zombie
  immediately without blocking. On kernels older than 5.3 where
  ``pidfd_open`` is unavailable we fall back to SIGKILL-then-forget
  rather than reintroducing polling.

The test hard-fails when ``DEEP_WORKER_REQUIRE_INTEGRATION=1`` (CI mode)
and the required binaries are missing, so a green CI run cannot be a
skipped-green run.
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import errno
import json
import os
import select
import shutil
import signal
import site
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home


# Hard budgets. If the assertions can't be reached inside these windows,
# something is wrong — better to fail loud than let a CI job hang.
_OVERALL_BUDGET_S = 60.0
_LAUNCH_BUDGET_S = 20.0
_MERGE_BUDGET_S = 15.0
_STOP_BUDGET_S = 5.0


_REPO_CLI_ROOT = Path(__file__).resolve().parents[2]  # deep_worker_cli/
_DEEPAGENTS_ROOT = _REPO_CLI_ROOT.parent / "deepagents"


def _skip_or_fail(msg: str) -> None:
    """Skip the test unless DEEP_WORKER_REQUIRE_INTEGRATION=1 → hard fail."""
    if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


# --- inotify helpers (copy of the pattern used by registry.py) ------------

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_MODIFY = 0x00000002
_INOTIFY_MASK = _IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE | _IN_MODIFY


def _open_inotify_watch(directory: Path) -> tuple[int, int] | None:
    """Open an inotify fd watching ``directory``. Returns (fd, wd) or None."""
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            return None
        libc = ctypes.CDLL(libc_name, use_errno=True)
        O_NONBLOCK = 0o4000
        fd = libc.inotify_init1(O_NONBLOCK)
        if fd < 0:
            return None
        wd = libc.inotify_add_watch(fd, str(directory).encode(), _INOTIFY_MASK)
        if wd < 0:
            os.close(fd)
            return None
        return fd, wd
    except (OSError, AttributeError):
        return None


def _drain_inotify_events(fd: int, target_name: bytes) -> bool:
    """Read pending events; return True if any names ``target_name``."""
    try:
        raw = os.read(fd, 4096)
    except BlockingIOError:
        return False
    except OSError:
        return False
    offset = 0
    hit = False
    while offset + 16 <= len(raw):
        _wd, _mask, _cookie, name_len = struct.unpack(
            b"iIII", raw[offset: offset + 16]
        )
        name = raw[offset + 16: offset + 16 + name_len].rstrip(b"\x00")
        if name == target_name:
            hit = True
        offset += 16 + name_len
    return hit


def _wait_for_content(
    file_path: Path, predicate, *, timeout: float
) -> bool:
    """Block until ``predicate(file_path)`` is True or timeout expires.

    Event-driven via inotify on the parent directory: ``select`` blocks in
    the kernel until the directory sees a write/create/rename, at which
    point we re-evaluate the predicate. Falls back to a single final
    predicate check if inotify is unavailable (so the test still functions
    on containers without CAP_SYS_ADMIN, at the cost of no waiting).
    """
    # Fast path: already true.
    try:
        if predicate(file_path):
            return True
    except (OSError, json.JSONDecodeError):
        pass

    watch = _open_inotify_watch(file_path.parent)
    if watch is None:
        # No inotify → single deadline-terminated select on nothing (i.e.
        # sleep once, then re-check). We cannot poll under the no-polling
        # rule; without a real primitive available, block once for the
        # full budget and re-check exactly once. Better a failed test than
        # a hidden poll loop.
        rr, _, _ = select.select([], [], [], timeout)
        try:
            return predicate(file_path)
        except (OSError, json.JSONDecodeError):
            return False

    fd, wd = watch
    target_name = file_path.name.encode()
    try:
        import time as _t

        deadline = _t.monotonic() + timeout
        while True:
            remaining = deadline - _t.monotonic()
            if remaining <= 0:
                # Final evaluation on timeout, in case a write completed
                # between our last check and the deadline.
                try:
                    return predicate(file_path)
                except (OSError, json.JSONDecodeError):
                    return False
            # Kernel-blocking wait for a filesystem event OR timeout.
            rr, _, _ = select.select([fd], [], [], remaining)
            if not rr:
                # Timeout with no events. Final predicate check.
                try:
                    return predicate(file_path)
                except (OSError, json.JSONDecodeError):
                    return False
            hit = _drain_inotify_events(fd, target_name)
            if hit:
                try:
                    if predicate(file_path):
                        return True
                except (OSError, json.JSONDecodeError):
                    # Partial write in progress — loop back to wait for
                    # the next inotify event.
                    continue
            # Event fired for a different file; loop and keep waiting.
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


# --- process helpers -------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _proc_children(parent_pid: int) -> list[int]:
    """PIDs whose /proc/<pid>/stat lists ``parent_pid`` as ppid."""
    kids: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return kids
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
            rparen = data.rfind(b")")
            if rparen < 0:
                continue
            fields = data[rparen + 2:].split()
            if len(fields) < 2:
                continue
            ppid = int(fields[1])
        except (OSError, ValueError):
            continue
        if ppid == parent_pid:
            kids.append(int(entry))
    return kids


def _proc_descendants(root_pid: int, *, max_depth: int = 4) -> list[int]:
    seen: set[int] = set()
    frontier = [root_pid]
    for _ in range(max_depth):
        next_frontier: list[int] = []
        for pid in frontier:
            for kid in _proc_children(pid):
                if kid not in seen:
                    seen.add(kid)
                    next_frontier.append(kid)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(seen)


def _cmdline_of(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    if not raw:
        return None
    return [
        s.decode("utf-8", "replace") for s in raw.rstrip(b"\x00").split(b"\x00")
    ]


def _pidfd_wait(pid: int, timeout: float) -> bool:
    """Kernel-block until ``pid`` exits, or return False on timeout.

    Uses Linux 5.3+ ``pidfd_open`` + ``select``. Not a poll loop: ``select``
    parks in the kernel until the pidfd becomes readable (which happens
    exactly when the process exits) OR the timeout expires.
    """
    if not hasattr(os, "pidfd_open"):
        return False
    try:
        fd = os.pidfd_open(pid)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return True  # already dead
        return False
    try:
        rr, _, _ = select.select([fd], [], [], timeout)
        return bool(rr)
    finally:
        os.close(fd)


def _kill_bounded(pid: int, *, budget: float) -> None:
    """SIGTERM → kernel-block on pidfd for ``budget`` → SIGKILL.

    No poll loop. The pidfd_open+select primitive waits IN THE KERNEL for
    the process to exit; when it doesn't within budget we escalate to
    SIGKILL, then wait once more (kernel-blocking) so we don't return
    before the process is truly gone.
    """
    if pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)
    if _pidfd_wait(pid, budget):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)
    # Final short wait so caller can trust the process is reaped.
    _pidfd_wait(pid, 1.0)


# --- test fixtures / helpers ----------------------------------------------


def _seed_launch_config(
    agent_home: Path, agent_name: str, working_dir: Path
) -> None:
    """Write a minimal but real session_metadata.json so ``--restart`` has
    something to merge against. Direct seeding is acceptable for this
    restart-specific integration."""
    data_dir = deep_worker_home(agent_home) / agent_name
    data_dir.mkdir(parents=True, exist_ok=True)
    launch_config = {
        "provider": "claude-code",
        "model": "sonnet",
        "model_provider": "claude-code",
        "working_dir": str(working_dir),
        "skip_permissions": True,
        "task_manager": False,
        "history_memory": False,
        "preferred_provider": "claude-code",
        "preferred_model": "sonnet",
        "storage_version": 5,
    }
    meta = {
        "status": "idle",
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": "",
        "storage_version": 5,
        "launch_config": launch_config,
    }
    meta_path = data_dir / "session_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    os.chmod(meta_path, 0o600)


def _isolated_env(agent_home: Path) -> dict[str, str]:
    """Env that swaps HOME to a sandbox while keeping dependency discovery.

    ``Path.home()`` is unconditional across the CLI (verified by
    ``grep -rn DEEP_WORKER_HOME`` returning zero). Swapping HOME to a
    tmpdir invalidates
    ``site.getusersitepackages()``; we snapshot the REAL user site BEFORE
    the swap and prepend it to PYTHONPATH so ``mcp`` etc. remain findable.

    We deliberately do NOT copy the operator's ``~/.claude`` credentials.
    This test never drives a real query — it stops at the daemon argv
    assertion — so authentication is unnecessary, and dragging live
    credentials into a sandbox needlessly widens the sensitive-data
    surface.
    """
    real_user_site = site.getusersitepackages()
    real_user_base = site.getuserbase()
    env = os.environ.copy()
    env["HOME"] = str(agent_home)
    existing_pp = env.get("PYTHONPATH", "")
    parts = [str(_REPO_CLI_ROOT), str(_DEEPAGENTS_ROOT), real_user_site]
    if existing_pp:
        parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUSERBASE", real_user_base)
    return env


@pytest.fixture
def agent_home(tmp_path: Path) -> Path:
    home = tmp_path / "isolated_home"
    home.mkdir()
    return home


@pytest.mark.timeout(int(_OVERALL_BUDGET_S) + 15)
def test_restart_with_fable_alias(agent_home: Path, tmp_path: Path) -> None:
    """Full launcher chain: --restart --model fable is persisted as fable
    AND appears exactly as ``--model fable`` on the daemon's own argv,
    with no reset warning on stderr."""
    # Preconditions: this test is only meaningful when the CLI binaries
    # actually exist. It does NOT need Claude authentication — it never
    # drives a query — so we do not check ``~/.claude``.
    if shutil.which("deep-worker") is None:
        _skip_or_fail("deep-worker CLI not on PATH")
    dev_launcher = _REPO_CLI_ROOT / "deep-worker"
    if not dev_launcher.exists():
        _skip_or_fail(f"dev-tree launcher missing at {dev_launcher}")

    # Unique agent name per run so a lingering daemon from a previous
    # aborted run cannot confuse the PID lookup.
    agent_name = f"int_fable_{os.getpid()}"
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    stale_pid = Path(f"/tmp/deep_worker_{agent_name}.pid")
    with contextlib.suppress(OSError):
        stale_pid.unlink()

    _seed_launch_config(agent_home, agent_name, working_dir)
    env = _isolated_env(agent_home)

    daemon_pid: int | None = None
    launcher: subprocess.Popen | None = None
    stdout_bytes = b""
    stderr_bytes = b""

    try:
        # Capture launcher output via files, NOT PIPE. The daemonized
        # grandchild inherits our stdout/stderr; holding them as PIPE
        # ends keeps ``Popen.wait`` waiting on EOF that never comes.
        stdout_file = agent_home / "_launcher.out"
        stderr_file = agent_home / "_launcher.err"
        with open(stdout_file, "wb") as sf_out, open(stderr_file, "wb") as sf_err:
            launcher = subprocess.Popen(
                [str(dev_launcher), "--dev", "--restart", agent_name,
                 "--model", "fable"],
                env=env,
                stdout=sf_out,
                stderr=sf_err,
                stdin=subprocess.DEVNULL,
            )
            # ``Popen.wait(timeout=…)`` is a busy-loop on POSIX CPython
            # (``waitpid(WNOHANG)`` + ``time.sleep``), forbidden by the
            # no-polling rule. Use pidfd instead: ``select`` on the pidfd
            # blocks in the kernel until the process exits or the timeout
            # expires. After the pidfd signals we call plain ``.wait()``
            # (no timeout) which returns immediately after reaping.
            if not _pidfd_wait(launcher.pid, _LAUNCH_BUDGET_S):
                launcher.kill()
                # Post-SIGKILL: pidfd_wait again (kernel-blocking) so we
                # do not return before the process is truly reaped.
                _pidfd_wait(launcher.pid, 2.0)
                launcher.wait()  # no timeout — process is dead, reap the zombie
                pytest.fail(
                    f"deep-worker --restart --model fable did not return "
                    f"within {_LAUNCH_BUDGET_S}s"
                )
            launcher.wait()  # no timeout — pidfd already fired, reap the zombie
        stdout_bytes = stdout_file.read_bytes() if stdout_file.exists() else b""
        stderr_bytes = stderr_file.read_bytes() if stderr_file.exists() else b""
        assert launcher.returncode == 0, (
            f"deep-worker exited {launcher.returncode}\n"
            f"stdout: {stdout_bytes.decode('utf-8', 'replace')}\n"
            f"stderr: {stderr_bytes.decode('utf-8', 'replace')}"
        )

        # === Assertion 1: persisted launch_config.model == 'fable' ===
        # The launcher returns as soon as daemonize() double-forks, i.e.
        # BEFORE the daemonized grandchild rewrites session_metadata.
        # inotify on the containing directory blocks in the kernel until
        # a write/create/rename happens, then we re-check the file's
        # content (json.load — cheap, only invoked on real fs events).
        meta_path = (
            deep_worker_home(agent_home) / agent_name / "session_metadata.json"
        )

        def _content_shows_fable(p: Path) -> bool:
            try:
                with open(p) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                return False
            return d.get("launch_config", {}).get("model") == "fable"

        merged = _wait_for_content(
            meta_path, _content_shows_fable, timeout=_MERGE_BUDGET_S
        )
        if not merged:
            log_dir = deep_worker_home(agent_home) / "logs"
            out_log = log_dir / f"{agent_name}.out.log"
            err_log = log_dir / f"{agent_name}.err.log"
            out_tail = out_log.read_text()[-2000:] if out_log.exists() else "(no out log)"
            err_tail = err_log.read_text()[-2000:] if err_log.exists() else "(no err log)"
            pytest.fail(
                f"daemon never wrote launch_config.model=fable within "
                f"{_MERGE_BUDGET_S}s.\n"
                f"launcher stdout: {stdout_bytes.decode('utf-8','replace')[:800]}\n"
                f"launcher stderr: {stderr_bytes.decode('utf-8','replace')[:800]}\n"
                f"daemon out.log tail: {out_tail}\n"
                f"daemon err.log tail: {err_tail}"
            )

        # === Assertion 2: no WARNING: model on stderr for a valid alias ===
        stderr_text = stderr_bytes.decode("utf-8", "replace")
        assert "WARNING: model" not in stderr_text, (
            f"honouring a valid alias must not emit a reset warning; "
            f"got stderr: {stderr_text!r}"
        )

        # === Assertion 3: daemon argv carries the exact --model/fable pair ===
        # The daemon is the ``deep_worker_cli.cli.agent_command`` python
        # process spawned by the launcher. Locate its PID and inspect
        # /proc/<pid>/cmdline. This proves the launcher AND all
        # merge/build_launch_config logic forwarded fable as the value of
        # ``--model``, not merely embedded fable as some unrelated token.
        daemon_pid = _daemon_pid_for(agent_home, agent_name)
        if daemon_pid is None:
            pytest.fail(
                f"could not locate the daemon PID for {agent_name}; "
                f"launcher stdout was "
                f"{stdout_bytes.decode('utf-8', 'replace')!r}"
            )
        assert _pid_alive(daemon_pid), (
            f"daemon PID {daemon_pid} died immediately after restart — "
            f"the model change may have crashed startup"
        )
        daemon_argv = _cmdline_of(daemon_pid) or []
        assert "--model" in daemon_argv, (
            f"daemon argv missing --model flag entirely; got: {daemon_argv!r}"
        )
        model_idx = daemon_argv.index("--model")
        assert model_idx + 1 < len(daemon_argv), (
            f"daemon argv has --model with no following value: "
            f"{daemon_argv!r}"
        )
        actual_value = daemon_argv[model_idx + 1]
        assert actual_value == "fable", (
            f"daemon argv --model value is {actual_value!r}, expected "
            f"'fable'. Full argv: {daemon_argv!r}"
        )

    finally:
        # Kill anything we started, hard-bounded so cleanup itself cannot
        # hang. Kernel-blocking waits only (Popen.wait/pidfd_open+select).
        if daemon_pid is not None and _pid_alive(daemon_pid):
            # Reap descendants first so the daemon does not respawn them.
            for kid in _proc_descendants(daemon_pid):
                _kill_bounded(kid, budget=1.0)
            _kill_bounded(daemon_pid, budget=_STOP_BUDGET_S)
        if launcher is not None and launcher.poll() is None:
            launcher.kill()
            # Kernel-block on the pidfd until the SIGKILL takes effect,
            # then reap without a timeout. Same reason as the main body:
            # ``Popen.wait(timeout=…)`` would busy-loop.
            _pidfd_wait(launcher.pid, 2.0)
            with contextlib.suppress(OSError):
                launcher.wait()
        # Best-effort tmp pid file cleanup so a subsequent run in the
        # same interpreter (unlikely — agent name embeds our PID — but
        # cheap) sees a clean slate.
        with contextlib.suppress(OSError):
            stale_pid.unlink()


def _daemon_pid_for(agent_home: Path, agent_name: str) -> int | None:
    """Read the daemon PID the agent registered under HOME=<agent_home>.

    ``daemonize()`` writes the pid to ``/tmp/deep_worker_<name>.pid`` (not
    inside HOME, confirmed by the launcher stdout line ``PID will be
    written to /tmp/deep_worker_<name>.pid``). The persisted
    ``session_metadata.json`` also records ``pid`` after the daemon
    settles, used as a secondary source.
    """
    tmp_pid = Path(f"/tmp/deep_worker_{agent_name}.pid")
    if tmp_pid.exists():
        try:
            return int(tmp_pid.read_text().strip())
        except (ValueError, OSError):
            pass
    meta_path = deep_worker_home(agent_home) / agent_name / "session_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            pid = meta.get("launch_config", {}).get("pid")
            if isinstance(pid, int):
                return pid
        except (json.JSONDecodeError, OSError):
            pass
    return None
