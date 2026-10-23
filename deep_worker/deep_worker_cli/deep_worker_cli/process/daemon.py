from __future__ import annotations

import json
import os
import select
import socket
import sys
from pathlib import Path
from collections.abc import Callable

from deep_worker_cli.storage_paths import deep_worker_home

DAEMON_UMASK = 0o022

# How long the launching terminal waits for the daemonized grandchild to
# report "ready" (agent registered, running) or "error" (init failed, but
# the process is still alive in AgentStatus.ERROR — see
# docs/DESIGN_FAIL_FAST_AGENT_STARTUP.md) over the readiness socketpair,
# before giving up and reporting a timeout. Mirrors ``web_daemon._START_TIMEOUT``.
READY_TIMEOUT = 90.0


def _detach_from_session_scope(agent_name: str | None = None) -> None:
    """Move the calling process out of any ``session-*.scope`` /
    ``tmux-spawn-*.scope`` / ``app-*.scope`` it currently lives in,
    into a freshly-created leaf cgroup under ``user@UID.service``.

    Why this is needed in addition to ``setsid()`` + double-fork:
    the classic Unix daemonization recipe gives the daemon ``PPid=1``
    and a fresh session/process-group, but it does NOT change the
    process's cgroup membership. On systemd hosts the launching
    shell lives in a session-scope (or tmux-/app-scope), and that
    scope is what gets torn down on SSH logout / tmux pane close /
    terminal exit. When the kernel tears down the scope it kills
    every process inside it — including a fully-daemonised one,
    because cgroup membership was inherited across ``fork`` and
    ``setsid`` does not touch it.

    Production incident this guards against: 2026-05-08, an agent
    died at the moment systemd deactivated the SSH
    session-scope of the bash that originally launched it, even
    though ``loginctl enable-linger`` was set on the user. ``linger``
    keeps ``user@UID.service`` running after SSH logout, but it
    does not make session-scopes immortal — and the daemon was in
    a session-scope, not in a delegated user-manager subgroup. See
    a related internal incident analysis (not included here).

    Mechanism (cgroup v2, the unified hierarchy used by every
    systemd v240+ default):

    1. ``user@UID.service`` itself has ``cgroup.subtree_control``
       enabled (typically ``cpu memory pids``). cgroup v2's
       "no internal processes" rule then forbids writing PIDs
       directly into ``user@UID.service/cgroup.procs`` — that
       returns ``EBUSY``. So we MUST create a leaf cgroup first.
    2. ``mkdir`` a fresh ``user@UID.service/dwagent-<name>-<pid>``
       directory. The user owns ``user@UID.service`` (systemd
       grants ``Delegate=yes`` by default) so this needs no root.
    3. Write our PID into ``<that>/cgroup.procs`` to migrate.

    The leaf cgroup is intentionally per-process (PID in the name)
    so that two agents launched concurrently never collide on the
    cgroup directory. systemd's user-manager will reap the empty
    cgroup automatically when the daemon exits and the directory
    becomes empty — no explicit cleanup is needed in the happy
    path.

    Failure handling: if any step fails (no cgroup v2, no
    delegation, sysfs read-only, container without a user manager,
    EPERM on mkdir, …) the function returns silently. The daemon
    keeps running in its current cgroup; the parent-shell-death
    risk remains on those platforms but we don't have a better
    primitive without taking on a hard dbus / systemd-run runtime
    dependency.

    Idempotent: if the process is already outside the unsafe scope
    set, it is a no-op.

    Args:
        agent_name: optional name to embed in the new cgroup
            directory name, for ``systemd-cgls`` readability. The
            PID is always appended, so collisions are impossible.
    """
    try:
        with open("/proc/self/cgroup") as f:
            line = f.read().strip()
    except OSError:
        return  # /proc not available — nothing we can do
    # cgroup v2 single-line entry: ``0::/path``. cgroup v1 has
    # multiple lines with ``N:controller:/path``; we don't support
    # migration there.
    if not line.startswith("0::"):
        return
    cur = line.split("::", 1)[1]
    user_service = (
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service"
    )
    safe_name = "".join(
        c if (c.isalnum() or c in "-_") else "_"
        for c in (agent_name or "agent")
    )[:64]
    leaf_basename = f"dwagent-{safe_name}-{os.getpid()}"
    leaf_dir = f"{user_service}/{leaf_basename}"
    # Skip migration in two cases:
    # 1. We are already in our OWN leaf cgroup (re-entry, e.g. a
    #    daemonize() called twice somehow). Idempotency.
    # 2. We are in a cgroup that is genuinely safe AND owned by us
    #    (i.e. another daemon's leaf is NOT considered safe — if
    #    that daemon dies and its leaf is reaped, our process will
    #    be migrated up by the kernel, but more importantly the
    #    parent daemon's lifetime should not own ours).
    # The cgroups that DO get killed when the parent shell dies:
    #   /user.slice/user-UID.slice/session-N.scope    SSH/login
    #   /user.slice/user-UID.slice/user@UID.service/tmux-spawn-X.scope
    #   /user.slice/user-UID.slice/user@UID.service/app-foo.scope
    # AND: any /dwagent-OTHER-PID leaf belonging to a DIFFERENT
    # process. If we inherited some other daemon's leaf via fork,
    # we should still split off into our own.
    if cur.endswith(f"/{leaf_basename}"):
        return  # already in our own leaf
    # If cur is some non-shell, non-dwagent cgroup (e.g. bare
    # user@UID.service path on a host without subgroups) we still
    # try to migrate — the cost of an unnecessary migration is one
    # mkdir, the cost of NOT migrating when we should is the bug.
    try:
        # exist_ok=True: in the extremely unlikely case the dir is
        # already there (re-execed daemon, debugger restart, …) we
        # just reuse it.
        os.makedirs(leaf_dir, exist_ok=True)
        with open(f"{leaf_dir}/cgroup.procs", "w") as f:
            f.write(f"{os.getpid()}\n")
    except (OSError, PermissionError):
        # Delegation not available, sysfs read-only, container, or
        # cgroup v1. Daemonization continues; the caller is on a
        # platform where the parent-shell-death-kills-daemon problem
        # may still apply, but we have no other primitive to use
        # without taking on a hard dbus / systemd-run dependency.
        pass


def _reap_empty_cgroup_leaves() -> None:
    """rmdir every EMPTY ``dwagent-*`` leaf cgroup under
    ``user@UID.service``.

    ``_detach_from_session_scope`` creates a ``dwagent-<name>-<pid>`` leaf per
    agent and migrates the agent into it. Nothing removed those leaves when
    the agent died: the kernel does NOT auto-reap an empty cgroup directory,
    and systemd's user-manager does not rmdir delegated leaves either. They
    accumulated without bound (272 stale leaves observed 2026-05-23,
    ``nr_dying_descendants`` climbing), which made cgroup operations slower /
    flakier under pressure — the root of the intermittent
    ``test_daemonize_cgroup_detach`` failures.

    A process on cgroup v2 cannot rmdir its OWN leaf while it lives in it (it
    can't move back into ``user@UID.service`` — the no-internal-processes rule
    forbids it once subtree_control is set). So we "reap the corpses": each
    agent removes every OTHER empty dwagent leaf at startup and shutdown. Its
    own leaf is reaped by the next agent's sweep after it dies.

    Safety: only EMPTY leaves (``cgroup.procs`` has no PIDs) are removed — a
    leaf with a live process is never touched. Best-effort: never raises;
    individual rmdir failures (race with another reaper, momentary
    non-emptiness) are ignored.
    """
    user_service = (
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service"
    )
    try:
        entries = os.listdir(user_service)
    except OSError:
        return  # no cgroup v2 / no delegation / not present — nothing to do
    for name in entries:
        if not name.startswith("dwagent-"):
            continue
        leaf = f"{user_service}/{name}"
        procs_file = f"{leaf}/cgroup.procs"
        try:
            # A leaf is reapable only if it has NO processes. Reading
            # cgroup.procs is the authoritative check (a non-empty file ==
            # live members). os.rmdir refuses a non-empty cgroup anyway, so
            # this is belt-and-suspenders against killing a live agent's leaf.
            with open(procs_file) as f:
                if f.read().strip():
                    continue  # still has procs — leave it alone
            os.rmdir(leaf)
        except OSError:
            # Not a dir, already gone, raced with another reaper, or briefly
            # non-empty — all benign. Skip and move on.
            continue


def daemonize(
    name: str,
    log_dir: Path | None = None,
    *,
    await_readiness: bool = False,
    detach: Callable[[str], None] | None = None,
    reap: Callable[[], None] | None = None,
) -> int | None:
    """Daemonize the current process (Unix double-fork + cgroup detach).

    Args:
        name: Agent name for log files
        log_dir: Directory for log files (default: the logs directory under the deep_worker home)
        await_readiness: Opt in to the startup handshake described below.

            MUST stay opt-in. ``web_daemon.main()`` also calls this
            function, but it owns a SEPARATE readiness channel of its own
            (``--ready-fd``, written right after ``daemonize()`` returns).
            Blocking the launching process here for an agent-style report
            that the web server never sends made ``web_daemon.start()``'s
            ``child.wait(timeout=30)`` expire — four web_daemon tests plus
            the installed-distribution test went red. Default False keeps
            every non-agent caller on the original print-and-exit path.

    Returns:
        With ``await_readiness=False`` (default): the historical
        behaviour — the original process prints "starting in background"
        and ``sys.exit(0)``s immediately after the first fork; the
        grandchild gets ``None``.

        With ``await_readiness=True``, in the daemonized grandchild: the
        write end (fd) of a readiness socketpair, inherited across both
        forks (``os.fork()`` copies the whole fd table — unlike
        ``subprocess.Popen``, no ``pass_fds`` is needed). The grandchild
        MUST, exactly once, before exiting:

        * send ``{"ready": true}\\n`` once the agent has reached its
          normal running state, OR
        * send ``{"error": "<cause>"}\\n`` if agent initialization failed
          (the process itself stays alive — AgentStatus.ERROR,
          docs/DESIGN_FAIL_FAST_AGENT_STARTUP.md — only the terminal
          report differs),

        then close the fd. The launching terminal (this function's
        caller, in the ORIGINAL process) blocks on this same channel and
        exits 0 on ``{"ready": true}``, exits 1 on ``{"error": ...}`` or
        on EOF (grandchild died before reporting either). ``None`` in
        the original process — it never reaches this return, since it
        ``sys.exit()``s once the channel resolves.
    """
    # Set up log directory
    if log_dir is None:
        log_dir = deep_worker_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # ``mkdir`` above inherits the CALLER's umask, and the parent process that
    # kicked off ``deep-worker --restart`` may itself run under umask 0
    # (older daemon, systemd unit, misconfigured shell), giving 0o777. Force
    # 0o755 explicitly so the log directory is not world-writable regardless
    # of the caller's umask. See ``docs/DESIGN_FIX_DAEMONIZE_UMASK.md``.
    try:
        log_dir.chmod(0o755)
    except OSError:
        # If we cannot chmod (foreign owner, read-only fs, …) proceed anyway;
        # the daemon still needs to run. The umask fix below covers all future
        # creates inside the daemon itself.
        pass

    # Log file paths
    stdout_log = log_dir / f"{name}.out.log"
    stderr_log = log_dir / f"{name}.err.log"
    err_log = stderr_log

    # Readiness channel, created BEFORE the first fork so both the parent
    # (which keeps ``reader``) and the grandchild (which inherits ``writer``
    # across both forks) share it. Mirrors web_daemon.start()'s socketpair,
    # adapted to a same-binary double-fork instead of subprocess.Popen — a
    # plain os.fork() needs no pass_fds, every open fd is already shared.
    # Only created when the caller opted in: a non-agent caller
    # (web_daemon) has its own channel and must not be made to wait on
    # this one. See ``await_readiness`` in the docstring.
    reader: socket.socket | None = None
    writer: socket.socket | None = None
    if await_readiness:
        reader, writer = socket.socketpair()

    # CRITICAL: Flush Python's stdio buffers BEFORE forking. Otherwise any
    # text that a caller printed before calling daemonize() still sits in
    # the Python-level buffer and gets duplicated into every forked process
    # (parent + child #1 + grandchild = same line printed 3 times).
    sys.stdout.flush()
    sys.stderr.flush()

    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            if reader is not None and writer is not None:
                # Parent process: close our copy of the write end (only the
                # grandchild must hold it — otherwise the kernel never
                # reports EOF if the grandchild dies without reporting
                # first, because this process would still be holding a
                # writable copy open).
                writer.close()
                _wait_for_readiness(reader, name, stdout_log, err_log)
            else:
                # Legacy path, byte-identical to the pre-handshake
                # behaviour: report the launch optimistically and leave.
                print(
                    f"Agent '{name}' starting in background (PID will be "
                    f"written to /tmp/deep_worker_{name}.pid)"
                )
                print(f"Logs: {stdout_log}")
                sys.stdout.flush()
                sys.exit(0)
    except OSError as e:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        print(f"Fork #1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # In the first child (pre-second-fork) AND the final grandchild: close
    # the read end, we only ever write to this channel from here on.
    if reader is not None:
        reader.close()

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    # Set the daemon umask to the normal user default (0o022) — files 644,
    # dirs 755, matching /etc/login.defs and interactive-shell behaviour. Do
    # NOT use the historical umask(0): it makes every implicit-mode file/dir
    # world-writable (666/777). Private paths are protected by explicit
    # chmod in storage_paths.py and are unaffected by this mask.
    os.umask(DAEMON_UMASK)

    # Reap leftover empty dwagent-* leaf cgroups from agents that have since
    # died. Nothing else removes them (kernel doesn't auto-rmdir empty cgroup
    # dirs; systemd doesn't reap delegated leaves), so without this they leak
    # without bound and eventually make cgroup ops flaky (2026-05-23). A
    # process can't rmdir its own live leaf, so each agent reaps the corpses
    # left by previous agents — including, on the next launch, our own.
    (reap or _reap_empty_cgroup_leaves)()

    # Detach from the launching shell's systemd cgroup (session-scope,
    # tmux-spawn-scope, app-scope, …) so that scope teardown does not
    # take the daemon down with it. See _detach_from_session_scope's
    # docstring for the why and the failure-mode rationale. Done after
    # setsid() but before the second fork, so the migrated cgroup is
    # what the final daemon process inherits.
    (detach or _detach_from_session_scope)(name)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # First child exits. Its copy of ``writer`` closes automatically
            # on process exit — only the surviving grandchild's copy matters
            # for the EOF/"ready"/"error" contract.
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # IMPORTANT (fork+thread ordering): os.fork() copies ONLY the calling
    # thread, so the writer threads MUST be created HERE — after the final
    # fork, in the surviving daemon. Creating them before the fork would leave
    # a dead, un-copied thread in the daemon and a full pipe would then block
    # the agent's first write() forever.
    from deep_worker_cli.process.log_rotation import start_pipe_writer

    # Point stdout/stderr at the WRITE end of a pipe instead of straight at the
    # log file. A per-stream daemon thread drains the pipe and appends through a
    # RotatingLogWriter that enforces a hard 512MB-active + 1-backup cap, so the
    # on-disk total per stream NEVER exceeds 1 GiB — enforced in-process, with no
    # logrotate/cron needed. Captures print, logging, AND C-level writes to fd
    # 1/2 alike. See log_rotation.py and a related internal incident analysis (not included here).
    stdout_write_fd, _stdout_log_thread = start_pipe_writer(str(stdout_log))
    stderr_write_fd, _stderr_log_thread = start_pipe_writer(str(stderr_log))

    # Redirect stdin to /dev/null
    stdin_fd = os.open(os.devnull, os.O_RDONLY)

    # Duplicate file descriptors. fd 1/2 now hold a dup of each pipe's write
    # end, which keeps that write end open until the process exits — so the
    # writer threads see EOF only at teardown.
    os.dup2(stdin_fd, sys.stdin.fileno())
    os.dup2(stdout_write_fd, sys.stdout.fileno())
    os.dup2(stderr_write_fd, sys.stderr.fileno())

    # Close the redundant fds. The dup'd copies on fd 0/1/2 keep stdin/stdout/
    # stderr alive; closing these extra write-end copies still leaves fd 1/2
    # holding a write end, so the pipes are NOT prematurely EOF'd.
    os.close(stdin_fd)
    os.close(stdout_write_fd)
    os.close(stderr_write_fd)

    # ``detach()``, NOT ``fileno()``: returning ``writer.fileno()`` would
    # leave the socket OBJECT unreferenced the moment this function returns,
    # and ``socket.__del__`` then closes the underlying fd at the next GC —
    # at C level, invisibly to any Python-level mock. The grandchild's later
    # readiness write would land on a closed (or, worse, recycled) descriptor
    # and the launching terminal would see EOF instead of the real cause.
    #
    # Caught by the step-5 end-to-end check from an installed wheel, NOT by the
    # unit tests: those hold their own reference to the socketpair, which
    # masks the leak exactly the way production does not.
    #
    # ``detach()`` hands the fd over and marks the object closed WITHOUT
    # closing it, so the descriptor outlives the object. Ownership moves to
    # AgentProcess._send_ready_event, which wraps it back into a socket and
    # closes it after the one-shot report.
    return writer.detach() if writer is not None else None


def _wait_for_readiness(
    reader: socket.socket, name: str, stdout_log: Path, err_log: Path
) -> None:
    """Block the ORIGINAL (launching) process until the daemonized
    grandchild reports its startup outcome, then print and ``sys.exit``.

    Event-driven, no polling: one blocking ``select`` + ``recv`` per
    message, exactly like ``web_daemon.start()``'s wait loop. Three
    outcomes, matching the grandchild's contract documented on
    ``daemonize()``:

    * ``{"ready": true}``  -> agent is up and running.   exit 0.
    * ``{"error": "..."}`` -> agent init failed; the process is still
      alive in AgentStatus.ERROR (the registered-but-broken state the
      No-Agent-Error-State policy requires — see
      docs/DESIGN_FAIL_FAST_AGENT_STARTUP.md). exit 1, printing the
      cause VERBATIM plus the log path, so the user never has to go
      read the log just to learn what broke.
    * EOF with no message at all -> the grandchild died before it
      could report either way (crash, SIGKILL, OOM). exit 1.

    Never raises: this function always ends in ``sys.exit()``.
    """
    buffer = b""
    try:
        while True:
            if not select.select([reader], [], [], READY_TIMEOUT)[0]:
                print(
                    f"Agent '{name}': startup timed out after "
                    f"{READY_TIMEOUT:.0f}s; see {err_log}",
                    file=sys.stderr,
                )
                sys.exit(1)
            data = reader.recv(4096)
            if not data:
                print(
                    f"Agent '{name}': process exited before reporting "
                    f"startup outcome; see {err_log}",
                    file=sys.stderr,
                )
                sys.exit(1)
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # malformed line — keep waiting for a real one
                if "error" in event:
                    print(
                        f"Agent '{name}' failed to initialize: {event['error']}\n"
                        f"Log: {err_log}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if event.get("ready"):
                    print(
                        f"Agent '{name}' started (PID will be written to "
                        f"/tmp/deep_worker_{name}.pid)"
                    )
                    print(f"Logs: {stdout_log}")
                    sys.stdout.flush()
                    sys.exit(0)
    finally:
        reader.close()
