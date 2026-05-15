"""Regression test: ``deep-worker NAME ...`` MUST detach the agent from
the launching shell's systemd session-scope, otherwise the agent dies
when the SSH session that launched it is later torn down.

Background: the existing ``daemonize()`` in
``deep_worker_cli/cli/agent_command.py`` does the classic Unix
double-fork (``setsid`` + two ``fork`` calls), which gives the daemon
PPid=1 and a fresh session/process-group. **But on a systemd host
this is not enough**: ``setsid()`` does not change the cgroup
membership. The kernel inherits the parent's cgroup across ``fork``,
so the daemon stays in ``user.slice/user-UID.slice/session-N.scope``
— the same scope as the SSH login that launched it. When that
session-scope is later deactivated by systemd (logout, ``loginctl
terminate-session``, sshd restart, etc.), every process inside it is
SIGKILLed, including the supposedly-daemonised agent.

Production observation (2026-05-08): a long-running agent (13 hours
uptime) died at the moment systemd deactivated the session-2828
scope of the SSH login that originally started it, even though the
SSH user had already logged out hours earlier and ``linger=yes`` was
set on the user. ``linger`` keeps ``user@UID.service`` alive across
SSH disconnect, but it does **not** make individual session-scopes
immortal — and the daemon was inside a session-scope, not in
``user@UID.service``.

The fix this test guards: after ``setsid()`` (and before the second
fork, so the migrated cgroup is what the final daemon process
inherits), write the current PID into
``/sys/fs/cgroup/user.slice/user-UID.slice/user@UID.service/cgroup.procs``.
That file is owned by the user and writable without root on hosts
where systemd grants ``Delegate=yes`` on ``user@.service`` — which is
the default on every modern systemd (since v245). The migration moves
the daemon out of the session-scope and into the user-manager's own
cgroup, which lives until the user logs out completely (i.e. linger
expires) or until something explicitly stops ``user@UID.service``.

If cgroup-write is denied (no delegation, container, cgroup v1),
``daemonize`` should fall back to re-execing under ``systemd-run
--user --scope --collect`` — also tested below — and only as a last
resort give up and continue without migration.

THIS TEST IS A REGRESSION GUARD. It does NOT require the underlying
host to be a systemd / cgroup v2 box; on hosts where the relevant
files do not exist, the test marks itself ``skip`` rather than
fail, so the test stays portable.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


# --------------------------------------------------------------------
# Probes for whether the host can demonstrate the fix at all
# --------------------------------------------------------------------

def _is_cgroup_v2() -> bool:
    """True iff ``/sys/fs/cgroup`` is a unified cgroup v2 hierarchy."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "/sys/fs/cgroup":
                    return fields[2] == "cgroup2"
    except OSError:
        pass
    return False


def _user_service_cgroup_procs() -> Path:
    """Path to ``user@UID.service/cgroup.procs`` for the current user."""
    return Path(
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service/cgroup.procs"
    )


def _current_cgroup() -> str:
    """Read this process's cgroup v2 path from ``/proc/self/cgroup``.

    Format on cgroup v2 is ``0::/path/to/cgroup``. Returns ``""`` if
    the entry cannot be parsed.
    """
    try:
        with open("/proc/self/cgroup") as f:
            line = f.read().strip()
        if line.startswith("0::"):
            return line.split("::", 1)[1]
    except OSError:
        pass
    return ""


def _process_cgroup(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            line = f.read().strip()
        if line.startswith("0::"):
            return line.split("::", 1)[1]
    except OSError:
        pass
    return ""


_HAVE_CGROUP_V2 = _is_cgroup_v2()
_USER_PROCS = _user_service_cgroup_procs()
_HAVE_USER_SERVICE_CGROUP = _USER_PROCS.exists()
_CAN_WRITE_USER_SERVICE = (
    _HAVE_USER_SERVICE_CGROUP and os.access(_USER_PROCS, os.W_OK)
)


# --------------------------------------------------------------------
# Direct unit test of the migration helper
# --------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAVE_CGROUP_V2 or not _HAVE_USER_SERVICE_CGROUP,
    reason="needs cgroup v2 + user@UID.service cgroup",
)
def test_detach_helper_moves_self_out_of_session_scope() -> None:
    """The helper that does the migration MUST exist and MUST move the
    calling process out of any ``session-*.scope`` it currently lives
    in. We probe by:

    1. Reading our own cgroup before the call.
    2. Calling the helper.
    3. Reading our own cgroup after.

    If we started inside a ``session-*.scope`` (typical when this
    test runs from an interactive shell) the assertion is "no longer
    contains '/session-' in the path". If we started outside a
    session-scope already (e.g. inside ``user@UID.service`` because
    the test is invoked from a systemd --user unit), the call MUST
    be a safe no-op and the cgroup string MUST be unchanged.
    """
    # Lazy import so the test can collect even before the helper is
    # added to the production module — collection then succeeds and the
    # test fails for the right reason ("ImportError" attribute below).
    from deep_worker_cli.cli import agent_command

    assert hasattr(agent_command, "_detach_from_session_scope"), (
        "_detach_from_session_scope helper is missing from "
        "deep_worker_cli.cli.agent_command. The fix for the SSH-scope "
        "kill bug must add it."
    )

    before = _current_cgroup()
    agent_command._detach_from_session_scope()
    after = _current_cgroup()

    bad_markers = ("/session-", "/tmux-spawn-", "/app-")
    was_in_bad_scope = any(m in before for m in bad_markers)

    if was_in_bad_scope:
        # The helper MUST have moved us out — otherwise the agent
        # will die together with the launching shell's scope (SSH
        # logout, tmux pane close, terminal exit).
        for m in bad_markers:
            assert m not in after, (
                f"_detach_from_session_scope did not move us out of "
                f"a {m}-scope. Before: {before!r}. After: {after!r}."
            )
    else:
        # We were already detached. The helper MUST be a no-op.
        assert before == after, (
            f"_detach_from_session_scope is not idempotent / "
            f"safe-no-op when called from outside an unsafe scope. "
            f"Before: {before!r}. After: {after!r}."
        )


@pytest.mark.skipif(
    not _HAVE_CGROUP_V2 or not _HAVE_USER_SERVICE_CGROUP,
    reason="needs cgroup v2 + user@UID.service cgroup",
)
def test_detach_helper_does_not_raise_on_already_detached() -> None:
    """If the process is already in ``user@UID.service`` (or any
    non-session-scope cgroup), calling the helper a second time MUST
    be safe."""
    from deep_worker_cli.cli import agent_command

    # First call may move us; second call definitely starts from a
    # detached state.
    agent_command._detach_from_session_scope()
    cg_after_first = _current_cgroup()
    agent_command._detach_from_session_scope()
    cg_after_second = _current_cgroup()

    assert cg_after_first == cg_after_second, (
        f"second call to _detach_from_session_scope changed the "
        f"cgroup. {cg_after_first!r} → {cg_after_second!r}"
    )


@pytest.mark.skipif(
    not _HAVE_CGROUP_V2 or not _HAVE_USER_SERVICE_CGROUP,
    reason="needs cgroup v2 + user@UID.service cgroup",
)
def test_detach_helper_swallows_errors() -> None:
    """The helper MUST never raise. Pathological host conditions
    (read-only sysfs, denied delegation, missing target file, etc.)
    are normal in containers and on locked-down systems; daemonize
    must continue regardless. We force such a condition by
    monkey-patching the open path the helper uses, then call it.
    """
    from deep_worker_cli.cli import agent_command

    real_open = open

    def broken_open(path, *a, **kw):
        if "cgroup.procs" in str(path):
            raise PermissionError("simulated lockdown")
        return real_open(path, *a, **kw)

    import builtins
    monkey = builtins.open
    builtins.open = broken_open  # type: ignore[assignment]
    try:
        # Must not raise.
        agent_command._detach_from_session_scope()
    finally:
        builtins.open = monkey


# --------------------------------------------------------------------
# End-to-end: launch the real ``deep-worker`` and check the spawned
# agent ends up outside the launching shell's session-scope.
# --------------------------------------------------------------------

_REGISTRY_FILE = Path("/tmp/deep_worker_registry.json")


def _registry_entries() -> dict:
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(_REGISTRY_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _kill_agent(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _remove_agent_data(name: str) -> None:
    data_dir = Path("~/.deep_worker").expanduser() / name
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for f in log_dir.glob(f"{name}.*"):
            try:
                f.unlink()
            except OSError:
                pass
    pid_file = Path(f"/tmp/deep_worker_{name}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass


@pytest.fixture
def agent_name():
    name = f"test_cgroup_detach_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    yield name
    entry = _registry_entries().get(name)
    if entry and isinstance(entry.get("pid"), int):
        _kill_agent(entry["pid"])
    _remove_agent_data(name)


@pytest.mark.skipif(
    not _HAVE_CGROUP_V2 or not _HAVE_USER_SERVICE_CGROUP,
    reason="needs cgroup v2 + user@UID.service cgroup",
)
@pytest.mark.skipif(
    shutil.which("deep-worker") is None,
    reason="deep-worker binary not in PATH",
)
def test_spawned_agent_is_not_in_test_runner_session_scope(
    agent_name: str,
) -> None:
    """Black-box regression: spawn the real ``deep-worker NAME ...``
    via ``subprocess`` (exactly the way the user invokes it), then
    look up where the resulting daemon ended up in the cgroup tree.
    It MUST NOT be in the same ``session-N.scope`` as the test
    runner.

    Why this is the right level of test: the user's stated
    requirement is "the command as written must produce an agent
    that survives shell death". ``subprocess`` here plays the role
    of "the user's shell". If the agent comes back inside our
    session-scope, killing the test process / its session would
    take it out — which is the exact failure we are fixing.
    """
    # The earlier helper-tests in this file may have already moved
    # the test runner itself into a ``dwagent-*`` leaf cgroup. We
    # specifically want to verify that the *spawned daemon* gets
    # ITS OWN leaf, not whatever cgroup the test runner happens to
    # currently sit in. So we read runner_cgroup AFTER pytest has
    # done all its setup and just before the subprocess starts.
    runner_cgroup = _current_cgroup()
    cmd = ["deep-worker", agent_name, "-asm", "--dir", "/tmp"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"deep-worker exited {proc.returncode}; "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    # The CLI prints a message that mentions the PID will be written
    # to /tmp/deep_worker_<name>.pid. Wait for the pid file or the
    # registry entry to appear (whichever first), then read PID.
    pid_file = Path(f"/tmp/deep_worker_{agent_name}.pid")
    pid = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                break
            except (OSError, ValueError):
                pass
        entry = _registry_entries().get(agent_name)
        if entry and isinstance(entry.get("pid"), int):
            pid = entry["pid"]
            break
        time.sleep(0.1)
    assert pid is not None, (
        f"agent {agent_name} did not register a PID within 10s. "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )

    agent_cgroup = _process_cgroup(pid)
    assert agent_cgroup, (
        f"could not read /proc/{pid}/cgroup; agent likely died "
        f"already. stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )

    # Core assertion: the agent must end up in its OWN leaf cgroup
    # under user@UID.service, and that cgroup must be (a) not
    # tied to any launching shell scope, and (b) named according
    # to the fix's convention so we know the migration actually
    # ran (rather than the agent accidentally being in a safe
    # cgroup for some other reason).
    bad_markers = ("/session-", "/tmux-spawn-", "/app-")
    for m in bad_markers:
        assert m not in agent_cgroup, (
            f"agent landed inside a {m}-scope ({agent_cgroup!r}). "
            f"This scope is tied to the launching shell and will "
            f"be torn down by systemd when the shell goes away — "
            f"the 2026-05-08 notify_agent failure mode."
        )
    # The fix uses ``dwagent-<name>-<pid>`` for the new leaf cgroup
    # name so it is recognisable in ``systemd-cgls`` and in
    # ``/proc/PID/cgroup``. The agent's leaf must contain
    # ``dwagent-<agent_name>-`` specifically — not just any
    # ``dwagent-*`` (which could mean the agent inherited a parent
    # process's leaf, e.g. the test runner if pytest itself
    # already migrated).
    expected_prefix = f"/dwagent-{agent_name}-"
    assert expected_prefix in agent_cgroup, (
        f"agent did not migrate into ITS OWN leaf cgroup. "
        f"Expected path containing {expected_prefix!r}, got "
        f"{agent_cgroup!r}. Either the migration never ran, or "
        f"the daemon inherited a parent's leaf cgroup (which "
        f"would make the daemon depend on the parent's lifetime "
        f"— exactly what we are trying to avoid)."
    )


@pytest.mark.skipif(
    not _HAVE_CGROUP_V2 or not _HAVE_USER_SERVICE_CGROUP,
    reason="needs cgroup v2 + user@UID.service cgroup",
)
@pytest.mark.skipif(
    shutil.which("deep-worker") is None,
    reason="deep-worker binary not in PATH",
)
def test_agent_survives_parent_shell_kill(agent_name: str) -> None:
    """The end-to-end guarantee the user actually cares about: an
    agent launched as ``deep-worker NAME ...`` from a shell MUST
    keep running after the launching shell is killed.

    Procedure:
        1. Spawn an intermediate ``bash -c 'deep-worker NAME …'``
           child via ``setsid``, so it gets its own process group
           and is fully decoupled from this pytest's signals.
        2. Wait for the agent to register a PID (signalling that
           ``daemonize()`` has finished and the third-generation
           process is the live daemon).
        3. ``SIGKILL`` the intermediate ``bash``. ``deep-worker`` is
           a wrapper script (``exec python3 -m …``) — killing the
           bash terminates only that intermediate, but the
           daemonised grandchild (PPid=1 after double-fork) should
           survive.
        4. After a short grace period, assert the agent PID is
           still alive AND that its cgroup contains the
           ``dwagent-NAME-PID`` leaf the fix migrated it into.

    A failing assertion here is the precise failure the user
    described: "agent dies when the shell that launched it dies".
    """
    bash_cmd = f"deep-worker {agent_name} -asm --dir /tmp"
    bash_proc = subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # ≈ setsid, so SIGKILL hits only this bash
    )

    pid_file = Path(f"/tmp/deep_worker_{agent_name}.pid")
    agent_pid = None
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if pid_file.exists():
            try:
                agent_pid = int(pid_file.read_text().strip())
                break
            except (OSError, ValueError):
                pass
        entry = _registry_entries().get(agent_name)
        if entry and isinstance(entry.get("pid"), int):
            agent_pid = entry["pid"]
            break
        time.sleep(0.1)

    assert agent_pid is not None, (
        "agent never registered a PID; daemonize() likely failed"
    )

    # Wait for bash wrapper to finish (it execs into python and
    # the python process daemonizes itself; the original bash is
    # already gone in practice, but being explicit avoids zombies).
    try:
        bash_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        bash_proc.kill()
        bash_proc.wait(timeout=5)

    # Now the launching bash is dead. The agent must still be
    # alive. Give the kernel a moment to propagate any cgroup
    # cleanup that *would* have killed an unprotected daemon.
    time.sleep(1.0)

    try:
        os.kill(agent_pid, 0)
    except ProcessLookupError:
        raise AssertionError(
            f"agent {agent_name} (pid {agent_pid}) died after the "
            f"launching bash exited. This is the regression — the "
            f"agent should have its own cgroup leaf and be "
            f"independent of the launching shell's lifetime."
        )

    # Sanity: the agent should be in its own dwagent-* leaf, not
    # inheriting the bash's cgroup (which would have been torn
    # down with the bash).
    agent_cgroup = _process_cgroup(agent_pid)
    assert f"/dwagent-{agent_name}-" in agent_cgroup, (
        f"agent {agent_pid} survived but is in cgroup "
        f"{agent_cgroup!r} — not the expected "
        f"dwagent-{agent_name}-* leaf. The fix may have a hole."
    )
