"""The lock that keeps this suite's waits event-driven.

The binding policy is no-polling: a wait must block in the kernel until
the thing it waits for happens, not wake N times a second to ask again.
The production code obeys it under ``test_no_polling_policy_all.py``,
which scans only ``deep_worker_cli/`` — the test tree is outside its
reach, so a rig could quietly reintroduce the pattern the product forbids.

This file closes that gap for the semantic-memory integration suite. It
is deliberately stricter than the production scanner: there, the ban is
on ``while`` + ``sleep``; here it is on ``sleep`` at all. A bare
``time.sleep`` in a test is the same defect wearing a different shape —
"wait a bit and hope" — and it is how a polling loop grows back one line
at a time.
"""

from __future__ import annotations

import ast
import errno
import signal
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SUITE = Path(__file__).parent


def _sleep_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Every ``*.sleep(...)`` / bare ``sleep(...)`` call, with its line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "sleep":
            owner = getattr(func.value, "id", "?")
            found.append((node.lineno, f"{owner}.sleep(...)"))
        elif isinstance(func, ast.Name) and func.id == "sleep":
            found.append((node.lineno, "sleep(...)"))
    return found


def test_rig_has_no_polling_loops() -> None:
    """No file in this suite may wait by sleeping.

    The scan covers the suite's own sources, including this file — the
    ban has no exemption for the thing that enforces it.
    """
    offenders: list[str] = []
    for path in sorted(_SUITE.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, what in _sleep_calls(tree):
            offenders.append(f"{path.name}:{line}: {what}")

    assert offenders == [], (
        "The semantic-memory integration suite must wait on kernel events "
        "(pidfd + select), never by sleeping:\n  "
        + "\n  ".join(offenders)
        + "\nUse conftest.wait_pid_exit / assert_pid_exits instead."
    )


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() is gone from conftest.py — update this lock")


def test_kill_pid_signals_through_the_pidfd_not_the_number() -> None:
    """B7.3: teardown must not signal by pid number.

    A pid number is reusable; a pidfd is not. ``os.kill`` inside the
    teardown helper is therefore the defect itself, not merely a style
    choice — after the target exits, the same number may name a stranger,
    and the escalation path would SIGKILL them.

    Checked statically because the race is real but rare: a test that
    tried to *observe* a reused pid would be flaky, while the property
    that prevents it — signals go through the descriptor — is plain in
    the source.
    """
    tree = ast.parse((_SUITE / "conftest.py").read_text(encoding="utf-8"))
    kill_pid = _function(tree, "kill_pid")

    raw_kills = [
        node.lineno
        for node in ast.walk(kill_pid)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "kill"
        and getattr(node.func.value, "id", None) == "os"
    ]
    assert raw_kills == [], (
        f"kill_pid() signals by pid number at line(s) {raw_kills}; a pid can "
        "be reused between the target's exit and the signal. Send through "
        "the pidfd opened before the first signal instead."
    )

    sends = [
        node
        for node in ast.walk(kill_pid)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pidfd_send_signal"
    ]
    assert len(sends) == 2, (
        "kill_pid() should send exactly SIGTERM and SIGKILL through the "
        f"pidfd; found {len(sends)} pidfd_send_signal call(s)"
    )


def test_one_pidfd_serves_term_wait_kill_wait(monkeypatch) -> None:
    """The whole escalation rides a single descriptor, opened up front.

    The static lock above proves signals are not sent by number; this one
    proves they all reach the *same* identity — the fd is opened before
    the first signal and never reopened, so a pid reused mid-escalation
    cannot be signalled by mistake.
    """
    from . import conftest as rig

    opened: list[int] = []
    signalled: list[tuple[int, int]] = []
    waited: list[int] = []
    closed: list[int] = []

    sentinel_fd = 4242

    def fake_open(pid: int) -> int:
        opened.append(pid)
        return sentinel_fd

    def fake_send(fd: int, sig: int) -> None:
        signalled.append((fd, sig))

    def fake_wait(fd: int, *, timeout_s: float) -> bool:
        waited.append(fd)
        # Survive SIGTERM so the escalation to SIGKILL is exercised; the
        # confirming wait after SIGKILL then reports the exit.
        return len(waited) > 1

    monkeypatch.setattr(rig, "_open_pidfd", fake_open)
    monkeypatch.setattr(rig.signal, "pidfd_send_signal", fake_send)
    monkeypatch.setattr(rig, "_wait_fd_exit", fake_wait)
    monkeypatch.setattr(rig.os, "close", lambda fd: closed.append(fd))

    rig.kill_pid(31337, grace_s=0.1)

    assert opened == [31337], "the pidfd must be opened once, before signalling"
    assert signalled == [
        (sentinel_fd, signal.SIGTERM),
        (sentinel_fd, signal.SIGKILL),
    ], f"both signals must go through the one descriptor; got {signalled}"
    assert waited == [sentinel_fd, sentinel_fd], (
        "both waits must watch that same descriptor — including the "
        f"confirming wait after SIGKILL; got {waited}"
    )
    assert closed == [sentinel_fd]


def test_kill_pid_is_a_noop_for_a_process_that_is_already_gone(monkeypatch) -> None:
    """Nothing is signalled when the pid no longer exists.

    This is the reuse window itself: the process is gone, its number may
    already belong to someone else, and the only safe action is none.
    """
    from . import conftest as rig

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(rig, "_open_pidfd", lambda pid: None)
    monkeypatch.setattr(
        rig.signal,
        "pidfd_send_signal",
        lambda fd, sig: signalled.append((fd, sig)),
    )

    rig.kill_pid(31337)

    assert signalled == []


def test_a_signal_failure_that_is_not_esrch_is_not_swallowed(monkeypatch) -> None:
    """B7.4: only "already gone" is a benign signal failure.

    ``except OSError: return`` would report a successful cleanup for
    EPERM, EINVAL or EBADF alike — cases where the signal never landed and
    the process is still running. A teardown that reports success it did
    not achieve is worse than one that fails: the leak goes unnoticed.
    """
    from . import conftest as rig

    closed: list[int] = []
    sentinel_fd = 4242

    def refuse(fd: int, sig: int) -> None:
        raise PermissionError(errno.EPERM, "not permitted")

    monkeypatch.setattr(rig, "_open_pidfd", lambda pid: sentinel_fd)
    monkeypatch.setattr(rig.signal, "pidfd_send_signal", refuse)
    monkeypatch.setattr(rig.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(PermissionError):
        rig.kill_pid(31337, grace_s=0.1)

    assert closed == [sentinel_fd], "the descriptor must be closed even on failure"


def test_esrch_from_the_signal_stays_a_safe_return(monkeypatch) -> None:
    """The one failure that *is* benign: the target exited already.

    It can die between ``pidfd_open`` and the signal, and that is the
    outcome teardown wanted anyway.
    """
    from . import conftest as rig

    closed: list[int] = []
    sentinel_fd = 4242

    def gone(fd: int, sig: int) -> None:
        raise ProcessLookupError(errno.ESRCH, "no such process")

    monkeypatch.setattr(rig, "_open_pidfd", lambda pid: sentinel_fd)
    monkeypatch.setattr(rig.signal, "pidfd_send_signal", gone)
    monkeypatch.setattr(rig.os, "close", lambda fd: closed.append(fd))

    rig.kill_pid(31337, grace_s=0.1)

    assert closed == [sentinel_fd]


def test_surviving_sigkill_fails_teardown_instead_of_reporting_success(
    monkeypatch,
) -> None:
    """B7.4: the confirming wait must be acted on, not merely performed.

    If the process is still alive after SIGKILL and the grace window, the
    fixture is about to return while a service holding the heavy stack
    keeps running. Silence there turns the rig's "guaranteed dead"
    promise into a wish; the failure has to reach the test report.
    """
    from . import conftest as rig

    closed: list[int] = []
    sentinel_fd = 4242

    monkeypatch.setattr(rig, "_open_pidfd", lambda pid: sentinel_fd)
    monkeypatch.setattr(rig.signal, "pidfd_send_signal", lambda fd, sig: None)
    # Never exits: SIGTERM is survived, and so is SIGKILL.
    monkeypatch.setattr(rig, "_wait_fd_exit", lambda fd, *, timeout_s: False)
    monkeypatch.setattr(rig.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(AssertionError, match="survived SIGKILL"):
        rig.kill_pid(31337, grace_s=0.1)

    assert closed == [sentinel_fd], "the descriptor must be closed even on failure"
