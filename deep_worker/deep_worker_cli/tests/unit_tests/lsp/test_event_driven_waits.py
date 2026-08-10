"""The supervisor's waits must be event-driven, not polled.

Project policy forbids polling loops (``test_no_polling_policy*.py``
enforces the AST shape). These tests enforce the BEHAVIOUR the policy
exists for: each wait must react to the real kernel event promptly,
rather than waking on a timer or waiting out the full budget.

They exercise ``_wait_socket_or_exit`` / ``_wait_pid_exit`` against
real child processes and a real filesystem — no mocks, no fakes.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.supervisor import (
    _wait_pid_exit,
    _wait_socket_or_exit,
)

#: Generous ceiling: the point is "reacted to the event" vs "waited out
#: the budget", so we assert well below the budget, not a tight bound.
_SLACK_S = 2.0


@pytest.fixture()
def sleeper():
    """A child that outlives the test unless explicitly killed."""
    procs: list[subprocess.Popen] = []

    def _spawn(seconds: str = "30") -> subprocess.Popen:
        p = subprocess.Popen(["sleep", seconds])
        procs.append(p)
        return p

    yield _spawn
    for p in procs:
        p.kill()
        p.wait()


def test_socket_creation_wakes_the_wait_immediately(tmp_path: Path, sleeper):
    """inotify edge: must return as soon as the socket appears.

    If inotify silently degraded, this would wait out the full budget.
    """
    sock = tmp_path / "grpc.sock"
    child = sleeper("30")

    def _create() -> None:
        time.sleep(0.3)
        sock.touch()

    threading.Thread(target=_create, daemon=True).start()

    t0 = time.monotonic()
    result = _wait_socket_or_exit(sock, child.pid, timeout_s=30.0)
    elapsed = time.monotonic() - t0

    assert result == "ready"
    assert elapsed < 0.3 + _SLACK_S, f"took {elapsed:.2f}s — inotify not firing?"


def test_child_death_wakes_the_wait_immediately(tmp_path: Path):
    """pidfd edge: a child that dies without creating the socket."""
    sock = tmp_path / "grpc.sock"
    child = subprocess.Popen(["sleep", "0.3"])
    try:
        t0 = time.monotonic()
        result = _wait_socket_or_exit(sock, child.pid, timeout_s=30.0)
        elapsed = time.monotonic() - t0
    finally:
        child.wait()

    assert result == "exited"
    assert elapsed < 0.3 + _SLACK_S, f"took {elapsed:.2f}s — pidfd not firing?"


def test_wait_reports_timeout_without_overrunning(tmp_path: Path, sleeper):
    sock = tmp_path / "grpc.sock"
    child = sleeper("30")

    t0 = time.monotonic()
    result = _wait_socket_or_exit(sock, child.pid, timeout_s=0.5)
    elapsed = time.monotonic() - t0

    assert result == "timeout"
    assert 0.5 <= elapsed < 0.5 + _SLACK_S


def test_preexisting_socket_returns_without_waiting(tmp_path: Path, sleeper):
    """Closes the create-before-watch race: if the child won the race
    before we armed inotify, we must not block for the full budget."""
    sock = tmp_path / "grpc.sock"
    sock.touch()
    child = sleeper("30")

    t0 = time.monotonic()
    result = _wait_socket_or_exit(sock, child.pid, timeout_s=30.0)
    elapsed = time.monotonic() - t0

    assert result == "ready"
    assert elapsed < _SLACK_S


def test_unrelated_sibling_file_does_not_end_the_wait(tmp_path: Path, sleeper):
    """An inotify event for a DIFFERENT file (e.g. the spawn lock) must
    not be mistaken for readiness — we must keep blocking."""
    sock = tmp_path / "grpc.sock"
    child = sleeper("30")

    def _noise_then_ready() -> None:
        time.sleep(0.2)
        (tmp_path / ".spawn.lock").touch()  # unrelated
        time.sleep(0.4)
        sock.touch()  # the real thing

    threading.Thread(target=_noise_then_ready, daemon=True).start()

    t0 = time.monotonic()
    result = _wait_socket_or_exit(sock, child.pid, timeout_s=30.0)
    elapsed = time.monotonic() - t0

    assert result == "ready"
    # Must NOT have returned at the 0.2s sibling event.
    assert elapsed >= 0.55, f"returned at {elapsed:.2f}s — sibling event leaked"
    assert elapsed < 0.6 + _SLACK_S


def test_wait_pid_exit_reacts_to_death_and_reports_timeout(sleeper):
    dying = subprocess.Popen(["sleep", "0.3"])
    try:
        t0 = time.monotonic()
        assert _wait_pid_exit(dying.pid, timeout_s=30.0) is True
        assert time.monotonic() - t0 < 0.3 + _SLACK_S
    finally:
        dying.wait()

    alive = sleeper("30")
    t0 = time.monotonic()
    assert _wait_pid_exit(alive.pid, timeout_s=0.5) is False
    assert 0.5 <= time.monotonic() - t0 < 0.5 + _SLACK_S
