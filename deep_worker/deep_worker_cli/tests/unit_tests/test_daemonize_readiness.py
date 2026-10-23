"""Regression tests for the readiness channel ``daemon.daemonize()`` sets
up before forking, and for ``daemon._wait_for_readiness()`` — the blocking
wait the ORIGINAL (launching) process performs on it.

Background: ``deep-worker <name> ...`` used to return control to the shell
right after the first fork, long before the daemonized grandchild had
actually registered the agent or even attempted ``create_model``/startup.
A silent startup failure (bad API key, broken MCP config, …) left the user
staring at "Agent started" while the process was already dead or stuck in
``AgentStatus.ERROR`` with no indication in the terminal. The fix: a
``socket.socketpair()`` created before the first fork, inherited across
both forks (``os.fork()`` shares the whole fd table, unlike
``subprocess.Popen`` — no ``pass_fds`` needed), through which the
grandchild sends exactly one ``{"ready": true}`` or ``{"error": "..."}``
line before the launching terminal is allowed to return. See
``AgentProcess._report_ready`` / ``_report_startup_error`` /
``_send_ready_event`` for the grandchild side.

These tests exercise ``_wait_for_readiness`` directly against a real
``socket.socketpair()`` (no process fork needed — it only touches the
``reader`` end) for the three outcomes the contract promises: ready,
error, and EOF-without-a-message. A fourth test drives ``daemonize()``
itself with ``os.fork`` mocked (same technique as
``test_daemonize_umask.py``) to confirm the grandchild path actually
returns the writer fd, so ``AgentProcess`` has something to report
through.
"""

from __future__ import annotations

import gc
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_DW_CLI_ROOT = Path(__file__).resolve().parents[2]
if str(_DW_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DW_CLI_ROOT))

from deep_worker_cli.process import daemon  # noqa: E402


def _logs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "test.out.log", tmp_path / "test.err.log"


class TestWaitForReadinessReady:
    def test_ready_message_exits_zero_and_prints_pid_file_hint(
        self, tmp_path, capsys
    ) -> None:
        reader, writer = socket.socketpair()
        stdout_log, err_log = _logs(tmp_path)
        writer.sendall(json.dumps({"ready": True}).encode() + b"\n")
        writer.close()

        with pytest.raises(SystemExit) as exc_info:
            daemon._wait_for_readiness(reader, "test_agent", stdout_log, err_log)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "started" in out
        assert str(stdout_log) in out


class TestWaitForReadinessError:
    def test_error_message_exits_one_and_prints_cause_and_log_path(
        self, tmp_path, capsys
    ) -> None:
        reader, writer = socket.socketpair()
        stdout_log, err_log = _logs(tmp_path)
        writer.sendall(
            json.dumps({"error": "boom: bad API key"}).encode() + b"\n"
        )
        writer.close()

        with pytest.raises(SystemExit) as exc_info:
            daemon._wait_for_readiness(reader, "test_agent", stdout_log, err_log)

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "boom: bad API key" in err
        assert str(err_log) in err


class TestWaitForReadinessEOF:
    def test_eof_without_any_message_exits_one(self, tmp_path, capsys) -> None:
        """The grandchild dying (crash, SIGKILL, OOM) before it can send
        either outcome must be reported as a failure, not hang forever
        and not be mistaken for success."""
        reader, writer = socket.socketpair()
        stdout_log, err_log = _logs(tmp_path)
        writer.close()  # no message at all — immediate EOF

        with pytest.raises(SystemExit) as exc_info:
            daemon._wait_for_readiness(reader, "test_agent", stdout_log, err_log)

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "exited before reporting" in err
        assert str(err_log) in err


class TestWaitForReadinessTimeout:
    def test_timeout_without_any_message_exits_one(self, tmp_path, capsys) -> None:
        """A grandchild that neither reports nor dies (hung init) must not
        block the launching terminal forever — ``READY_TIMEOUT`` bounds
        the wait."""
        reader, writer = socket.socketpair()
        stdout_log, err_log = _logs(tmp_path)
        # Keep writer open (no EOF) but never send anything — forces the
        # select() branch rather than the recv()-returns-empty branch.

        with patch.object(daemon, "READY_TIMEOUT", 0.05):
            with pytest.raises(SystemExit) as exc_info:
                daemon._wait_for_readiness(
                    reader, "test_agent", stdout_log, err_log
                )

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "timed out" in err
        assert str(err_log) in err
        writer.close()


class TestWaitForReadinessMalformedLines:
    def test_malformed_line_is_skipped_until_a_real_message_arrives(
        self, tmp_path, capsys
    ) -> None:
        """A garbage line on the channel (should never happen, but must
        not crash the wait) is skipped; the wait keeps blocking for a
        parseable message instead of exiting on noise."""
        reader, writer = socket.socketpair()
        stdout_log, err_log = _logs(tmp_path)
        writer.sendall(b"not json at all\n")
        writer.sendall(json.dumps({"ready": True}).encode() + b"\n")
        writer.close()

        with pytest.raises(SystemExit) as exc_info:
            daemon._wait_for_readiness(reader, "test_agent", stdout_log, err_log)

        assert exc_info.value.code == 0


class TestDaemonizeReturnsWriterFdInGrandchild:
    """``daemonize()`` MUST hand the grandchild the write end of the
    readiness socketpair (as its return value), and the ORIGINAL process
    MUST block in ``_wait_for_readiness`` instead of exiting immediately
    after the first fork — the regression this whole channel fixes.
    """

    def test_grandchild_receives_a_usable_writer_fd(self, tmp_path) -> None:
        # Fork strategy, same technique as test_daemonize_umask.py:
        #  1st fork: return 0 -> we are "the child", flow continues.
        #  2nd fork: return 0 -> we are "the grandchild", flow continues
        #     to the final return instead of sys.exit(0).
        fork_results = iter([0, 0])

        def _fake_fork() -> int:
            return next(fork_results)

        # Intercept ``socket.socketpair`` to get at the pair the function
        # creates, so the test can inspect the channel afterwards.
        #
        # HISTORY — do NOT "simplify" this back. An earlier version of this
        # test held a strong reference to ``writer`` here and dismissed the
        # fd being closed on return as "a same-process test artifact". That
        # was wrong, and the rationalisation hid a REAL production bug:
        # ``daemonize()`` returned ``writer.fileno()``, the socket object
        # became unreferenced the moment it returned, and GC closed the fd
        # at C level before the grandchild ever reported. The launching
        # terminal then printed "process exited before reporting startup
        # outcome" instead of the actual cause. Caught only by the п.5
        # end-to-end run from an installed wheel.
        #
        # The fix is ``writer.detach()``. ``test_returned_fd_survives_gc``
        # below is the regression oracle for it.
        created: dict[str, socket.socket] = {}
        real_socketpair = daemon.socket.socketpair

        def _tracking_socketpair(*a, **kw):
            reader, writer = real_socketpair(*a, **kw)
            created["writer"] = writer
            return reader, writer

        # Everything from "redirect stdin/stdout/stderr" onward is real fd
        # plumbing (os.open/dup2/close) irrelevant to the readiness-channel
        # contract under test, and pytest's captured streams have no real
        # fileno() to dup2 onto. Stub the pipe-writer startup and the
        # fd-redirect calls so the grandchild path can run to its `return`
        # without touching the test runner's actual stdio.
        with (
            patch.object(daemon.os, "fork", _fake_fork),
            patch.object(daemon.os, "chdir"),
            patch.object(daemon.os, "setsid"),
            patch.object(daemon.os, "umask"),
            patch.object(daemon.os, "open", return_value=0),
            patch.object(daemon.os, "dup2"),
            patch.object(daemon.os, "close"),
            patch.object(daemon.socket, "socketpair", _tracking_socketpair),
            patch(
                "deep_worker_cli.process.log_rotation.start_pipe_writer",
                return_value=(0, None),
            ),
            patch.object(daemon.sys, "stdin"),
            patch.object(daemon.sys, "stdout"),
            patch.object(daemon.sys, "stderr"),
        ):
            writer_fd = daemon.daemonize(
                "test_agent_readiness",
                log_dir=tmp_path,
                await_readiness=True,
                detach=lambda _name: None,
                reap=lambda: None,
            )

            assert writer_fd is not None, (
                "daemonize() must return the write-end fd in the "
                "grandchild so AgentProcess can report ready/error "
                "through it (see ready_fd on AgentProcess.__init__)."
            )
            # ``daemonize()`` hands ownership over via ``detach()``, which
            # marks the socket object closed (``fileno() == -1``) while
            # leaving the DESCRIPTOR open. Both halves matter: the object
            # must have released it, and the fd must still be usable.
            assert created["writer"].fileno() == -1, (
                "daemonize() must detach() the writer, not just read its "
                "fileno() — otherwise the socket object still owns the fd "
                "and GC closes it out from under the grandchild."
            )
            # The fd must still be OPEN at this point — not just a
            # number that happened to be valid when captured. The
            # paired ``reader`` end is already closed by ``daemonize()``
            # itself here (correct per contract: the grandchild only
            # ever writes), so this checks openness via ``fstat``
            # rather than a round-trip ``sendall``, which would raise
            # ``BrokenPipeError`` for an unrelated reason.
            os.fstat(writer_fd)
            os.close(writer_fd)

    def test_returned_fd_survives_gc(self, tmp_path) -> None:
        """REGRESSION (found by the п.5 wheel run, missed by unit tests).

        ``daemonize()`` used to ``return writer.fileno()``. The socket
        object was then unreferenced, and ``socket.__del__`` closed the fd
        at C level on the next collection — so by the time the grandchild
        reported ready/error, it was writing to a closed (potentially
        recycled) descriptor and the launcher saw EOF.

        Holding no reference and forcing a collection is the whole oracle:
        with ``fileno()`` the ``fstat`` below raises EBADF; with
        ``detach()`` it succeeds.
        """
        fork_results = iter([0, 0])

        with (
            patch.object(daemon.os, "fork", lambda: next(fork_results)),
            patch.object(daemon.os, "chdir"),
            patch.object(daemon.os, "setsid"),
            patch.object(daemon.os, "umask"),
            patch.object(daemon.os, "open", return_value=0),
            patch.object(daemon.os, "dup2"),
            patch.object(daemon.os, "close"),
            patch(
                "deep_worker_cli.process.log_rotation.start_pipe_writer",
                return_value=(0, None),
            ),
            patch.object(daemon.sys, "stdin"),
            patch.object(daemon.sys, "stdout"),
            patch.object(daemon.sys, "stderr"),
        ):
            # Deliberately keep NO reference to the socketpair objects.
            writer_fd = daemon.daemonize(
                "test_agent_gc",
                log_dir=tmp_path,
                await_readiness=True,
                detach=lambda _name: None,
                reap=lambda: None,
            )

        assert writer_fd is not None
        gc.collect()

        # Still open after collection → ownership really was transferred.
        os.fstat(writer_fd)
        os.close(writer_fd)

    def test_original_process_blocks_on_readiness_instead_of_exiting_at_fork(
        self, tmp_path
    ) -> None:
        """Before the fix, the parent branch of the first fork printed
        'starting in background' and called ``sys.exit(0)`` immediately
        — it never looked at what the grandchild actually did. Now it
        must block inside ``_wait_for_readiness`` and its exit code must
        reflect the grandchild's reported outcome, not a blind success.
        """
        received: dict[str, object] = {}

        def _fake_wait_for_readiness(reader, name, stdout_log, err_log):
            received["called"] = True
            received["name"] = name
            reader.close()
            sys.exit(0)

        def _fake_fork() -> int:
            return 4321  # parent branch: pid > 0

        with (
            patch.object(daemon.os, "fork", _fake_fork),
            patch.object(daemon, "_wait_for_readiness", _fake_wait_for_readiness),
        ):
            with pytest.raises(SystemExit) as exc_info:
                daemon.daemonize(
                    "test_agent_block", log_dir=tmp_path, await_readiness=True
                )

        assert exc_info.value.code == 0
        assert received.get("called") is True
        assert received.get("name") == "test_agent_block"
