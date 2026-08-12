"""No-network socket-allowlist guard canaries.

The guard (``tests/socket_guard.py``) is injected as source into fresh
interpreters; these canaries prove, in a subprocess (so the test session's
own ``socket`` module is never mutated):

1. external IPv4 / IPv6 / hostname addresses are blocked (AssertionError);
2. localhost IPv4 / IPv6 / ``localhost`` are allowed (real connect attempt
   proceeds — the guard does not reject loopback);
3. both ``socket.create_connection`` and the direct
   ``socket.socket().connect`` path are covered;
4. the guard leaves ``socket.socket`` constructible (the subclass is
   functional), which is what the real RPyC / A2A lifecycle tests rely on.
"""

from __future__ import annotations

import subprocess
import sys

from tests.socket_guard import BLOCKED_ADDRESSES, GUARD_SOURCE


def _run_guard(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", GUARD_SOURCE + body],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""},
        timeout=60,
    )


def _expect_blocked(addr_repr: str) -> None:
    result = _run_guard(
        "import socket\n"
        f"try:\n"
        f"    socket.socket().connect({addr_repr})\n"
        "except AssertionError as e:\n"
        "    assert 'OUTBOUND SOCKET BLOCKED' in str(e), e\n"
        "    print('BLOCKED-OK')\n"
        "else:\n"
        "    raise SystemExit('expected AssertionError, got no exception')\n"
    )
    assert result.returncode == 0, f"guard failed:\n{result.stdout}\n{result.stderr}"
    assert "BLOCKED-OK" in result.stdout


def test_external_ipv4_and_ipv6_and_hostname_blocked() -> None:
    """Canary 1: external addresses raise AssertionError before any connect."""
    for addr in BLOCKED_ADDRESSES:
        _expect_blocked(repr(addr))


def test_loopback_allowed_direct_socket() -> None:
    """Canary 2 + 3: direct ``socket.socket().connect`` to loopback proceeds.

    A real connect to a port with no listener yields ConnectionRefusedError —
    NOT the guard's AssertionError — proving the guard allowed it.
    """
    for host in ("127.0.0.1", "localhost", "::1"):
        result = _run_guard(
            "import socket\n"
            "try:\n"
            f"    socket.socket().connect(({host!r}, 1))\n"
            "except AssertionError as e:\n"
            "    raise SystemExit(f'loopback must not be blocked: {e}')\n"
            "except (ConnectionRefusedError, OSError):\n"
            "    print('ALLOWED-OK')\n"
        )
        assert result.returncode == 0, (
            f"loopback {host!r} should be allowed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert "ALLOWED-OK" in result.stdout


def test_create_connection_guarded_both_ways() -> None:
    """Canary 3: ``socket.create_connection`` is guarded too.

    Blocked for external; allowed (real attempt) for loopback.
    """
    blocked = _run_guard(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=1)\n"
        "except AssertionError as e:\n"
        "    assert 'OUTBOUND SOCKET BLOCKED' in str(e), e\n"
        "    print('BLOCKED-OK')\n"
        "else:\n"
        "    raise SystemExit('expected AssertionError, got no exception')\n"
    )
    assert blocked.returncode == 0, blocked.stderr
    assert "BLOCKED-OK" in blocked.stdout

    allowed = _run_guard(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 1), timeout=1)\n"
        "except AssertionError as e:\n"
        "    raise SystemExit(f'loopback must not be blocked: {e}')\n"
        "except (ConnectionRefusedError, OSError):\n"
        "    print('ALLOWED-OK')\n"
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "ALLOWED-OK" in allowed.stdout


def test_socket_still_constructible_under_guard() -> None:
    """Canary 4: the subclass keeps the real socket surface usable.

    Binding a loopback listener and accepting a connection must work under
    the guard — this is what the RPyC / A2A lifecycle tests depend on.
    """
    result = _run_guard(
        "import socket, threading\n"
        "srv = socket.socket()\n"
        "srv.bind(('127.0.0.1', 0))\n"
        "srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def accept_one():\n"
        "    c, _ = srv.accept()\n"
        "    c.sendall(b'hi')\n"
        "    c.close()\n"
        "    srv.close()\n"
        "th = threading.Thread(target=accept_one)\n"
        "th.start()\n"
        "cli = socket.socket()\n"
        "cli.connect(('127.0.0.1', port))\n"
        "assert cli.recv(2) == b'hi'\n"
        "cli.close()\n"
        "th.join()\n"
        "print('SOCKET-OK')\n"
    )
    assert result.returncode == 0, (
        f"guarded socket must stay usable:\n{result.stdout}\n{result.stderr}"
    )
    assert "SOCKET-OK" in result.stdout
