"""No-network socket-allowlist hook (DESIGN_PROVIDERS_FROM_WHEEL §5.8).

Every in-venv assertion script in the provider-from-wheel artifact tests runs
under an ACTIVE outbound-socket guard: only loopback (127.x / ::1 /
localhost) connections are permitted — RPyC and A2A live on localhost —
anything else raises ``AssertionError``.  This turns the "we promise not to
call the LLM" rule into an enforced invariant.

The guard is injected as source text (``GUARD_SOURCE``) into fresh
interpreters because the artifact tests must never mutate the test-session
``socket`` module.  It replaces ``socket.socket`` with a subclass that
overrides ``connect``/``connect_ex`` (assigning instance attributes is
rejected by CPython: ``'socket' object attribute 'connect' is read-only``)
and also guards ``socket.create_connection`` explicitly.

Canaries (see ``tests/unit_tests/test_socket_guard.py``):
1. external IPv4 / IPv6 addresses are blocked (AssertionError);
2. localhost IPv4 / IPv6 / ``localhost`` are allowed (real connect attempt);
3. both ``socket.create_connection`` and the direct ``socket.socket().connect``
   paths are covered;
4. real RPyC lifecycle / A2A tests keep working under the guard.
"""

GUARD_SOURCE = (
    "import socket as _socket\n"
    "def _chk(addr):\n"
    "    host = addr[0] if isinstance(addr, tuple) else addr\n"
    "    if isinstance(host, str):\n"
    "        if host == 'localhost' or host.startswith('127.') or host == '::1':\n"
    "            return\n"
    "    raise AssertionError(f'OUTBOUND SOCKET BLOCKED: {addr!r}')\n"
    "class _GuardedSocket(_socket.socket):\n"
    "    def connect(self, address):\n"
    "        _chk(address)\n"
    "        return super().connect(address)\n"
    "    def connect_ex(self, address):\n"
    "        _chk(address)\n"
    "        return super().connect_ex(address)\n"
    "_socket.socket = _GuardedSocket\n"
    "_orig_create_connection = _socket.create_connection\n"
    "def _guarded_create_connection(address, *a, **k):\n"
    "    _chk(address)\n"
    "    return _orig_create_connection(address, *a, **k)\n"
    "_socket.create_connection = _guarded_create_connection\n"
)

# Addresses the guard must allow (loopback family) and must block.
ALLOWED_ADDRESSES = ("localhost", "127.0.0.1", "::1")
BLOCKED_ADDRESSES = (
    ("8.8.8.8", 53),  # external IPv4
    ("2001:4860:4860::8888", 53),  # external IPv6
    ("example.com", 443),  # external hostname
)
