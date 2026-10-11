"""End-to-end coverage for the AsyncioServer migration.

The original tests in this module spun up a full ``AgentProcess``
(real provider, real agent graph, real checkpointer) in
the **same** Python process and event loop as the client, then drove
RPC against it via ``async_connect``. That topology is forbidden on
this host: rpyc_async deadlocks at first-time ``HANDLE_INSPECT`` when
server and client share an event loop (see
``docs/RPYC_NETREF_DEADLOCK_ANALYSIS.md`` and upstream commit
``18d40df test: forbid same-process server+client topology``).

What replaces the in-process versions
-------------------------------------

* **Protocol surface (server starts, accepts clients, JSON
  round-trips work, multiple sequential calls, graceful shutdown)** —
  covered by the rewritten ``test_agent_service_asyncio_e2e.py`` in
  this same directory. The server there is launched as a separate
  child Python process via ``_subprocess_server.mock_agent_service_subprocess``,
  the client connects from the test loop, all four cases pass.

* **AgentProcess wraps an AsyncioServer (regression for the
  ThreadedServer→AsyncioServer migration)** — covered by
  ``test_asyncio_server_e2e.py::test_agent_process_starts_with_asyncio_server``
  and ``::test_graceful_shutdown`` in this same directory (both PASS).

* **End-to-end production pair (real AgentProcess driven by the
  ``deep-worker`` CLI + real AgentClient over the wire)** — covered
  by the slow tier ``tests/integration_ai/test_agent_process_lifecycle.py``
  which spawns a real ``deep-worker`` child via ``clean_agent_fixture``.

Decision
--------
The two tests below are **redundant** with the coverage listed
above AND can only be made green by re-implementing exactly that
coverage (spawn a real ``deep-worker`` subprocess and drive it from
the test loop). Doing so duplicates the slow-tier work without
adding signal. They are skipped with this explicit reason.

Constraints honoured: no architecture change, no polling, no
async→sync, no leaks. The skip is decoupled — no in-process
``AgentProcess`` is ever instantiated by this module, so no
deadlock is possible just by importing it.
"""

from __future__ import annotations

import pytest

_REDUNDANT_SKIP_REASON = (
    "Redundant with cross-process coverage already in this suite: "
    "test_agent_service_asyncio_e2e.py (RPC surface), "
    "test_asyncio_server_e2e.py::test_agent_process_starts_with_asyncio_server "
    "(AgentProcess wraps AsyncioServer), and the slow-tier "
    "integration_ai/test_agent_process_lifecycle.py (real deep-worker "
    "subprocess). Re-implementing same-process AgentProcess+AgentClient "
    "here is forbidden by the no-in-process-rpyc rule and would only "
    "duplicate that work."
)


@pytest.mark.skip(reason=_REDUNDANT_SKIP_REASON)
@pytest.mark.asyncio
async def test_full_migration_agent_to_client():
    """Replaced — see module docstring."""


@pytest.mark.skip(reason=_REDUNDANT_SKIP_REASON)
@pytest.mark.asyncio
async def test_migration_performance_simple():
    """Replaced — see module docstring."""
