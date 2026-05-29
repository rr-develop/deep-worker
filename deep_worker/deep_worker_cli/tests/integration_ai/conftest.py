"""Per-suite conftest for ``tests/integration_ai`` — owns the agent-cleanup
infrastructure.

Cleanup policy
==============
The user's directive is exact and inviolable::

    passed_test     + created_by_fixture  →  terminate the agent
    failed_test     + created_by_fixture  →  preserve for `dwcli` debug
    pre-existing                          →  never touch

It is implemented in two layers:

  * **Per-fixture**: each agent-spawning fixture finalizer reads
    ``request.node.rep_call.passed`` and either terminates or preserves.
    This is the fast path that handles all normal pytest runs.

  * **Safety net**: ``pytest_sessionfinish`` sweeps every agent that
    fixtures registered during this session. Any tracked agent whose
    test passed but whose finalizer never ran (e.g. because pytest
    itself was SIGKILL'd by an external timeout) is killed here.

Both layers go through :mod:`tests.integration_ai.cleanup_policy`, which
is covered by hermetic tests under
``tests/integration/test_fixture_cleanup_policy.py``.

Architectural notes
-------------------
* Strictly event-driven: ``register_test_agent`` is called once per
  spawn; ``pytest_runtest_makereport`` is hooked once per test; the
  sweep runs once per session. No polling, no timers.

* No memory leak: ``_AGENT_TRACKER`` is module-global to the conftest
  but bounded by the number of ``test_*`` agents per session, and
  cleared implicitly when the pytest process exits.

* Defence in depth: the policy module silently drops names that do not
  start with ``test_``, so a misconfigured fixture cannot schedule a
  production agent for termination.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any

import pytest

from . import cleanup_policy as policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-session tracker (single source of truth)
# ---------------------------------------------------------------------------

# Mapping: agent_name → TrackedAgent. Populated by fixtures via
# ``register_test_agent``; consumed by ``pytest_sessionfinish``.
_AGENT_TRACKER: dict[str, policy.TrackedAgent] = {}


def register_test_agent(
    agent_name: str,
    item: Any,
    *,
    created_by_fixture: bool = True,
) -> None:
    """Public wrapper — fixtures call this right after spawning an agent.

    A wrapper instead of re-exporting :func:`cleanup_policy.register_test_agent`
    so the call site is easy to grep (``register_test_agent(...)``) and
    so the tracker dict stays an implementation detail of this module.
    """
    policy.register_test_agent(
        tracker=_AGENT_TRACKER,
        agent_name=agent_name,
        item=item,
        created_by_fixture=created_by_fixture,
    )


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    """Expose the test outcome on the item so fixture finalizers can read it.

    Standard pytest idiom — see
    https://docs.pytest.org/en/stable/example/simple.html
    #making-test-result-information-available-in-fixtures.
    The attribute name is ``rep_<phase>`` where ``<phase>`` is one of
    ``setup``, ``call``, ``teardown``. Fixture teardown looks at
    ``rep_call`` because that's the actual test body.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _wipe_leftover_test_agents_after_each_test(request: pytest.FixtureRequest):
    """Per-test safety net for tests that bypass ``register_test_agent``.

    If a test uses ``with agent_fixture(...)`` without passing
    ``pytest_item=request.node``, or spawns a ``deep-worker`` subprocess
    in a way that doesn't register with the sweeper, this fixture still
    enforces the policy at the end of the test:

      * passed → drop any *new* ``test_*`` registry entry that didn't
                 exist before this test (and SIGTERM its PID).
      * failed → leave everything alone for ``dwcli`` debug.
    """
    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    before: set[str] = {entry.name for entry in registry.list_all()}

    yield

    rep_call = getattr(request.node, "rep_call", None)
    if rep_call is None or not rep_call.passed:
        return

    after_entries = registry.list_all()
    for entry in after_entries:
        name = entry.name
        if not name.startswith("test_"):
            continue
        if name in before:
            continue
        try:
            _terminate(entry.pid)
        except Exception:  # noqa: BLE001
            pass
        try:
            registry.unregister(name)
        except Exception:  # noqa: BLE001
            pass


def _registry_lookup(agent_name: str) -> Any | None:
    """Bind the real ``ProcessRegistry.lookup`` for the sweeper.

    Done lazily inside the helper because importing ProcessRegistry at
    module load time is fine, but kept here to localise the only IO
    dependency of the conftest.
    """
    from deep_worker_cli.process.registry import ProcessRegistry

    return ProcessRegistry().lookup(agent_name)


def _registry_unregister(agent_name: str) -> None:
    from deep_worker_cli.process.registry import ProcessRegistry

    ProcessRegistry().unregister(agent_name)


def _is_alive(pid: int) -> bool:
    """Return ``True`` iff a process with *pid* exists.

    ``os.kill(pid, 0)`` is the canonical Linux probe — it sends no
    signal but raises ``ProcessLookupError`` if the process is gone.
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _terminate(pid: int) -> None:
    """SIGTERM then SIGKILL after 5 s grace period.

    The grace period is bounded — pytest_sessionfinish is the last thing
    that runs in the pytest process, so we cannot afford to hang here.
    Each iteration is a 100 ms sleep; if the process doesn't respect
    SIGTERM in 50 iterations (5 s), escalate.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    for _ in range(50):
        if not _is_alive(pid):
            return
        time.sleep(0.1)

    # Still alive — escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Sweep any test_* agent the fixtures asked us to track.

    The decision per agent is delegated to ``cleanup_policy.decide``;
    failed tests retain their agent so the developer can ``dwcli
    test_<name>`` into it.
    """
    if not _AGENT_TRACKER:
        return

    killed = policy.sweep_leftovers(
        tracker=_AGENT_TRACKER,
        registry_lookup=_registry_lookup,
        registry_unregister=_registry_unregister,
        is_alive=_is_alive,
        terminate=_terminate,
    )

    # Clear the tracker so re-entrant pytest_sessionfinish (rare but
    # possible with pytest-xdist) is a no-op.
    _AGENT_TRACKER.clear()

    if killed:
        logger.info(
            "pytest_sessionfinish swept %d leaked test agent(s): %s",
            len(killed),
            sorted(killed),
        )
