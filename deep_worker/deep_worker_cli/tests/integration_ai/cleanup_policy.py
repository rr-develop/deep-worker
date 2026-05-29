"""Cleanup policy for the integration test suite — agent lifecycle decisions.

The rule is exactly what the user mandated:

    * passed   + created_by_fixture → TERMINATE   (default; keeps host clean)
    * failed   + created_by_fixture → PRESERVE    (debug artefact)
    * pre-existing                  → SKIP        (production agents
                                                   must never be touched)
    * no rep_call                   → PRESERVE    (pytest crashed before
                                                   makereport — we err on
                                                   the side of caution)

This module is **pure** by design — no global state, no real
``ProcessRegistry``, no ``os.kill`` calls. The wiring happens in
``conftest.py`` which injects the live registry + the kill helper. The
unit suite ``tests/integration/test_fixture_cleanup_policy.py`` drives
the policy with fake doubles.

Architectural notes
-------------------
* Event-driven only. Decisions are derived from pytest report objects;
  nothing polls.
* No memory leaks: ``sweep_leftovers`` returns a set of killed agents
  and the tracker is the caller's data — it can be reset / cleared
  between sessions without holding refs anywhere in this module.
* Strict typing: every callable is annotated with the precise shape it
  expects so misuse fails at pyright time, not at the next leaked agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Decision(Enum):
    """What to do with a tracked agent at the end of its test."""

    TERMINATE = auto()
    PRESERVE = auto()
    SKIP = auto()


class _ReportLike(Protocol):
    """Subset of ``_pytest.reports.TestReport`` we care about."""

    passed: bool


class _ItemLike(Protocol):
    """Subset of ``_pytest.nodes.Item`` we care about.

    ``rep_call`` is set by the :func:`pytest_runtest_makereport` hook in
    ``conftest.py``. It is optional because a hard crash (segfault,
    SIGKILL) can preempt the hook entirely.
    """

    nodeid: str


@dataclass(frozen=True)
class TrackedAgent:
    """Bookkeeping entry for an agent the fixture layer asked us to track.

    Attributes
    ----------
    agent_name :
        Registry name of the agent, e.g. ``test_taskmanager_tui``.
    item :
        The pytest item that owns the fixture. Used to read ``rep_call``
        at session-finish time.
    created_by_fixture :
        ``True`` when the fixture spawned the agent (we own it),
        ``False`` when the fixture only attached to a pre-existing
        agent (the user owns it). Pre-existing agents must never be
        terminated by the sweeper.
    """

    agent_name: str
    item: Any  # pytest Item — kept Any to avoid pulling _pytest into runtime
    created_by_fixture: bool


# ---------------------------------------------------------------------------
# Public policy
# ---------------------------------------------------------------------------


def decide(entry: TrackedAgent) -> Decision:
    """Return the cleanup decision for *entry* based on the policy."""
    if not entry.created_by_fixture:
        return Decision.SKIP
    rep_call: _ReportLike | None = getattr(entry.item, "rep_call", None)
    if rep_call is None:
        # We never saw a test outcome — could be a hard crash or a
        # collection-time error. Don't touch the agent.
        return Decision.PRESERVE
    return Decision.TERMINATE if rep_call.passed else Decision.PRESERVE


def register_test_agent(
    *,
    tracker: dict[str, TrackedAgent],
    agent_name: str,
    item: Any,
    created_by_fixture: bool,
) -> None:
    """Record an agent so the session-finish sweeper can decide later.

    Names that do not start with ``test_`` are silently ignored — this
    is a defence-in-depth invariant so that a buggy fixture cannot
    accidentally schedule a production agent for termination.
    """
    if not agent_name.startswith("test_"):
        return
    tracker[agent_name] = TrackedAgent(
        agent_name=agent_name,
        item=item,
        created_by_fixture=created_by_fixture,
    )


# ---------------------------------------------------------------------------
# Sweeper — used by pytest_sessionfinish
# ---------------------------------------------------------------------------


def sweep_leftovers(
    *,
    tracker: dict[str, TrackedAgent],
    registry_lookup: Callable[[str], Any | None],
    registry_unregister: Callable[[str], None],
    is_alive: Callable[[int], bool],
    terminate: Callable[[int], None],
) -> set[str]:
    """Apply the policy to every tracked entry and return killed names.

    The four callables are injected so this function stays hermetic and
    has no dependency on ``ProcessRegistry`` / ``os.kill``. The real
    wiring is in ``conftest.py``.

    Behaviour:
        * ``Decision.TERMINATE`` → call ``terminate(pid)`` if the
          process is still alive, then unregister.
        * ``Decision.PRESERVE``  → leave the agent and the registry
          entry alone (the user wants to ``dwcli`` into it).
        * ``Decision.SKIP``      → no-op (pre-existing agent owned by
          the user).

    Idempotent: calling twice with the same tracker is safe — the
    second call finds no entries in the registry and exits cleanly.
    """
    killed: set[str] = set()
    for name in list(tracker.keys()):
        entry = tracker[name]
        decision = decide(entry)
        if decision is Decision.SKIP:
            continue
        if decision is Decision.PRESERVE:
            continue
        # decision is TERMINATE
        reg_entry = registry_lookup(name)
        if reg_entry is None:
            # Process already cleaned up by a yield-finalizer earlier —
            # nothing to do.
            continue
        pid = reg_entry.pid
        if is_alive(pid):
            terminate(pid)
            killed.add(name)
        # Always remove the registry entry so a re-run with the same
        # ``test_*`` name does not see a stale record.
        registry_unregister(name)
    return killed
