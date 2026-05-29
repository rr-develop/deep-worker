"""Hermetic tests for the test-agent cleanup policy.

These tests do NOT spawn real deep-worker agents. They drive the
``register_test_agent`` / ``pytest_sessionfinish`` machinery in
``tests/integration_ai/conftest.py`` against a fake registry and a fake
kill-function, exercising the three policy branches:

    1. passed  + created_by_fixture → kill          ← user-mandated default
    2. failed  + created_by_fixture → preserve      ← user-mandated debug path
    3. pre-existing                 → never touch   ← safety invariant

By keeping the tests hermetic we get fast feedback (< 1 s) and we can run
them under the regular ``pytest tests/integration -m integration`` suite
without dragging in any AI providers or RPyC ports.
"""

from __future__ import annotations

import pytest

# The module under test lives in the sibling integration_ai package.
# Importing it here drags in ``deep_worker_cli.process.registry`` which
# is already available from the existing fixtures, so no extra setup.
from tests.integration_ai import cleanup_policy as policy

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubReport:
    """Stand-in for ``_pytest.reports.TestReport``."""

    def __init__(self, *, passed: bool) -> None:
        self.passed = passed
        self.when = "call"
        self.outcome = "passed" if passed else "failed"


class _StubItem:
    """Stand-in for ``_pytest.nodes.Item``.

    Only the attributes the cleanup hook touches are modelled —
    ``nodeid`` for diagnostics and ``rep_call`` set by
    :func:`pytest_runtest_makereport`.
    """

    def __init__(self, nodeid: str, *, rep_call: _StubReport | None = None) -> None:
        self.nodeid = nodeid
        if rep_call is not None:
            self.rep_call = rep_call


class _FakeRegistry:
    """Minimal in-memory ProcessRegistry stand-in."""

    def __init__(self, alive_pids: set[int]) -> None:
        # alive_pids stays under our control so the policy code can ask
        # "is this PID alive?" deterministically.
        self._entries: dict[str, int] = {}  # agent_name → pid
        self._alive_pids: set[int] = set(alive_pids)

    def add(self, name: str, pid: int) -> None:
        self._entries[name] = pid

    def lookup(self, name: str) -> object | None:  # mimics AgentProcessInfo
        pid = self._entries.get(name)
        if pid is None:
            return None

        class _Info:
            def __init__(self, pid: int) -> None:
                self.pid = pid
                self.rpyc_port = 0
                self.working_dir = "/tmp"

        return _Info(pid)

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def is_alive(self, pid: int) -> bool:
        return pid in self._alive_pids

    def kill(self, pid: int) -> None:
        self._alive_pids.discard(pid)

    # convenience for assertions
    def all_names(self) -> set[str]:
        return set(self._entries.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    """Pure policy: given a tracked entry, return TERMINATE / PRESERVE / SKIP."""

    def test_passed_test_with_fixture_agent_returns_terminate(self) -> None:
        item = _StubItem("test_x", rep_call=_StubReport(passed=True))
        entry = policy.TrackedAgent(agent_name="test_a", item=item, created_by_fixture=True)
        assert policy.decide(entry) is policy.Decision.TERMINATE

    def test_failed_test_with_fixture_agent_returns_preserve(self) -> None:
        item = _StubItem("test_x", rep_call=_StubReport(passed=False))
        entry = policy.TrackedAgent(agent_name="test_a", item=item, created_by_fixture=True)
        assert policy.decide(entry) is policy.Decision.PRESERVE

    def test_pre_existing_agent_always_returns_skip(self) -> None:
        # Pass or fail — pre-existing must never be touched.
        for passed in (True, False):
            item = _StubItem("test_x", rep_call=_StubReport(passed=passed))
            entry = policy.TrackedAgent(
                agent_name="test_a", item=item, created_by_fixture=False
            )
            assert policy.decide(entry) is policy.Decision.SKIP

    def test_no_rep_call_means_preserve(self) -> None:
        # pytest crashed before makereport — we err on the side of preserving.
        item = _StubItem("test_x", rep_call=None)
        entry = policy.TrackedAgent(agent_name="test_a", item=item, created_by_fixture=True)
        assert policy.decide(entry) is policy.Decision.PRESERVE


class TestSweepLeftovers:
    """``sweep_leftovers`` walks the tracker dict and applies the policy."""

    def test_sweep_kills_only_passed_fixture_agents(self) -> None:
        tracker: dict[str, policy.TrackedAgent] = {
            "test_passed": policy.TrackedAgent(
                agent_name="test_passed",
                item=_StubItem("test_p", rep_call=_StubReport(passed=True)),
                created_by_fixture=True,
            ),
            "test_failed": policy.TrackedAgent(
                agent_name="test_failed",
                item=_StubItem("test_f", rep_call=_StubReport(passed=False)),
                created_by_fixture=True,
            ),
            "test_preexisting": policy.TrackedAgent(
                agent_name="test_preexisting",
                item=_StubItem("test_e", rep_call=_StubReport(passed=True)),
                created_by_fixture=False,
            ),
        }
        registry = _FakeRegistry(alive_pids={101, 102, 103})
        registry.add("test_passed", 101)
        registry.add("test_failed", 102)
        registry.add("test_preexisting", 103)

        killed = policy.sweep_leftovers(
            tracker=tracker,
            registry_lookup=registry.lookup,
            registry_unregister=registry.unregister,
            is_alive=registry.is_alive,
            terminate=registry.kill,
        )

        assert killed == {"test_passed"}
        # The passed agent must be dead and unregistered.
        assert 101 not in registry._alive_pids
        assert "test_passed" not in registry.all_names()
        # The failed agent must be untouched (preserved for debug).
        assert 102 in registry._alive_pids
        assert "test_failed" in registry.all_names()
        # Pre-existing must be untouched.
        assert 103 in registry._alive_pids
        assert "test_preexisting" in registry.all_names()

    def test_sweep_skips_already_dead_agents_gracefully(self) -> None:
        # Agent recorded as passed, but its process already died (e.g. the
        # yield-based teardown already killed it). Sweep must not error.
        tracker: dict[str, policy.TrackedAgent] = {
            "test_passed": policy.TrackedAgent(
                agent_name="test_passed",
                item=_StubItem("test_p", rep_call=_StubReport(passed=True)),
                created_by_fixture=True,
            ),
        }
        registry = _FakeRegistry(alive_pids=set())  # nothing alive
        registry.add("test_passed", 101)

        killed = policy.sweep_leftovers(
            tracker=tracker,
            registry_lookup=registry.lookup,
            registry_unregister=registry.unregister,
            is_alive=registry.is_alive,
            terminate=registry.kill,
        )

        # Nothing actually killed (already dead), but the registry entry
        # must be cleaned up so the sweep is idempotent.
        assert killed == set()
        assert "test_passed" not in registry.all_names()

    def test_sweep_is_idempotent(self) -> None:
        tracker: dict[str, policy.TrackedAgent] = {
            "test_passed": policy.TrackedAgent(
                agent_name="test_passed",
                item=_StubItem("test_p", rep_call=_StubReport(passed=True)),
                created_by_fixture=True,
            ),
        }
        registry = _FakeRegistry(alive_pids={101})
        registry.add("test_passed", 101)

        first = policy.sweep_leftovers(
            tracker=tracker,
            registry_lookup=registry.lookup,
            registry_unregister=registry.unregister,
            is_alive=registry.is_alive,
            terminate=registry.kill,
        )
        # Calling a second time: registry is already empty, second call is a no-op.
        second = policy.sweep_leftovers(
            tracker=tracker,
            registry_lookup=registry.lookup,
            registry_unregister=registry.unregister,
            is_alive=registry.is_alive,
            terminate=registry.kill,
        )

        assert first == {"test_passed"}
        assert second == set()


class TestRegisterTestAgent:
    """``register_test_agent`` refuses to track non-test_ names."""

    def test_register_rejects_non_test_prefix(self) -> None:
        tracker: dict[str, policy.TrackedAgent] = {}
        # Production agent name (no test_ prefix) — must be silently ignored
        # so a misconfigured test cannot accidentally schedule a kill on
        # a production agent.
        policy.register_test_agent(
            tracker=tracker,
            agent_name="prod_agent",
            item=_StubItem("test_x"),
            created_by_fixture=True,
        )
        assert tracker == {}

    def test_register_accepts_test_prefix(self) -> None:
        tracker: dict[str, policy.TrackedAgent] = {}
        item = _StubItem("test_x")
        policy.register_test_agent(
            tracker=tracker,
            agent_name="test_alpha",
            item=item,
            created_by_fixture=True,
        )
        assert "test_alpha" in tracker
        assert tracker["test_alpha"].created_by_fixture is True
