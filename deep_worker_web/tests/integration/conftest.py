"""Per-suite conftest for ``<web-app-dir>/tests/integration``.

Mirrors the cleanup policy in ``deep_worker_cli/tests/integration_ai``::

    passed_test     + created_by_fixture  →  terminate the agent
    failed_test     + created_by_fixture  →  preserve for `dwcli` debug
    pre-existing                          →  never touch

We import the policy module from the cli package so there is exactly
one source of truth (``deep_worker_cli/tests/integration_ai/cleanup_policy.py``),
covered by hermetic tests under
``deep_worker_cli/tests/integration/test_fixture_cleanup_policy.py``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest


# Reuse the cleanup_policy module that lives next to the cli tests so
# there is exactly one source of truth for the cleanup decision tree.
# We load it by file path (not via ``tests.integration_ai``) because the
# string ``tests`` would resolve to the *web* package when running this
# conftest from the web app tree.
import importlib.util as _importlib_util

_POLICY_PATH = Path(
    "/path/to/dev-tree/deep_worker/deep_worker_cli/"
    "tests/integration_ai/cleanup_policy.py"
)
_spec = _importlib_util.spec_from_file_location(
    "_dw_cleanup_policy",
    _POLICY_PATH,
)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load cleanup_policy from {_POLICY_PATH}")
policy = _importlib_util.module_from_spec(_spec)
sys.modules["_dw_cleanup_policy"] = policy
_spec.loader.exec_module(policy)

logger = logging.getLogger(__name__)


_AGENT_TRACKER: dict[str, policy.TrackedAgent] = {}


def register_test_agent(
    agent_name: str,
    item: Any,
    *,
    created_by_fixture: bool = True,
) -> None:
    """Thin wrapper around :func:`cleanup_policy.register_test_agent`."""
    policy.register_test_agent(
        tracker=_AGENT_TRACKER,
        agent_name=agent_name,
        item=item,
        created_by_fixture=created_by_fixture,
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _wipe_leftover_test_agents_after_each_test(request: pytest.FixtureRequest):
    """Per-test cleanup: enforce ``passed → kill`` on the registry side.

    Tests in this suite use ``with agent_fixture() as agent_info:`` and do
    NOT pass a ``pytest_item=request.node`` argument. The fixture itself
    correctly terminates the agent *process* in its ``finally`` clause,
    but the ProcessRegistry record can outlive that on edge cases (race
    between mp.Process.terminate and registry.unregister). This autouse
    fixture compares the registry snapshot before vs. after the test and,
    if the test passed, sweeps any new ``test_*`` entry that did not
    exist before this test ran.
    """
    from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]

    registry = ProcessRegistry()
    before: set[str] = {entry.name for entry in registry.list_all()}

    yield

    rep_call = getattr(request.node, "rep_call", None)
    if rep_call is None or not rep_call.passed:
        # Failed (or never ran call phase) — preserve everything.
        return

    after_entries = registry.list_all()
    for entry in after_entries:
        name = entry.name
        if not name.startswith("test_"):
            continue  # never touch production
        if name in before:
            continue  # was already there → not ours to kill
        # New ``test_*`` agent introduced by this test → terminate it
        # so the policy holds even when the user-facing context manager
        # forgot to pass ``pytest_item``.
        try:
            _terminate(entry.pid)
        except Exception:  # noqa: BLE001
            pass
        try:
            registry.unregister(name)
        except Exception:  # noqa: BLE001
            pass


def _registry_lookup(agent_name: str) -> Any | None:
    from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]

    return ProcessRegistry().lookup(agent_name)


def _registry_unregister(agent_name: str) -> None:
    from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]

    ProcessRegistry().unregister(agent_name)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    for _ in range(50):
        if not _is_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _AGENT_TRACKER:
        return
    killed = policy.sweep_leftovers(
        tracker=_AGENT_TRACKER,
        registry_lookup=_registry_lookup,
        registry_unregister=_registry_unregister,
        is_alive=_is_alive,
        terminate=_terminate,
    )
    _AGENT_TRACKER.clear()
    if killed:
        logger.info(
            "pytest_sessionfinish swept %d leaked test agent(s): %s",
            len(killed),
            sorted(killed),
        )
