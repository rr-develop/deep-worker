"""Shared fixtures for deep_worker_cli integration tests.

Purpose: kill stale ProcessRegistry entries / orphan agents LEFT BEHIND by
previous runs (pytest aborted, test crashed, OS OOM-killed the agent) so
tests that use fixed agent names (e.g. ``test-asyncio-agent`` in
test_asyncio_server_e2e.py) never see a spurious ``AgentAlreadyExistsError``
on startup.

The cleanup walks /tmp/deep_worker_registry.json and drops any entry whose
``pid`` is no longer a live process OR whose name matches one of the
well-known integration-test agent names.

Also arms ``faulthandler.dump_traceback_later`` so a hanging test prints
its stack instead of getting silently SIGKILL'd by ``timeout(1)``.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

# Dump every thread's stack after 60 s with no progress. Repeats so we
# see the stack every 60 s of hang, not just the first one.
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)

_REGISTRY_FILE = Path("/tmp/deep_worker_registry.json")
_PID_FILE_PATTERN = "/tmp/deep_worker_{name}.pid"

# Agent names used by in-tree integration tests. Any stale entry with one of
# these names is assumed to be an orphan from a previous run and evicted.
_TEST_AGENT_NAMES: frozenset[str] = frozenset({
    "test-asyncio-agent",
    "test-asyncio-agent-2",
    "test-asyncio-agent-3",
    "test-asyncio-agent-4",
    "test-asyncio-migration-agent",
    "test-asyncio-migration-agent-2",
    "test-agent-shutdown",
    "debug-hang-test",
})


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _cleanup_stale_registry() -> None:
    """Remove dead / test-named entries from the shared registry."""
    if not _REGISTRY_FILE.exists():
        return
    try:
        with _REGISTRY_FILE.open("r") as f:
            reg: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    changed = False
    for name in list(reg.keys()):
        entry = reg[name]
        pid = int(entry.get("pid", 0))
        is_stale_by_death = pid > 0 and not _is_alive(pid)
        is_test_name = name in _TEST_AGENT_NAMES or name.startswith("test-asyncio-")
        if is_stale_by_death or is_test_name:
            # Best-effort kill if the PID IS alive but we know it is a
            # leftover test agent — they must never survive across test
            # sessions.
            if pid > 0 and is_test_name and _is_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            # Drop the entry.
            reg.pop(name, None)
            changed = True
            # Remove the pid file too.
            pid_file = Path(_PID_FILE_PATTERN.format(name=name))
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    if changed:
        try:
            with _REGISTRY_FILE.open("w") as f:
                json.dump(reg, f, indent=2)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _clean_registry_around_each_test() -> Any:
    """Run registry cleanup before AND after every integration test.

    Before: give the test a clean slate.
    After:  evict any entry the test itself leaked (crash before
            ``agent.shutdown()``).
    """
    _cleanup_stale_registry()
    yield
    _cleanup_stale_registry()
