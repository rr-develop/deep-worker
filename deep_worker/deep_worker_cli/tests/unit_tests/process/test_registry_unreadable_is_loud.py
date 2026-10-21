"""An unreadable registry must be loud, not an empty registry.

Reported symptom (user on WSL2, ``uv tool install deep-worker``): the web UI
logged ``Discovered 0 agents from registry`` while agents started fine and
nothing in any log explained it. The agent path was unaffected.

Root cause: ``ProcessRegistry._load_registry`` caught
``(json.JSONDecodeError, OSError)`` and returned ``{}`` under the comment
"Corrupted or unreadable registry, start fresh". An ``OSError`` there is
EACCES/EIO — the registry exists and may be full of live agents, only the
read failed. Collapsing that into ``{}`` makes a permission fault
indistinguishable from "nothing is registered", so the web reported zero
agents forever and sent debugging towards inotify and rpyc instead of
``ls -l``.

The asymmetry between the two callers is the tell: the agent path calls
``validate_writable()`` first (``cli/agent_command.py``) and so reports a
real error, while the web path has no such check and went straight into the
silent branch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.exceptions import RegistryReadError
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo

#: ``<repo>/deep_worker/deep_worker_cli`` — the distribution root.
_CLI_ROOT = Path(__file__).resolve().parents[3]

#: ``<repo>`` — holds ``deep_worker/`` and ``deep_worker_web/``.
_REPO_ROOT = _CLI_ROOT.parents[1]

#: Source layout of the web application, a sibling of ``deep_worker/``.
_WEB_SOURCE = _REPO_ROOT / "deep_worker_web"


def _registry_with_one_agent(tmp_path: Path) -> tuple[ProcessRegistry, Path]:
    """A registry file holding exactly one live-looking agent."""
    registry_file = tmp_path / "registry.json"
    registry = ProcessRegistry(
        registry_file=registry_file, lock_file=tmp_path / "registry.lock"
    )
    registry.register(
        AgentProcessInfo(
            name="ghost",
            pid=os.getpid(),  # this process is alive and its cmdline says python
            pid_file=tmp_path / "ghost.pid",
            working_dir=str(tmp_path),
            started_at=datetime.now(),
            rpyc_port=18861,
        )
    )
    return registry, registry_file


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission bits")
def test_unreadable_registry_raises_instead_of_reporting_empty(tmp_path: Path):
    """mode 000 must raise RegistryReadError, not yield {}.

    The positive control in the same test is what gives the negative half
    its meaning: an empty result from a broken read is otherwise
    indistinguishable from an empty registry.
    """
    registry, registry_file = _registry_with_one_agent(tmp_path)

    # Positive control: readable → the agent is really there.
    assert list(registry._load_registry().keys()) == ["ghost"]
    assert [a.name for a in registry.list_all()] == ["ghost"]

    os.chmod(registry_file, 0o000)
    try:
        # The file still exists — this is a read fault, not absence.
        assert registry_file.exists()
        with pytest.raises(RegistryReadError) as excinfo:
            registry._load_registry()
        # The message must point at the actual remedy.
        assert str(registry_file) in str(excinfo.value)
        assert "permission" in str(excinfo.value).lower()

        with pytest.raises(RegistryReadError):
            registry.list_all()
    finally:
        os.chmod(registry_file, 0o666)

    # Recovery is automatic once permissions are restored.
    assert [a.name for a in registry.list_all()] == ["ghost"]


def test_corrupted_registry_still_starts_fresh(tmp_path: Path):
    """Genuine corruption keeps the old tolerant behaviour.

    Guards the other side of the split: the fix must not turn a mangled
    registry into a hard failure, only an unreadable one.
    """
    registry_file = tmp_path / "registry.json"
    registry_file.write_text("{not json at all")
    registry = ProcessRegistry(
        registry_file=registry_file, lock_file=tmp_path / "registry.lock"
    )

    assert registry._load_registry() == {}
    assert registry.list_all() == []


def test_missing_registry_is_still_empty_not_an_error(tmp_path: Path):
    """A registry that was never created is legitimately empty."""
    registry = ProcessRegistry(
        registry_file=tmp_path / "absent.json", lock_file=tmp_path / "absent.lock"
    )
    assert registry._load_registry() == {}
    assert registry.list_all() == []


# --------------------------------------------------------------------------
# The real path the user hit: the web service reading the registry.
# --------------------------------------------------------------------------

_WEB_PROBE = """
import asyncio, json, logging, os, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, {web!r})

from app.services.agent_service import AgentService

reg_file = Path({reg!r})

class Reg:
    '''Stands in for ProcessRegistry at the seam the web actually uses.

    Raises the same error the real registry now raises for an unreadable
    file, so the probe tests the web's HANDLING without needing to run as
    a second uid.
    '''
    registry_file = reg_file

    def list_all(self):
        from deep_worker_cli.exceptions import RegistryReadError
        raise RegistryReadError(
            "Cannot read registry %s: [Errno 13] Permission denied" % reg_file
        )

records = []
class Catch(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))

logging.getLogger("app.services.agent_service").addHandler(Catch())
logging.getLogger("app.services.agent_service").setLevel(logging.DEBUG)

loop = asyncio.get_event_loop_policy().new_event_loop()

# Entry point 1: startup discovery.
svc = AgentService.__new__(AgentService)
svc._registry = Reg()
svc._agents = {{}}
loop.run_until_complete(svc._discover_agents())

discover_errors = [m for (lvl, m) in records if lvl == "ERROR"]
discover_zero = [m for (lvl, m) in records if "Discovered 0 agents" in m]

# Entry point 2: the inotify-driven reconcile. This is the one that runs on
# a LIVE web server, and it must neither drop the known agents nor let the
# error escape and tear down the monitor task.
records.clear()
svc2 = AgentService.__new__(AgentService)
svc2._registry = Reg()
svc2._agents = {{"ghost": object()}}
reconcile_raised = ""
try:
    loop.run_until_complete(svc2._reconcile_from_registry())
except BaseException as exc:
    reconcile_raised = type(exc).__name__ + ": " + str(exc)

reconcile_errors = [m for (lvl, m) in records if lvl == "ERROR"]

print("DISCOVER_ERRORS=" + json.dumps(discover_errors))
print("DISCOVER_ZERO=" + json.dumps(discover_zero))
print("RECONCILE_ERRORS=" + json.dumps(reconcile_errors))
print("RECONCILE_RAISED=" + json.dumps(reconcile_raised))
print("RECONCILE_KEPT=" + json.dumps(sorted(svc2._agents.keys())))
"""


@pytest.mark.skipif(
    not (_WEB_SOURCE / "app" / "services" / "agent_service.py").is_file(),
    reason=f"web source tree not present at {_WEB_SOURCE}",
)
def test_web_logs_an_error_and_never_claims_zero_agents(tmp_path: Path):
    """The web must report the read failure, not "Discovered 0 agents".

    Run out-of-process: importing ``app.services`` pulls the whole web
    application in, which must not leak into this suite's import state.
    """
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(json.dumps({"ghost": {"pid": 1}}))

    probe = _WEB_PROBE.format(web=str(_WEB_SOURCE), reg=str(reg_file))
    # PYTHONPATH is REPLACED, not extended. The inherited value points at
    # whatever tree launched the test runner — on the author's host that is a
    # different checkout, whose deep_worker_cli lacks RegistryReadError. The
    # probe then failed to import and an earlier `pytest.skip` on non-zero
    # exit swallowed it, so this test silently asserted nothing. The probe
    # must run against THIS tree, and an import failure must fail the test.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_CLI_ROOT), str(_REPO_ROOT / "deep_worker" / "deepagents")]
    )

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(probe)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 0, (
        "web application probe did not run; this test proves nothing until it "
        f"does.\nstderr={result.stderr[-800:]!r}"
    )

    def _field(tag: str):
        line = next(
            l for l in result.stdout.splitlines() if l.startswith(tag + "=")
        )
        return json.loads(line[len(tag) + 1 :])

    discover_errors = _field("DISCOVER_ERRORS")
    discover_zero = _field("DISCOVER_ZERO")
    reconcile_errors = _field("RECONCILE_ERRORS")
    reconcile_raised = _field("RECONCILE_RAISED")
    reconcile_kept = _field("RECONCILE_KEPT")

    # Startup discovery: loud, and never the misleading zero count.
    assert discover_errors, (
        "the web swallowed an unreadable registry without logging an error; "
        f"stdout={result.stdout!r}"
    )
    assert any("registry" in m.lower() for m in discover_errors), discover_errors
    assert not discover_zero, (
        "the web reported 'Discovered 0 agents' for a registry it could not "
        f"read: {discover_zero!r}"
    )

    # Reconcile: also loud. Separate assertion because it is a separate
    # call site — a guard on _discover_agents alone leaves the live server
    # (which only ever reconciles) silent.
    assert reconcile_errors, (
        "the inotify-driven reconcile swallowed an unreadable registry "
        f"silently; stdout={result.stdout!r}"
    )
    assert any("registry" in m.lower() for m in reconcile_errors), reconcile_errors

    # It must not propagate: that would kill the monitor task, so the web
    # would never recover even after permissions are fixed.
    assert reconcile_raised == "", (
        f"reconcile let the error escape and tear down the monitor: "
        f"{reconcile_raised}"
    )

    # And it must not conclude the agents died.
    assert reconcile_kept == ["ghost"], (
        f"reconcile dropped known agents on a read failure: {reconcile_kept}"
    )
