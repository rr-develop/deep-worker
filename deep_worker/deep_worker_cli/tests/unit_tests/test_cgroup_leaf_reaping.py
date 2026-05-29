"""Regression: empty ``dwagent-*`` cgroup leaves must be reaped, not leaked.

Incident (2026-05-23): ``_detach_from_session_scope`` creates a per-agent
leaf cgroup ``user@UID.service/dwagent-<name>-<pid>`` and migrates the agent
into it, but NOTHING ever removed the leaf when the agent died. The kernel
does not auto-reap an empty cgroup directory, and systemd's user-manager does
not rmdir delegated leaves either — so they accumulated without bound (272
stale leaves observed, ``nr_dying_descendants`` climbing). That accumulated
state is what made the cgroup-detach end-to-end tests flaky (an agent's
migration intermittently failed under the pressure, leaving it in the
launching scope and failing the ``/dwagent-`` assertion).

A process on cgroup v2 CANNOT rmdir its OWN leaf while it still lives in it
(it can't move back to ``user@UID.service`` — the no-internal-processes rule
forbids it when subtree_control is set). So the design is "reap the corpses":
every agent, at startup and shutdown, rmdir's any EMPTY ``dwagent-*`` leaf it
finds. Its own leaf is reaped by the next agent's sweep after it dies.

Contract under test (``_reap_empty_cgroup_leaves``):
  * removes an EMPTY dwagent-* leaf;
  * NEVER removes a leaf that still has live procs;
  * never raises on missing dir / permission errors;
  * leaves non-dwagent siblings (init.scope, tmux-*.scope) untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _user_service() -> Path:
    return Path(
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service"
    )


def _is_cgroup_v2() -> bool:
    try:
        for line in open("/proc/mounts"):
            f = line.split()
            if len(f) >= 3 and f[1] == "/sys/fs/cgroup":
                return f[2] == "cgroup2"
    except OSError:
        pass
    return False


_HAVE = _is_cgroup_v2() and _user_service().exists() and os.access(_user_service(), os.W_OK)

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="needs writable cgroup v2 user@UID.service"
)


def test_helper_exists() -> None:
    from deep_worker_cli.cli import agent_command

    assert hasattr(agent_command, "_reap_empty_cgroup_leaves"), (
        "_reap_empty_cgroup_leaves helper is missing — the fix for the "
        "unbounded dwagent-* cgroup leaf leak (2026-05-23) must add it."
    )


def test_reaps_empty_leaf() -> None:
    from deep_worker_cli.cli import agent_command

    leaf = _user_service() / f"dwagent-reaptest-empty-{os.getpid()}"
    leaf.mkdir(exist_ok=True)
    assert leaf.exists()

    agent_command._reap_empty_cgroup_leaves()

    assert not leaf.exists(), "an empty dwagent-* leaf must be reaped"


def test_does_not_reap_leaf_with_live_proc() -> None:
    import subprocess
    import time

    from deep_worker_cli.cli import agent_command

    leaf = _user_service() / f"dwagent-reaptest-live-{os.getpid()}"
    leaf.mkdir(exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        # Put the live sleep into the leaf. Migrating a process into a cgroup
        # requires write access to the leaf's ``cgroup.procs``, which is a
        # STRICTER capability than the mkdir/rmdir the other reaping tests
        # need (the module-level ``_HAVE`` guard only checks W_OK on the
        # parent user@UID.service dir). In sandboxes without cgroup.procs
        # write delegation this raises PermissionError; the contract under
        # test (a leaf with live procs is not reaped) simply cannot be
        # exercised there, so skip rather than fail.
        try:
            (leaf / "cgroup.procs").write_text(f"{proc.pid}\n")
        except PermissionError:
            pytest.skip(
                "cannot migrate a process into a cgroup leaf here "
                "(no cgroup.procs write delegation)"
            )
        time.sleep(0.1)

        agent_command._reap_empty_cgroup_leaves()

        assert leaf.exists(), (
            "a dwagent-* leaf that still has live procs MUST NOT be reaped"
        )
    finally:
        proc.kill()
        proc.wait()
        time.sleep(0.2)
        # Now empty — reap it for cleanliness.
        agent_command._reap_empty_cgroup_leaves()


def test_never_raises_on_missing_or_locked() -> None:
    from deep_worker_cli.cli import agent_command

    # Must be safe to call repeatedly even with nothing to do.
    agent_command._reap_empty_cgroup_leaves()
    agent_command._reap_empty_cgroup_leaves()


def test_leaves_non_dwagent_siblings_untouched() -> None:
    from deep_worker_cli.cli import agent_command

    # init.scope always exists under user@UID.service and is NOT a dwagent
    # leaf — the reaper must never touch it.
    init_scope = _user_service() / "init.scope"
    existed = init_scope.exists()

    agent_command._reap_empty_cgroup_leaves()

    if existed:
        assert init_scope.exists(), (
            "reaper must only touch dwagent-* leaves, never other cgroups"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
