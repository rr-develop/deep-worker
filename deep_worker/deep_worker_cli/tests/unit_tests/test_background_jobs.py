"""Tests for the background-job behavior of CLIShellBackend.

New contract (2026-05-21, user request): when a shell command exceeds its
timeout, the backend does NOT kill it. Instead the command keeps running in
the background, the backend returns a ``job_id`` and the available actions,
and the agent decides whether to ``wait_for_job`` (wait longer) or
``kill_job`` (stop it).

These tests use REAL subprocesses (sleep/echo) because the semantics under
test — a process surviving the timeout and being waited on / killed later —
are exactly what mocks would hide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIShellBackend


@pytest.mark.asyncio
async def test_fast_command_returns_normally() -> None:
    """A command that finishes within the timeout behaves as before."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.execute("echo hello", timeout=10)
    assert "hello" in result.output
    assert result.exit_code == 0
    # No leftover job.
    assert backend._jobs == {}


@pytest.mark.asyncio
async def test_timeout_does_not_kill_and_registers_job() -> None:
    """On timeout the process is NOT killed; a job_id is returned and the
    job is registered as still-running."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.execute("sleep 5 && echo done", timeout=1)

    # Still running: no exit code, a job_id, and both action hints.
    assert result.exit_code is None
    assert "job_" in result.output
    assert "wait_for_job" in result.output
    assert "kill_job" in result.output
    assert len(backend._jobs) == 1

    job_id = next(iter(backend._jobs))
    job = backend._jobs[job_id]
    # Process is alive (returncode not yet set).
    assert job.proc.returncode is None

    # Clean up so the test doesn't leak a process.
    await backend.kill_job(job_id)
    assert backend._jobs == {}


@pytest.mark.asyncio
async def test_wait_for_job_returns_output_when_it_finishes() -> None:
    """wait_for_job must return the full output + exit code once the job
    completes."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    # 1s timeout, command takes ~2s, then wait up to 10s.
    started = await backend.execute("sleep 2 && echo finished", timeout=1)
    job_id = next(iter(backend._jobs))

    result = await backend.wait_for_job(job_id, timeout=10)
    assert "finished" in result.output
    assert result.exit_code == 0
    # Job is consumed once finished.
    assert backend._jobs == {}
    assert "job_" in started.output  # sanity: the first call did register it


@pytest.mark.asyncio
async def test_wait_for_job_can_time_out_again_and_keep_running() -> None:
    """If the job is still running after wait_for_job's own timeout, it must
    report still-running with the same options and keep the job alive."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    await backend.execute("sleep 5 && echo done", timeout=1)
    job_id = next(iter(backend._jobs))

    result = await backend.wait_for_job(job_id, timeout=1)
    assert result.exit_code is None
    assert "STILL running" in result.output
    assert "kill_job" in result.output
    assert job_id in backend._jobs  # not consumed

    await backend.kill_job(job_id)  # cleanup


@pytest.mark.asyncio
async def test_kill_job_stops_the_process() -> None:
    """kill_job must terminate the background process and clear the job."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    await backend.execute("sleep 30", timeout=1)
    job_id = next(iter(backend._jobs))

    result = await backend.kill_job(job_id)
    assert f"'{job_id}' killed" in result.output
    assert backend._jobs == {}
    # Negative exit code => killed by signal (SIGTERM=-15 / SIGKILL=-9).
    assert result.exit_code is None or result.exit_code < 0


@pytest.mark.asyncio
async def test_kill_job_kills_child_not_just_shell() -> None:
    """kill_job must kill the WHOLE process group, not just the shell.

    Regression (observed live 2026-05-22): ``sleep 30 && echo`` runs the
    sleep as a child of the shell; signalling only the shell pid left the
    sleep orphaned. The command now runs in its own session/process group
    and kill_job signals the group.
    """
    import asyncio
    import os

    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    # Unique marker so we can find the specific child process.
    marker = "dwtest_killgroup_marker_zzz"
    await backend.execute(f"sleep 300 # {marker}", timeout=1)
    job_id = next(iter(backend._jobs))
    pid = backend._jobs[job_id].proc.pid
    pgid = os.getpgid(pid)

    await backend.kill_job(job_id)

    # The process group must be gone: killpg(pgid, 0) raises ProcessLookupError.
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@pytest.mark.asyncio
async def test_wait_for_unknown_job_is_an_error() -> None:
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.wait_for_job("job_999", timeout=1)
    assert "no background job" in result.output
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_kill_unknown_job_is_an_error() -> None:
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.kill_job("job_999")
    assert "no background job" in result.output
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_job_ids_are_unique_and_incrementing() -> None:
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    await backend.execute("sleep 30", timeout=1)
    await backend.execute("sleep 30", timeout=1)
    ids = sorted(backend._jobs.keys())
    assert len(ids) == 2
    assert ids[0] != ids[1]
    for jid in list(backend._jobs):
        await backend.kill_job(jid)
    assert backend._jobs == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
