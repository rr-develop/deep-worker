"""End-to-end test: dev-tree agent with claude-code MUST know its
tool schemas (and the project_memory contract) at turn 1.

What this test does
===================

1. Spawns a real ``deep-worker`` agent in DEV mode:

        deep-worker <name> --dev --provider claude-code \\
            -sp --dir /path/to/dev-tree

   The launcher re-execs into the dev-tree launcher (the --dev path
   under test) and the agent process loads code from the dev tree —
   the exact tree this change set lives in.

2. Connects to the agent via RPyC and sends three questions in turn 1
   of a *fresh* conversation:

   (a) "Are the schemas for your tools fully loaded right now?"
   (b) "Do you have a tool named ``project_memory`` with a known
        parameter schema?"
   (c) "Were you instructed to use ``project_memory`` FIRST when
        searching for project-specific information?"

3. Parses each AIMessage response and looks for a clearly affirmative
   answer (``YES`` token). If any of the three is not affirmative the
   test fails with the verbatim agent reply included in the failure
   message.

Why this is the right oracle
----------------------------

A direct memory inspection of Claude SDK internal state would couple
the test to private structures of the Anthropic CLI. Instead we ask
the agent itself — that is *exactly* the user-visible property we
care about ("agent knows its tools at turn 1, before any tool was
called"). If tool_search defers tool schemas, the agent will tell us
"I see only tool names, not schemas" or it will simply not know the
shape of ``project_memory``.

The agent's instructions about ``project_memory`` come from a
LangChain middleware that injects a system-prompt section listing
``project_memory`` as the FIRST tool to use for project lookups.
That section MUST be in turn-1 context regardless of tool_search.

Dependencies
============

* ``deep-worker`` launcher must be in $PATH (this is the
  launcher from this repository tree; --dev re-execs it).
* the dev tree must exist (so --dev dispatch succeeds).
* ``claude`` CLI must be installed and authenticated locally
  (provider claude-code). The test skips otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths to the launcher under test and the AgentClient that talks to it.
# Both are __file__-relative so this test file works the same way in
# every tree it is checked out in.
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# tests/integration_ai/<file> -> deep_worker_cli/
_DEEP_WORKER_CLI = _THIS_FILE.parent.parent.parent
# deep_worker_cli/ -> deep_worker/
_DEEP_WORKER = _DEEP_WORKER_CLI.parent
# deep_worker/ -> <TREE>/
_TREE_ROOT = _DEEP_WORKER.parent

LAUNCHER = _DEEP_WORKER_CLI / "deep-worker"
DEV_TREE = Path("/path/to/dev-tree")
DEV_TREE_AGENT_DIR = DEV_TREE  # working dir = the dev tree itself


# ---------------------------------------------------------------------------
# Pre-conditions: skip cleanly when the environment can't run the test.
# ---------------------------------------------------------------------------


def _skip_if_prereqs_missing() -> None:
    if not LAUNCHER.exists():
        pytest.skip(f"launcher not found: {LAUNCHER}")
    if not DEV_TREE.exists():
        pytest.skip(f"dev tree absent: {DEV_TREE}")
    if shutil.which("claude") is None:
        pytest.skip("`claude` CLI not in PATH; claude-code provider unusable")


# ---------------------------------------------------------------------------
# Fixture: spawn a real dev-tree agent with claude-code, clean up after.
# ---------------------------------------------------------------------------


@pytest.fixture
def dev_agent_with_claude_code(request: pytest.FixtureRequest):
    """Yields a tuple ``(agent_name, agent_info)`` for a freshly-spawned
    dev-tree agent.

    Cleanup policy: passed → terminate + clean, failed → preserve for
    debug. The session-finish sweeper is the safety net.
    """
    from .conftest import register_test_agent

    _skip_if_prereqs_missing()

    # Use a unique name per-run so parallel CI doesn't collide and so
    # there's no stale checkpoint history hiding the issue.
    name = f"test_itest_devschemas_{os.getpid()}_{uuid.uuid4().hex[:6]}"

    # Make sure no stale state of the same name exists. The fixture
    # owns the name end-to-end.
    _kill_and_clean(name)

    cmd = [
        str(LAUNCHER),
        name,
        "--dev",
        "--provider", "claude-code",
        "--model", "sonnet",  # cheapest first-party model that supports tool_search
        "--skip-permissions",
        "--dir", str(DEV_TREE_AGENT_DIR),
        "--clean",
    ]
    logger.info("Spawning agent: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # The launcher daemonizes; do NOT start_new_session=True or we
        # lose the ability to read stdout/stderr cleanly. The default
        # is fine — the launcher's double-fork already detaches it.
    )

    # The deep-worker launcher daemonizes very quickly (< 5s) and
    # prints "starting in background ...". Wait for registry entry.
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    deadline = time.monotonic() + 60.0
    info = None
    while time.monotonic() < deadline:
        info = registry.lookup(name)
        if info:
            break
        # If launcher itself exited non-zero, surface that fast.
        if proc.poll() is not None and proc.returncode != 0:
            out, err = proc.communicate(timeout=2)
            pytest.fail(
                f"deep-worker launcher exited {proc.returncode} before "
                f"agent registered.\nstdout: {out!r}\nstderr: {err!r}"
            )
        time.sleep(0.5)

    if info is None:
        _kill_and_clean(name)
        pytest.fail(f"agent '{name}' did not register within 60s")

    logger.info("Agent registered: pid=%s rpyc_port=%s",
                info.pid, info.rpyc_port)

    # Also assert the agent really is running on the DEV tree —
    # otherwise a misconfiguration could silently send the test
    # through prod and we'd be testing the wrong code.
    assert str(DEV_TREE_AGENT_DIR) in str(info.working_dir), (
        f"agent working_dir {info.working_dir!r} does not look like "
        f"the dev tree {DEV_TREE_AGENT_DIR!r}; --dev dispatch broke"
    )

    register_test_agent(name, request.node, created_by_fixture=True)

    yield name, info

    # Teardown: passed → terminate, failed → preserve.
    rep_call = getattr(request.node, "rep_call", None)
    if rep_call is not None and not rep_call.passed:
        logger.warning(
            "Test FAILED → preserving agent '%s' for debugging. "
            "Connect with: dwcli %s", name, name,
        )
        return

    logger.info("Test passed → terminating agent '%s'", name)
    _kill_and_clean(name)


def _kill_and_clean(name: str) -> None:
    """Best-effort: kill the named agent and remove its on-disk state.

    Idempotent. Used as both setup safety net and teardown.
    """
    pid_file = Path(f"/tmp/deep_worker_{name}.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            # Wait up to 5s, then SIGKILL.
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
        try:
            pid_file.unlink()
        except OSError:
            pass

    data_dir = Path("~/.deep_worker").expanduser() / name
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)

    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for f in log_dir.glob(f"{name}.*"):
            try:
                f.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helper: ask the agent one question, return its reply text.
# ---------------------------------------------------------------------------


async def _ask(agent_name: str, message: str, timeout: float = 180.0) -> str:
    """Send a single message to the agent and return its assistant
    reply as a string. Raises if the call fails or times out."""
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient(agent_name=agent_name)
    await client.connect()
    try:
        response = await asyncio.wait_for(
            client.invoke_agent(message=message, stream=False),
            timeout=timeout,
        )
    finally:
        await client.disconnect()

    assert response.get("success") is True, (
        f"invoke_agent failed: {response!r}"
    )
    reply = response.get("response", {}).get("content", "")
    assert isinstance(reply, str) and reply.strip(), (
        f"empty reply from agent: {response!r}"
    )
    return reply


# Match a YES/NO at the start of the reply, tolerant to bolding,
# trailing punctuation, and case. The test prompt instructs the agent
# to begin its reply with YES or NO.
_YES_RE = re.compile(r"^\s*[*_]*\s*YES\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*[*_]*\s*NO\b", re.IGNORECASE)


def _is_affirmative(reply: str) -> bool:
    if _YES_RE.match(reply):
        return True
    if _NO_RE.match(reply):
        return False
    # Fall back to a loose contains-check on the first line. Some
    # models prefix with a sentence before YES/NO.
    first_line = reply.splitlines()[0] if reply.splitlines() else reply
    if re.search(r"\bYES\b", first_line) and not re.search(r"\bNO\b", first_line):
        return True
    if re.search(r"\bNO\b", first_line) and not re.search(r"\bYES\b", first_line):
        return False
    # Ambiguous — treat as failure (test wants an explicit answer).
    return False


# ---------------------------------------------------------------------------
# The actual test.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_agent_has_tool_schemas_and_project_memory_at_turn_1(
    dev_agent_with_claude_code,
) -> None:
    """At turn 1 of a freshly-spawned ``deep-worker ... --dev`` agent
    with the claude-code provider:

    * the full schemas of its tools must be loaded (NOT deferred via
      Claude Code's tool_search beta);
    * the schema of ``project_memory`` must be specifically known;
    * the agent must have been instructed to use ``project_memory``
      FIRST when looking up project-specific information.

    Any failure here means the tool_search defer-loading regression
    is back and ``BUILT_IN_TOOLS`` no longer contains ``"ToolSearch"``,
    or that the project_memory middleware lost its system-prompt
    injection.
    """
    agent_name, _ = dev_agent_with_claude_code

    q1 = (
        "Please answer in this exact shape: start your reply with the "
        "single uppercase word YES or NO on the first line, then on the "
        "next lines give a short justification. "
        "Question: Right now, at the start of this conversation, do you "
        "already have the FULL JSON input schemas (parameters, their "
        "types, descriptions) of every tool you are allowed to use — "
        "without first having to query any meta-tool like ToolSearch to "
        "fetch them? YES means you can see argument lists and types "
        "right now; NO means you only see tool names with their "
        "schemas hidden behind a separate lookup step."
    )
    reply1 = await _ask(agent_name, q1)
    logger.info("Q1 reply: %s", reply1)
    assert _is_affirmative(reply1), (
        "Agent does NOT have full tool schemas at turn 1 — "
        "tool_search deferred them. Reply was:\n" + reply1
    )

    q2 = (
        "Same answer format as before: first line is YES or NO. "
        "Question: Among the tools currently available to you, is "
        "there one named exactly ``project_memory``, and do you have "
        "its full input schema (its parameters with their types) "
        "available to you right now, without needing to fetch it?"
    )
    reply2 = await _ask(agent_name, q2)
    logger.info("Q2 reply: %s", reply2)
    assert _is_affirmative(reply2), (
        "Agent does NOT see project_memory with a full schema at "
        "turn 1. Reply was:\n" + reply2
    )

    q3 = (
        "Same answer format as before: first line is YES or NO. "
        "Question: According to the instructions you received at the "
        "start of this conversation, when you need to find information "
        "that is specific to THIS project (its terminology, where a "
        "feature is implemented, its architecture), are you supposed to "
        "use ``project_memory`` FIRST, before resorting to other tools "
        "like grep, read_file, web_search, or task delegation?"
    )
    reply3 = await _ask(agent_name, q3)
    logger.info("Q3 reply: %s", reply3)
    assert _is_affirmative(reply3), (
        "Agent was NOT instructed to use project_memory first. "
        "Reply was:\n" + reply3
    )
