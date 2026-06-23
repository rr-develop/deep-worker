"""Reproduce: claude-code agent hangs after a tool_call → text-only final answer.

Live evidence (an observed live hang, 2026-06-14 22:19):

  ...
  [MCP_DIAG] tool_handler_entry tool=execute call_id=toolu_0195QCQrF7rggXrTLHb6xVty
  [ClaudeCodeChatModel.bind_tools] Called with 23 tools
  (no further activity — agent idle forever, session status RUNNING)

py-spy on the running process: main asyncio loop in epoll, sched_count=0,
no pending Future tasks. Both child claude-code CLI subprocesses sit in
``ep_poll`` waiting for our next ``query`` over stdin. Nobody is going to
send it, because we are blocked inside ``ClaudeCodeChatModel.ainvoke`` on
the unconditional ``await proxy_mcp.wait_for_any_pending_call()``
(``claude_code_provider/chat_model.py:1171``).

That ``await`` was wrapped in ``asyncio.wait_for(..., timeout=5.0)`` plus
``except asyncio.TimeoutError: pass`` before a recent commit. The
timeout existed precisely to discriminate between two indistinguishable
SDK-stream states:

  A) Claude wants another tool — a new ``tools/call`` control-request
     will arrive within milliseconds and a handler will register.
  B) Claude finished this turn with a text-only final answer — no
     ``tools/call`` will ever arrive; the only signal is the absence
     of one for "long enough".

Removing the timeout collapses (B) into "wait forever".

Scenario this test reproduces
-----------------------------

A single ``dwcli -p`` headless query that drives the agent through:

  1. one ``HumanMessage`` from the user;
  2. the model asks for one tool call (e.g. read a tiny file);
  3. ``ToolMessage`` is sent back to Claude via ``resolve_call``
     (``ainvoke`` re-entered with ``tool_resolved=True,
     user_message_sent=False``);
  4. Claude responds with a final text-only answer (no further
     ``ToolUseBlock``).

Without the regression, the whole exchange completes in well under 60s.
With the regression, ``ainvoke`` parks on the unconditional
``wait_for_any_pending_call()`` inside step 4 and ``dwcli -p`` never
returns. ``subprocess.run(timeout=120)`` then raises
``TimeoutExpired`` and the test fails with a clear hang signature.

Marked ``integration`` so it stays out of the default unit-test run.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


_AGENT_NAME = "test_cc_hang_after_tool"
# Single-shot prompt: forces ONE tool call (read a tiny file) then a
# text-only final answer. The whole exchange should complete in a few
# seconds on a healthy agent.
_PROMPT = (
    "Read the file /tmp/test_cc_hang_marker.txt with the read_file tool "
    "and tell me what it contains. Answer in exactly one short sentence. "
    "Do not call any other tool afterwards."
)
# How long we let ``dwcli -p`` run. Healthy: ~10-30s. Hung: never returns.
# 120s is comfortably above the healthy ceiling and well below pytest's
# overall ``--timeout`` so we get a clear ``TimeoutExpired`` diagnostic
# rather than a SIGTERM from pytest-timeout that masks the cause.
_HEADLESS_TIMEOUT_SECONDS = 120


@pytest.fixture
def _hang_repro_agent(request: pytest.FixtureRequest):
    """Start a fresh claude-code agent (no TaskManager) for this test."""
    from deep_worker_cli.process.registry import ProcessRegistry

    agent_name = _AGENT_NAME
    registry = ProcessRegistry()

    # Marker file the prompt will ask to read.
    marker = Path("/tmp/test_cc_hang_marker.txt")
    marker.write_text("hello-from-cc-hang-test\n", encoding="utf-8")

    # Kill any leftover from a previous run.
    existing = registry.lookup(agent_name)
    if existing:
        logger.info(f"Killing leftover agent PID {existing.pid}")
        try:
            os.kill(existing.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Let registry notice.
        for _ in range(30):
            if not registry.lookup(agent_name):
                break
            time.sleep(0.1)

    # Clean state dir + logs so the agent starts from a known empty
    # conversation (otherwise prior history could let it short-circuit).
    state_dir = Path.home() / Path("~/.deep_worker").name / agent_name
    if state_dir.exists():
        shutil.rmtree(state_dir)
    log_dir = Path.home() / Path("~/.deep_worker").name / "logs"
    if log_dir.exists():
        for f in log_dir.glob(f"{agent_name}.*"):
            f.unlink(missing_ok=True)

    # Start with claude-code (default model = "sonnet" per
    # ``_PROVIDER_DEFAULT_MODELS`` in ``config.py``). No TaskManager —
    # this scenario reproduces the hang inside one plain ainvoke
    # round-trip, no TM corrective prompts involved.
    logger.info("Starting fresh claude-code agent without TaskManager...")
    proc = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
            "--provider", "claude-code",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for registration.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if registry.lookup(agent_name):
            logger.info("Agent registered.")
            break
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("Agent did not register within 30s")

    yield agent_name

    # Teardown: passed → terminate; failed → preserve for live debug.
    rep_call = getattr(request.node, "rep_call", None)
    if rep_call is not None and not rep_call.passed:
        logger.warning(
            f"Test FAILED → preserving agent '{agent_name}' for py-spy. "
            f"Connect with: dwcli {agent_name}"
        )
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_claude_code_does_not_hang_after_tool_then_text(_hang_repro_agent):
    """Single headless turn: read file → text-only final answer.

    On the post-regression code, ``ainvoke`` inside the second
    re-entry (``tool_resolved=True, user_message_sent=False``) parks on
    ``await proxy_mcp.wait_for_any_pending_call()`` forever, because
    Claude has no further ``tools/call`` to send. ``dwcli -p`` therefore
    never returns and ``subprocess.run`` raises ``TimeoutExpired``.

    Expected on the healthy code path: ``dwcli -p`` returns
    ``returncode == 0`` and stdout contains some recognisable echo of
    the marker file content.
    """
    agent_name = _hang_repro_agent

    logger.info("Running dwcli -p headless turn (will fail with "
                "TimeoutExpired if the hang is back)...")
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["dwcli", "-p", _PROMPT, agent_name],
            capture_output=True,
            text=True,
            timeout=_HEADLESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        pytest.fail(
            f"dwcli -p did not return within {_HEADLESS_TIMEOUT_SECONDS}s "
            f"(elapsed: {elapsed:.1f}s). This is the 2026-06-14 "
            f"regression: ``ainvoke`` is parked on the unconditional "
            f"``wait_for_any_pending_call()`` in chat_model.py:1171, with "
            f"Claude having already given a text-only final answer (no "
            f"further ``tools/call`` will arrive). Inspect the agent with "
            f"py-spy and look for ``ainvoke`` parked on a Future inside "
            f"``ProxyMCPServer._pending_change_waiters``.\n"
            f"Captured stdout so far:\n{exc.stdout!r}\n"
            f"Captured stderr so far:\n{exc.stderr!r}"
        )

    elapsed = time.monotonic() - started
    logger.info(f"dwcli -p returned in {elapsed:.1f}s "
                f"(rc={result.returncode})")
    logger.info(f"stdout:\n{result.stdout}")
    if result.stderr:
        logger.info(f"stderr:\n{result.stderr}")

    assert result.returncode == 0, (
        f"dwcli -p exited non-zero (rc={result.returncode}); "
        f"stderr:\n{result.stderr}"
    )
    # A healthy answer must mention SOMETHING from the file the model was
    # asked to read. We do not pin the exact phrasing — Claude is free to
    # paraphrase — but the marker substring is short and distinctive.
    assert "hello-from-cc-hang-test" in result.stdout or "hello" in result.stdout.lower(), (
        f"dwcli -p returned but its answer does not appear to reflect the "
        f"marker file content. Output was:\n{result.stdout}"
    )
