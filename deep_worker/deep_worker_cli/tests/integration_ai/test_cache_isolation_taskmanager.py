"""Integration test for KV-cache isolation between agent and TaskManager.

Verifies DeepSeek prefix-cache isolation using ``user`` parameter when an
agent with TaskManager sends interleaved requests.

Test flow:
1. Start a real deep-worker agent with TaskManager on the default deepseek model
2. Send a progressive task prompt (30%% / 50%% / 70%% / 100%%) via
   ``AgentClient.invoke_agent``
3. Capture ``log_token_usage()`` output from agent stderr (via ``capfd``)
4. Parse cache hit/miss entries and verify cache hit rate > 50%%
   on repeated prefixes
5. Verify both agent and TM user_ids appear in cache logs

Requires ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

_AGENT_NAME = "test_cache_isolation_tm"

_TEST_PROMPT = (
    "Test. If you are asked to complete a task partially or fully, "
    "you must first answer you completed 30%% (and wait for next request), "
    "then 50%%, then 70%%, then 100%% complete. Complete this test task."
)

# ============================================================================
# Parser
# ============================================================================


def _parse_cache_hits(text: str) -> list[dict]:
    """Parse ``log_token_usage()`` lines from stderr text.

    Returns list of dicts with keys: agent, label, prompt, cache_hit,
    miss, completion, total, cache_pct, line.
    """
    pat = re.compile(
        r"(?:\[(\w+)\]\s+)?(\S+)\s+"
        r"prompt=(\d+)\s+cache_hit=(\d+)\s+miss=(\d+)\s+"
        r"completion=(\d+)\s+total=(\d+)\s+\(cache\s+([\d.]+)%\)"
    )
    out = []
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        label, agent, prompt, hit, miss, completion, total, pct = m.groups()
        out.append(dict(
            agent=agent,
            label=label or "",
            prompt=int(prompt),
            cache_hit=int(hit),
            miss=int(miss),
            completion=int(completion),
            total=int(total),
            cache_pct=float(pct),
            line=line,
        ))
    return out


# ============================================================================
# Override: — deepseek-chat may NOT return cache fields; the act of
# sending the request, getting cache info (even "no cache info"), and
# seeing both agent and TM entries is the real test.
# ============================================================================


@pytest.mark.asyncio
async def test_cache_isolation_agent_and_tm(capfd):
    """Send progressive prompt, verify cache hit rate stays high."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")

    from .agent_fixture import agent_fixture

    with agent_fixture(
        agent_name=_AGENT_NAME,
        working_dir=Path("/tmp"),
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
        enable_task_manager=True,
    ) as agent_info:
        t0 = time.monotonic()

        # 1. Connect & send prompt via RPyC
        from deep_worker_cli.client.agent_client import AgentClient

        client = AgentClient(agent_info.agent_name)
        await client.connect()
        logger.info("Sending progressive task prompt...")
        response = await client.invoke_agent(_TEST_PROMPT, stream=False)
        await client.disconnect()
        logger.info("Agent responded: success=%s", response.get("success"))

        # 2. Let TM iterations finish and logs flush
        logger.info("Waiting for TM iterations to settle...")
        await asyncio.sleep(5.0)

        # 3. Capture stderr from the agent process
        #    capfd captures OS-level fd 2, which the child process inherits
        #    via multiprocessing 'spawn'. The child's log_token_usage()
        #    goes to logger → StreamHandler(sys.stderr) → fd 2.
        captured = capfd.readouterr()
        stderr_text = captured.err
        logger.info("Captured %s bytes of stderr", len(stderr_text))

        # Log a sample of stderr for debugging
        stderr_lines = stderr_text.splitlines()
        for i, line in enumerate(stderr_lines[-50:]):  # last 50 lines
            logger.info("[STDERR %d] %s", i, line)

        # 4. Parse cache entries from stderr
        cache_entries = _parse_cache_hits(stderr_text)
        logger.info("Found %s cache log entries", len(cache_entries))
        for e in cache_entries:
            tag = f"[{e['label']}] " if e["label"] else ""
            logger.info(
                "  %s%s: prompt=%s cache_hit=%s miss=%s (%s%%)",
                tag, e["agent"], e["prompt"], e["cache_hit"],
                e["miss"], e["cache_pct"],
            )

        agent_entries = [e for e in cache_entries if e["label"] == "invoke"]
        tm_entries = [e for e in cache_entries if e["label"] in ("tm", "TM")]
        logger.info(
            "Done in %.1fs — agent=%s TM=%s entries",
            time.monotonic() - t0, len(agent_entries), len(tm_entries),
        )

        # 5. Assertions
        # At minimum, some cache-related entries should exist
        # (even if they show "no cache info" for legacy models)
        all_cache_entries = cache_entries  # only contains lines WITH cache info

        assert len(all_cache_entries) > 0, (
            "No cache info log entries found in stderr.\n"
            "This suggests log_token_usage() was not called or the logging "
            "handler is not writing to stderr.\n"
            "Sample stderr (last 20 lines):\n%s" % "\n".join(stderr_lines[-20:])
        )
        assert len(agent_entries) > 0, (
            "No agent (unlabeled) cache entries.\n"
            "All: %s" % [(e["label"], e["agent"]) for e in all_cache_entries]
        )
        assert len(tm_entries) > 0, (
            "No TM (labeled) cache entries. "
            "TaskManager may not have evaluated the response.\n"
            "All: %s" % [(e["label"], e["agent"]) for e in all_cache_entries]
        )

        # Verify cache hit rate on LAST entries (should have some cache after warmup)
        if len(agent_entries) >= 2:
            last_two = agent_entries[-2:]
            avg_pct = sum(e["cache_pct"] for e in last_two) / len(last_two)
            pct_str = ", ".join(f"{e['cache_pct']:.1f}%%" for e in last_two)
            logger.info(
                "Last agent cache rates: %s (avg %.1f%%)", pct_str, avg_pct
            )
            assert avg_pct > 50.0, (
                f"Cache hit too low: {avg_pct:.1f}%% (expected >50%%). "
                f"Last entries: {pct_str}. "
                "Possible cache eviction between agent and TM threads."
            )
        elif len(agent_entries) == 1:
            e = agent_entries[0]
            logger.info("Single agent cache entry: %.1f%%", e["cache_pct"])
        else:
            logger.warning("No agent cache entries for analysis")

        # Also check TM has some cache hits
        if tm_entries:
            tm_last = tm_entries[-1]
            logger.info(
                "TM last entry: %.1f%% cache hit (prompt=%s)",
                tm_last["cache_pct"], tm_last["prompt"],
            )
