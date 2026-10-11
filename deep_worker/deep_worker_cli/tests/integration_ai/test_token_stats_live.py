"""Live token-stats integration tests (token-stats design §8, §11.4).

These verify FOR REAL (no mocks) the end-to-end statistics-collection path:

  1. Standard provider (deepseek): a real agent → ``token_stats.jsonl``
     in ``~/.deep_worker/<name>/`` with records kind="agent", status="ok",
     and the invariant ``hit + miss = prompt`` where cache fields are present.
  2. TaskManager (``-tm``): kind="tm" records appear in the SAME file —
     several buckets in one file.
  3. codex: ``TurnCompletedEvent.usage`` → record provider="codex".
  4. grok: per-turn usage from ``session/prompt`` RESULT ``_meta`` → record
     provider="grok" (tokens, cache hit = cachedReadTokens; live-red-discovery
     0.2.73 — the ACP ``usage_update`` does not actually arrive).
  5. failover: the agent first runs on claude-code (record in file), then
     the same agent (same data_dir) is restarted on deepseek — the effect
     of a failover restart → in ONE file records from ≥2 different
     providers. (The auto-failover mechanism itself — a real usage-limit →
     ``_schedule_failover`` → spawn — is covered by
     ``test_grok_nested_402_failover_to_deepseek.py``; here we verify the
     token-statistics invariant: the recorder is bound to the agent's
     data_dir, not to the provider.)

Each test is gated on provider availability (CLI / API key) and is deselected
by default (``-m 'not integration'``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.stats.token_stats import (
    STATUS_OK,
    read_stats_file,
)
from deep_worker_cli.storage_paths import agent_root

from .agent_fixture import agent_fixture as _make_agent
from .provider_home import provider_login  # noqa: F401 — pytest fixture

pytestmark = pytest.mark.integration

_DEEPSEEK_MODEL = "deepseek-chat"


def _stats_path(agent_name: str) -> Path:
    """Resolved the way production resolves it — never hand-assembled.

    The old ``~/.deep_worker/<name>/`` layout moved to
    ``~/.deep_worker/agents/<name>/`` (see the agent-DB storage layout
    design, §3) and this helper was left behind, so every test here read an
    empty directory and reported "the provider wrote nothing".
    """
    return agent_root(agent_name, create=False) / "token_stats.jsonl"


def _read_records(agent_name: str) -> list[dict]:
    read = read_stats_file(_stats_path(agent_name))
    return [r.to_dict() for r in read.records]


async def _wait_until_idle(client: AgentClient, deadline: float) -> str:
    last = ""
    while time.monotonic() < deadline:
        st = (await client._send_command({"command": "status"})) or {}
        last = str(st.get("status", "")).lower()
        if last in ("idle", "ready"):
            return last
        await asyncio.sleep(0.5)
    return last


async def _invoke_and_wait_idle(
    client: AgentClient, prompt: str, *, idle_after: float = 60.0
) -> str:
    await client.invoke_agent(prompt, stream=False)
    return await _wait_until_idle(client, time.monotonic() + idle_after)


def _deepseek_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


@pytest.fixture
def deepseek_agent():
    key = _deepseek_key()
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")
    with _make_agent(
        agent_name="test_stats_deepseek",
        working_dir=Path("/tmp"),
        provider="deepseek",
        model=_DEEPSEEK_MODEL,
        skip_permissions=True,
        api_key=key,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_deepseek_agent_records_token_stats(deepseek_agent) -> None:
    client = AgentClient(deepseek_agent.agent_name)
    await client.connect()
    try:
        status = await _invoke_and_wait_idle(
            client, "Reply with exactly one word: OK"
        )
        assert status in ("idle", "ready"), f"agent did not go idle: {status}"
    finally:
        await client.disconnect()

    records = _read_records(deepseek_agent.agent_name)
    assert records, "token_stats.jsonl is empty — recorder is not writing"
    for rec in records:
        assert rec["provider"] == "deepseek"
        assert rec["kind"] == "agent"
        assert rec["schema_version"] == 1
    ok_records = [r for r in records if r["status"] == STATUS_OK]
    assert ok_records, "no status=ok record at all"
    # Previously the test accepted a record without tokens ("if prompt is
    # present — check the invariant"), so it could not distinguish working
    # statistics from all-no_usage. Require REAL tokens in at least one
    # status=ok record.
    with_tokens = [r for r in ok_records if r.get("prompt_tokens") is not None]
    assert with_tokens, (
        "no status=ok record contains prompt_tokens — usage does not arrive "
        "in the stream (check stream_usage=True in create_model and the "
        "usage_metadata parsing in normalize_usage)"
    )
    for rec in with_tokens:
        assert rec["prompt_tokens"] > 0, f"zero prompt_tokens: {rec}"
        assert rec["total_tokens"], f"no total_tokens: {rec}"
        hit = rec.get("cache_hit_tokens")
        miss = rec.get("cache_miss_tokens")
        if hit is not None and miss is not None:
            assert hit + miss == rec["prompt_tokens"], (
                f"invariant hit+miss=prompt violated: {rec}"
            )


@pytest.fixture
def deepseek_tm_agent():
    key = _deepseek_key()
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")
    with _make_agent(
        agent_name="test_stats_deepseek_tm",
        working_dir=Path("/tmp"),
        provider="deepseek",
        model=_DEEPSEEK_MODEL,
        skip_permissions=True,
        api_key=key,
        enable_task_manager=True,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_taskmanager_records_kind_tm_in_same_file(
    deepseek_tm_agent,
) -> None:
    client = AgentClient(deepseek_tm_agent.agent_name)
    await client.connect()
    try:
        status = await _invoke_and_wait_idle(
            client, "Reply with exactly one word: OK", idle_after=90.0
        )
        assert status in ("idle", "ready"), f"agent did not go idle: {status}"
    finally:
        await client.disconnect()

    records = _read_records(deepseek_tm_agent.agent_name)
    assert records, "token_stats.jsonl is empty"
    kinds = {r["kind"] for r in records}
    assert "agent" in kinds, f"no kind=agent records: {kinds}"
    assert "tm" in kinds, (
        f"TaskManager records (kind=tm) did not appear in the same file: {kinds}"
    )


# ---------------------------------------------------------------------------
# Custom providers (gated on local CLI / auth)
# ---------------------------------------------------------------------------
def _claude_available() -> bool:
    return shutil.which("claude") is not None


def _codex_available() -> bool:
    return shutil.which("codex") is not None


@pytest.fixture
def codex_agent(provider_login):
    if not _codex_available():
        pytest.skip("codex CLI not found; skipping integration test")
    codex_model = os.getenv("CODEX_MODEL", "gpt-5.5")
    with _make_agent(
        agent_name="test_stats_codex",
        working_dir=Path("/tmp"),
        provider="codex",
        model=codex_model,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_codex_agent_records_usage(codex_agent) -> None:
    client = AgentClient(codex_agent.agent_name)
    await client.connect()
    try:
        status = await _invoke_and_wait_idle(
            client, "Reply with exactly one word: OK", idle_after=120.0
        )
        assert status in ("idle", "ready"), f"agent did not go idle: {status}"
    finally:
        await client.disconnect()

    records = _read_records(codex_agent.agent_name)
    assert records, "token_stats.jsonl is empty — codex did not record usage"
    providers = {r["provider"] for r in records}
    assert "codex" in providers, f"no codex records: {providers}"


_GROK_ENABLED = os.environ.get("GROK_INTEGRATION_TESTS", "").lower() in (
    "1",
    "true",
    "yes",
)


@pytest.fixture
def grok_agent(provider_login):
    if not _GROK_ENABLED:
        pytest.skip("set GROK_INTEGRATION_TESTS=1 to enable live grok tests")
    if shutil.which("grok") is None:
        pytest.skip("grok CLI not found; skipping integration test")
    if not (Path.home() / ".grok" / "auth.json").exists():
        pytest.skip("~/.grok/auth.json missing; skipping integration test")
    with _make_agent(
        agent_name="test_stats_grok",
        working_dir=Path("/tmp"),
        provider="grok",
        model=None,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_grok_agent_records_token_usage(grok_agent) -> None:
    client = AgentClient(grok_agent.agent_name)
    await client.connect()
    try:
        status = await _invoke_and_wait_idle(
            client, "Reply with exactly one word: OK", idle_after=120.0
        )
        assert status in ("idle", "ready"), f"agent did not go idle: {status}"
    finally:
        await client.disconnect()

    records = _read_records(grok_agent.agent_name)
    # LIVE-RED-DISCOVERY (grok 0.2.73, DESIGN §2.2 live-probe): per-turn
    # usage arrives in the session/prompt RESULT._meta → a regular token
    # record (not usage_update/context_snapshot). cache hit = cachedReadTokens.
    usage_recs = [r for r in records if r.get("usage_semantics") == "tokens"]
    assert usage_recs, f"grok did not record token usage: {records}"
    for rec in usage_recs:
        assert rec["provider"] == "grok"
        assert rec["prompt_tokens"] is not None
        assert rec["completion_tokens"] is not None
        assert rec["total_tokens"] is not None
        assert rec["cache_hit_tokens"] is not None
        assert rec["cache_miss_tokens"] is not None


@pytest.fixture
def failover_agent():
    if not _claude_available():
        pytest.skip("claude CLI not found; skipping integration test")
    if not _deepseek_key():
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")
    # Simulate the effect of a failover restart: ONE agent (same data_dir),
    # but two different providers used sequentially. The real auto-failover
    # (usage-limit → _schedule_failover → deep-worker --restart
    # --provider <next>) is covered by test_grok_nested_402_failover_to_deepseek.py;
    # here we verify the token-statistics invariant — the recorder is bound
    # to the agent's data_dir, not to the provider, so after a provider
    # change records land in the SAME token_stats.jsonl.
    with _make_agent(
        agent_name="test_stats_failover",
        working_dir=Path("/tmp"),
        provider="claude-code",
        model=None,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_failover_records_two_providers_in_one_file(
    failover_agent,
) -> None:
    client = AgentClient(failover_agent.agent_name)
    await client.connect()
    try:
        status = await _invoke_and_wait_idle(
            client,
            "Reply with exactly one word: OK",
            idle_after=180.0,
        )
        assert status in ("idle", "ready"), f"claude-code agent did not go idle: {status}"
    finally:
        await client.disconnect()

    # Failover effect: the same agent (same data_dir), new provider.
    # Stop the claude-code process and restart the agent on deepseek.
    from .agent_fixture import _terminate_agent  # noqa: PLC0415
    from deep_worker_cli.process.registry import ProcessRegistry  # noqa: PLC0415

    _terminate_agent(failover_agent)
    ProcessRegistry().unregister(failover_agent.agent_name)

    key = _deepseek_key()
    with _make_agent(
        agent_name="test_stats_failover",
        working_dir=Path("/tmp"),
        provider="deepseek",
        model=_DEEPSEEK_MODEL,
        skip_permissions=True,
        api_key=key,
        enable_task_manager=False,
    ) as info2:
        client2 = AgentClient(info2.agent_name)
        await client2.connect()
        try:
            status2 = await _invoke_and_wait_idle(
                client2,
                "Reply with exactly one word: OK",
                idle_after=90.0,
            )
            assert status2 in ("idle", "ready"), (
                f"deepseek agent did not go idle: {status2}"
            )
        finally:
            await client2.disconnect()
        # Read BEFORE exiting the with block: fixture cleanup removes data_dir.
        records = _read_records(failover_agent.agent_name)

    assert records, "token_stats.jsonl is empty"
    providers = {r["provider"] for r in records}
    assert len(providers) >= 2, (
        f"after a provider change (failover effect) one file must contain "
        f"records from ≥2 providers, got: {providers}"
    )
