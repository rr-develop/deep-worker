"""Live proof, no mocks: the context moves on EVERY model call.

POLICIES §13 "Cadence" (owner directive, 2026-08-25): a provider turn is
not one model call — grok has been measured at 98 inferences inside a
single turn — so a context number that only arrives with the final answer
cannot guard that turn, cannot move the status bar while the agent works,
and arrives too late to summarize.

The unit suites pin the mechanism (``test_grok_live_context.py``,
``test_codex_live_context.py``). They
cannot prove the CLI actually emits what we parse, that the parser finds
the live session, or that anything reaches the recorder in a real process.
These tests do: a REAL agent process on a REAL provider CLI, given work
that cannot be done in one model call, watched through the file the web UI
itself reads (``~/.deep_worker/<agent>/token_stats.jsonl``).

Nothing here is mocked, stubbed or monkeypatched. The only thing the test
supplies is the prompt.

What each test asserts, and why each assertion is not redundant:

  1. at least TWO context readings for one invoke — one reading is what the
     defect produced;
  2. the readings are not all the same number — a gauge that repeats one
     value is frozen, not live;
  3. more readings than terminal per-turn records — if the number only
     moved with the final answer these two counts would be EQUAL. This is
     the assertion that distinguishes "reports per turn" from "reports per
     model call", and it is the one the old code failed;
  4. a reading arrived while the agent was still working — observed by
     polling the file until the agent reports idle. This is the user-facing
     claim ("the number must change while the agent is working"), and no
     end-of-turn implementation can satisfy it.

Gating follows the neighbourhood (``test_token_stats_live.py``): each
provider is skipped unless its CLI/auth is present, and the whole module is
``-m integration``, deselected by default.

⚠ BASELINE. ``agent_fixture`` reuses a pre-existing agent when one is
running, and ``token_stats.jsonl`` is append-only across runs. Every count
below is taken against a baseline captured immediately before the invoke —
without it a stale file greens the test on someone else's readings.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.stats.token_stats import read_stats_file
from deep_worker_cli.storage_paths import agent_root

from .agent_fixture import agent_fixture as _make_agent
from .provider_home import provider_login  # noqa: F401 — pytest fixture

pytestmark = pytest.mark.integration


# Work that cannot be finished in a single model call: each step needs the
# result of the previous one, so the CLI must go back to the model between
# them. Deliberately trivial per step — we are measuring the gauge, not the
# model's reasoning.
def multi_step_prompt(scratch: Path) -> str:
    """Work that cannot be finished in a single model call.

    Each step needs the previous step's result, so the CLI must go back to
    the model between them — that is the whole fixture. Deliberately trivial
    per step: we are measuring the gauge, not the model's reasoning.

    The path is ABSOLUTE. Saying "in the working directory" left it to the
    CLI to resolve, and a run duly wrote ``steps.txt`` into the repository
    checkout — a test that litters the tree it is testing.
    """
    target = scratch / "steps.txt"
    return (
        "Do the following, ONE step at a time, checking the result of each "
        "step before starting the next. Use the absolute path "
        f"{target} throughout and write nowhere else:\n"
        f"1. create {target} containing the single line 'one'\n"
        "2. append a line 'two' to it\n"
        "3. append a line 'three' to it\n"
        f"4. read {target} back\n"
        "Then reply with the file's contents on one line."
    )


def _stats_path(agent_name: str) -> Path:
    """Where the agent ACTUALLY writes its token stats.

    Resolved through ``agent_root`` — the same helper production uses — and
    never hand-assembled. The older ``~/.deep_worker/<name>/`` layout still
    appears in some live tests; against a real agent it silently resolves to
    an empty directory, which reads as "the provider reported nothing" and
    is indistinguishable from the defect these tests exist to catch
    (measured 2026-08-26: a 53 s grok turn scored zero readings that way).
    """
    return agent_root(agent_name, create=False) / "token_stats.jsonl"


def _records(agent_name: str) -> list[dict]:
    path = _stats_path(agent_name)
    if not path.exists():
        return []
    return [r.to_dict() for r in read_stats_file(path).records]


def _is_reading(rec: dict) -> bool:
    """A context reading — the row the gauge and the web UI read."""
    used = rec.get("context_used")
    return isinstance(used, int) and not isinstance(used, bool) and used > 0


def _is_terminal_turn(rec: dict) -> bool:
    """A per-turn spend row, i.e. one provider turn finished."""
    prompt = rec.get("prompt_tokens")
    return isinstance(prompt, int) and not isinstance(prompt, bool) and prompt > 0


@dataclass
class _Watch:
    """What the poller saw, and when."""

    readings: list[int] = field(default_factory=list)
    terminal_turns: int = 0
    readings_before_idle: int = 0
    idle_status: str = ""
    seconds: float = 0.0
    stats_path: str = ""
    records_total: int = 0

    def report(self) -> str:
        return (
            f"readings={self.readings} "
            f"(before idle: {self.readings_before_idle}) "
            f"terminal_turns={self.terminal_turns} "
            f"status={self.idle_status!r} after {self.seconds:.1f}s "
            f"stats={self.stats_path} (total rows {self.records_total})"
        )


async def _run_and_watch(
    agent_name: str, prompt: str, *, timeout: float
) -> _Watch:
    """Invoke the agent for real and watch the stats file WHILE it works.

    The invoke runs as its own task on purpose. ``invoke_agent`` does not
    return until the agent has produced its answer, so awaiting it first and
    polling afterwards can only ever see the finished state — the poller
    would report "nothing arrived before the answer" for a perfectly live
    gauge, which is a test that cannot fail correctly.

    The polling itself is pure observation: it reads the same append-only
    file the web UI reads and never touches the agent.
    """
    client = AgentClient(agent_name)
    watch = _Watch()
    await client.connect()
    try:
        baseline = len(_records(agent_name))
        started = time.monotonic()
        turn = asyncio.create_task(client.invoke_agent(prompt, stream=False))

        deadline = started + timeout
        while time.monotonic() < deadline:
            if turn.done():
                break
            new_records = _records(agent_name)[baseline:]
            # Everything visible now predates the answer: the turn's task
            # has not returned yet.
            watch.readings_before_idle = sum(
                1 for r in new_records if _is_reading(r)
            )
            await asyncio.sleep(0.5)

        with contextlib.suppress(Exception):
            await asyncio.wait_for(turn, timeout=max(5.0, deadline - time.monotonic()))
        watch.idle_status = await _wait_until_idle(
            client, time.monotonic() + 60.0
        )

        new_records = _records(agent_name)[baseline:]
        watch.readings = [r["context_used"] for r in new_records if _is_reading(r)]
        watch.terminal_turns = sum(1 for r in new_records if _is_terminal_turn(r))
        watch.seconds = time.monotonic() - started
        watch.stats_path = str(_stats_path(agent_name))
        watch.records_total = len(_records(agent_name))
    finally:
        await client.disconnect()
    return watch


async def _wait_until_idle(client: AgentClient, deadline: float) -> str:
    last = ""
    while time.monotonic() < deadline:
        state = (await client._send_command({"command": "status"})) or {}
        last = str(state.get("status", "")).lower()
        if last in ("idle", "ready"):
            return last
        await asyncio.sleep(0.5)
    return last


def _assert_gauge_moved_per_model_call(watch: _Watch, provider: str) -> None:
    """The four claims of POLICIES §13 "Cadence", checked on live data."""
    # Printed, not merely asserted: a live test is also evidence, and the
    # numbers are the point. Visible with -s.
    print(f"[live-context] {provider}: {watch.report()}")
    assert watch.idle_status in ("idle", "ready"), (
        f"{provider}: the agent never finished the turn — {watch.report()}"
    )
    assert len(watch.readings) >= 2, (
        f"{provider}: {len(watch.readings)} context reading(s) for a turn that "
        f"needed several model calls — the gauge reports once per turn, not "
        f"per model call. {watch.report()}"
    )
    assert len(set(watch.readings)) >= 2, (
        f"{provider}: the same number was reported {len(watch.readings)} "
        f"times — the gauge is frozen, not live. {watch.report()}"
    )
    assert len(watch.readings) > watch.terminal_turns, (
        f"{provider}: {len(watch.readings)} readings against "
        f"{watch.terminal_turns} finished turn(s). Equal counts mean the "
        f"number moves only with the final answer — the defect. "
        f"{watch.report()}"
    )
    assert watch.readings_before_idle >= 1, (
        f"{provider}: no reading arrived while the agent was still working; "
        f"all {len(watch.readings)} appeared with the final answer. This is "
        f"exactly the frozen web-UI gauge. {watch.report()}"
    )


# ---------------------------------------------------------------------------
# grok — ACP shared layer, reads the CLI's own unified log
# ---------------------------------------------------------------------------
_GROK_ENABLED = os.environ.get("GROK_INTEGRATION_TESTS", "").lower() in (
    "1",
    "true",
    "yes",
)


@pytest.fixture
def grok_live_ctx_agent(tmp_path, provider_login):
    if not _GROK_ENABLED:
        pytest.skip("set GROK_INTEGRATION_TESTS=1 to enable live grok tests")
    if shutil.which("grok") is None:
        pytest.skip("grok CLI not found")
    if not (Path.home() / ".grok" / "auth.json").exists():
        pytest.skip("~/.grok/auth.json missing")
    with _make_agent(
        agent_name="test_livectx_grok",
        working_dir=tmp_path,
        provider="grok",
        model=None,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_grok_context_moves_on_every_model_call(grok_live_ctx_agent, tmp_path) -> None:
    """grok: ``LiveContextWatcher`` over ``~/.grok/logs/unified.jsonl``.

    The reading is ``ctx.prompt_tokens`` of each ``shell.turn.inference_done``
    filtered to OUR session id. Before the fix the number reaching the
    recorder came from the ``session/prompt`` reply, i.e. once per turn.
    """
    watch = await _run_and_watch(
        grok_live_ctx_agent.agent_name,
        multi_step_prompt(tmp_path),
        timeout=420.0,
    )
    _assert_gauge_moved_per_model_call(watch, "grok")


# ---------------------------------------------------------------------------
# codex — reads the session rollout
# ---------------------------------------------------------------------------
@pytest.fixture
def codex_live_ctx_agent(tmp_path, provider_login):
    if shutil.which("codex") is None:
        pytest.skip("codex CLI not found")
    with _make_agent(
        agent_name="test_livectx_codex",
        working_dir=tmp_path,
        provider="codex",
        # None on purpose: the codex provider's contract is that without
        # --model the CLI picks its own current default. Pinning a stale
        # name here routes the request to api.openai.com and 401s.
        model=os.getenv("CODEX_MODEL") or None,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_codex_context_moves_on_every_model_call(codex_live_ctx_agent, tmp_path) -> None:
    """codex: ``RolloutContextWatcher`` over the session rollout JSONL.

    The reading is each ``token_count`` event. Before the fix only
    ``turn.completed`` reached the recorder — one row per ``codex exec``
    turn, however many model calls it spanned.
    """
    watch = await _run_and_watch(
        codex_live_ctx_agent.agent_name,
        multi_step_prompt(tmp_path),
        timeout=420.0,
    )
    _assert_gauge_moved_per_model_call(watch, "codex")


# ---------------------------------------------------------------------------
# claude-code (ACP) — reads the wire protocol's own usage_update
# ---------------------------------------------------------------------------
@pytest.fixture
def claude_acp_live_ctx_agent(tmp_path, provider_login):
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not found")
    with _make_agent(
        agent_name="test_livectx_claude_acp",
        working_dir=tmp_path,
        provider="claude-code",
        model=None,
        skip_permissions=True,
        api_key=None,
        enable_task_manager=False,
    ) as info:
        yield info


@pytest.mark.asyncio
async def test_claude_acp_context_moves_on_every_model_call(
    claude_acp_live_ctx_agent, tmp_path
) -> None:
    """claude-code over ACP: a reading per ``usage_update``.

    No log to tail here — the wire protocol already carries the CLI's own
    gauge during the turn. The layer received those updates and parked them,
    writing a row only at the turn boundary, so the web UI showed
    "Provider: claude-code / Model: opus / Context: 286.2k / Limit: 250k"
    for a live agent whose real context was 167831: the provider and model
    came from the running agent, the number from the PREVIOUS process on
    grok. This is the test that would have caught it.
    """
    watch = await _run_and_watch(
        claude_acp_live_ctx_agent.agent_name,
        multi_step_prompt(tmp_path),
        timeout=420.0,
    )
    _assert_gauge_moved_per_model_call(watch, "claude-code")
