"""A parked tool job MUST be advertised under the canonical ``job_N``
namespace — never under the originating ``tool_use_id``.

Live incident (2026-07-31). Reproduced many times in one session:

    execute(...)
    → Tool 'execute' is still running after 120s. status: in_progress.
      job_id: toolu_01ABC…
    wait_for_job(job_id="toolu_01ABC…")
    → Error: no background job with id 'toolu_01ABC…'.

Root cause: TWO registries with different id namespaces and the SAME 120s
default. ``BoundedToolCallMiddleware`` parked under ``tool_call_id`` in a
registry nothing ever read, while ``wait_for_job``/``kill_job`` only ever
resolved the shell backend's ``job_N`` registry. The outer bound also covers
stage-A dispatch, so it fires FIRST on CLI providers — the backend never got
to mint ``job_N`` and the alias was never written.

The system, not the model, emitted an id it could not honour.

This suite pins the contract:
  * the in_progress notice advertises a canonical ``job_N``;
  * the parked job resolves by that ``job_N`` AND by the ``tool_use_id``
    alias (backwards compatible with the alias fix);
  * ids minted by the middleware and by the shell backend never collide;
  * ``ToolMessage.tool_call_id`` still round-trips the ``toolu_…`` value,
    because that field is the LangChain protocol — conflating it with the
    job id is what caused the defect.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from langchain.tools.tool_node import ToolCallRequest, ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.middleware.bounded_tool_call import (
    DEFAULT_TOOL_WAIT_SECONDS,
    BoundedToolCallMiddleware,
)

pytestmark = pytest.mark.asyncio

_JOB_RE = re.compile(r"\bjob_\d+\b")


def _request(
    tool_call_id: str = "toolu_TEST", name: str = "execute"
) -> ToolCallRequest:
    state = {"messages": []}
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    return ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": name, "args": {"x": 1}},
        state=state,
        tool=None,
    )


async def _park(
    mw: BoundedToolCallMiddleware, tool_call_id: str = "toolu_TEST"
) -> ToolMessage:
    """Drive a never-finishing tool past the bound and return the notice.

    Mocks ONLY the clock so the production bound fires instantly.
    """
    started = asyncio.Event()

    async def never_returns(_req: ToolCallRequest) -> ToolMessage:
        started.set()
        await asyncio.sleep(10_000)
        return ToolMessage(content="never", tool_call_id=tool_call_id)

    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(
            mw.awrap_tool_call(_request(tool_call_id), never_returns)
        )
        await started.wait()
        await asyncio.sleep(0.02)
        skew["v"] = DEFAULT_TOOL_WAIT_SECONDS + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        out = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]
    assert isinstance(out, ToolMessage)
    return out


def _notice_text(msg: ToolMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


# ── 1. the notice must advertise job_N, never the tool_use_id ─────────────


async def test_middleware_parks_under_job_n_namespace() -> None:
    mw = BoundedToolCallMiddleware()
    out = await _park(mw, "toolu_PARK1")
    text = _notice_text(out)

    assert out.status != "error", "a timeout is NEVER an error (POLICIES §7.1)"
    assert "in_progress" in text
    # The advertised job id is canonical...
    assert _JOB_RE.search(text), (
        "in_progress notice must advertise a canonical job_N id, "
        f"got: {text!r}"
    )
    # ...and is NOT the tool_use_id, which wait_for_job could never resolve.
    assert "toolu_PARK1" not in text, (
        "the notice must not offer the tool_use_id as job_id — that id is "
        "not in the job registry and wait_for_job/kill_job cannot honour it"
    )
    # The LangChain protocol field still round-trips the tool_use_id.
    assert out.tool_call_id == "toolu_PARK1"


# ── 2. the advertised id actually resolves ───────────────────────────────


async def test_advertised_job_id_resolves_in_registry() -> None:
    from deepagents.middleware.parked_job_registry import (
        get_parked_job_registry,
    )

    mw = BoundedToolCallMiddleware()
    out = await _park(mw, "toolu_PARK2")
    match = _JOB_RE.search(_notice_text(out))
    assert match is not None
    job_id = match.group(0)

    registry = get_parked_job_registry()
    assert registry.resolve(job_id) is not None, (
        f"the id advertised to the model ({job_id}) must resolve — "
        "advertising an unresolvable id is the whole defect"
    )


# ── 3. backwards compatibility with the alias fix ────────────────────────


async def test_parked_tool_job_also_resolves_by_tool_use_alias() -> None:
    from deepagents.middleware.parked_job_registry import (
        get_parked_job_registry,
    )

    mw = BoundedToolCallMiddleware()
    await _park(mw, "toolu_PARK3")

    registry = get_parked_job_registry()
    by_alias = registry.resolve("toolu_PARK3")
    assert by_alias is not None, (
        "a model that passes the originating tool_use_id must still reach "
        "the job (alias contract)"
    )


# ── 4. the two layers share one counter → no id collision ────────────────


async def test_shell_and_middleware_job_ids_never_collide() -> None:
    from deepagents.middleware.parked_job_registry import (
        get_parked_job_registry,
    )

    registry = get_parked_job_registry()
    mw = BoundedToolCallMiddleware()

    first = _JOB_RE.search(_notice_text(await _park(mw, "toolu_A")))
    # An independent allocation (as the shell backend does) must not reuse it.
    shell_id = registry.allocate()
    second = _JOB_RE.search(_notice_text(await _park(mw, "toolu_B")))

    assert first is not None and second is not None
    ids = {first.group(0), shell_id, second.group(0)}
    assert len(ids) == 3, f"job ids must be unique across layers, got {ids}"


# ── 5. a finished tool job leaves no leak ────────────────────────────────


async def test_finished_tool_job_is_dropped_from_registry() -> None:
    from deepagents.middleware.parked_job_registry import (
        get_parked_job_registry,
    )

    mw = BoundedToolCallMiddleware()
    started = asyncio.Event()
    release = asyncio.Event()
    tool_call_id = "toolu_DROP"

    async def slow(_req: ToolCallRequest) -> ToolMessage:
        started.set()
        await release.wait()
        return ToolMessage(content="finished", tool_call_id=tool_call_id)

    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(
            mw.awrap_tool_call(_request(tool_call_id), slow)
        )
        await started.wait()
        await asyncio.sleep(0.02)
        skew["v"] = DEFAULT_TOOL_WAIT_SECONDS + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        out = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]

    match = _JOB_RE.search(_notice_text(out))
    assert match is not None
    job_id = match.group(0)

    registry = get_parked_job_registry()
    assert registry.resolve(job_id) is not None

    # Let the parked work finish; the registry must drop it event-driven
    # (done callback), with no polling.
    release.set()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.resolve(job_id) is None:
            break
    assert registry.resolve(job_id) is None, (
        "a finished parked job must be dropped from the registry — and its "
        "alias with it (no stale alias, no leak)"
    )
    assert registry.resolve(tool_call_id) is None
