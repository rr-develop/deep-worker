"""The ONE place where every tool call is time-bounded — for ALL providers.

HARD POLICY (POLICIES.md §7.1, owner 2026-06-17): a tool call may never hang
the agent forever, and the bound is enforced in EXACTLY ONE function —
``BoundedToolCallMiddleware.awrap_tool_call`` — that wraps tool execution in
LangGraph's ToolNode. Because middleware lives on the agent graph (see
``deepagents/graph.py``), this hook wraps EVERY tool of EVERY provider
(deepseek / openai / anthropic / google / vertexai / claude-code-sdk / codex /
…) identically. No provider, and no other code path, may wait on a tool.

Contract on the wait elapsing (default 120s, env
``DW_LONG_RUNNING_WAIT_SECONDS``):
  * the tool's work is PARKED in the background — NOT cancelled (``shield``);
  * the model gets a NON-error ``in_progress`` ``ToolMessage`` carrying a
    CANONICAL ``job_N`` as ``job_id`` (minted from the shared
    ``ParkedJobRegistry`` — the same namespace ``wait_for_job`` / ``kill_job``
    resolve) plus those two actions. The ``tool_call_id`` is recorded only as
    an alias; advertising it as the ``job_id`` was an internal defect
    (2026-07-31) and is forbidden;
  * a timeout is NEVER ``status="error"`` — it informs and the agent
    continues / retries.

There is NO second timer and NO bare ``await`` on a tool anywhere else. The
per-provider proxy servers delegate their bound here (see
``DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.middleware.parked_job_registry import get_parked_job_registry
from deepagents.middleware.tool_dispatch_bridge import get_dispatch_waiter

DEFAULT_TOOL_WAIT_SECONDS = 120.0

IMAGINE_TOOL_WAIT_SECONDS = 600.0

_IMAGINE_BUILTIN_TITLES = frozenset({
    "GenerateImage",
    "image_gen",
    "image_edit",
    "EditImage",
    "image_to_video",
    "reference_to_video",
})

TOOL_WAIT_OVERRIDES: dict[str, float] = {
    name: IMAGINE_TOOL_WAIT_SECONDS for name in _IMAGINE_BUILTIN_TITLES
}


def resolve_tool_wait_seconds(explicit: float | None = None) -> float:
    """THE single source of truth for the per-tool-call bound, shared by every
    place that may bound a tool wait (this middleware AND a provider's stage-A
    dispatch wait in ``ainvoke`` — variant A, see
    docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md).

    Resolution: explicit arg > ``DW_LONG_RUNNING_WAIT_SECONDS`` env > the policy
    default (120s). A non-positive / unparseable env is ignored. There must be
    NO other timer constant for tool waits anywhere.
    """
    if explicit is not None and explicit > 0:
        return explicit
    raw = os.environ.get("DW_LONG_RUNNING_WAIT_SECONDS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_TOOL_WAIT_SECONDS


def _imagine_wait_seconds() -> float:
    raw = os.environ.get("DW_IMAGINE_WAIT_SECONDS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return IMAGINE_TOOL_WAIT_SECONDS


def resolve_effective_tool_wait_seconds(
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    default: float | None = None,
) -> float:
    """Per-tool bound override. Imagine tools get a longer wait (600s default,
    overridable via ``DW_IMAGINE_WAIT_SECONDS``). ``GrokBuiltin`` uses the
    imagine bound only when ``_grok_original_title`` names an imagine tool."""
    base = resolve_tool_wait_seconds(default)
    if tool_name == "GrokBuiltin":
        orig = (args or {}).get("_grok_original_title")
        if orig in _IMAGINE_BUILTIN_TITLES:
            return _imagine_wait_seconds()
        return base
    override = TOOL_WAIT_OVERRIDES.get(tool_name)
    if override is not None:
        return _imagine_wait_seconds() if tool_name in _IMAGINE_BUILTIN_TITLES else override
    return base


def build_in_progress_notice(tool_name: str, job_id: str, wait_seconds: float) -> str:
    """THE single in_progress message shared by every bounded wait site, so a
    timeout reads identically wherever the bound fires. NEVER an error.

    ``job_id`` MUST be a canonical ``job_N`` from ``ParkedJobRegistry`` — the
    caller is asserting that real work has been parked and can be reached by
    ``wait_for_job`` / ``kill_job``. Callers that never parked work (e.g.
    CLI-provider stage-A dispatch timeouts, where a control request never
    arrived) MUST use ``build_dispatch_timeout_notice`` instead — advertising a
    ``job_id`` for work that does not exist is the same internal defect
    (2026-07-31). See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.
    """
    return (
        f"Tool '{tool_name}' is still running after {wait_seconds:g}s. "
        f"status: in_progress. job_id: {job_id}. "
        f"actions: wait_for_job(job_id=\"{job_id}\", timeout=<seconds>) "
        f"or kill_job(job_id=\"{job_id}\")."
    )


def build_dispatch_timeout_notice(tool_name: str, wait_seconds: float) -> str:
    """CLI stage-A dispatch timeout notice: the CLI announced the tool call
    but the control request never arrived within the bound.

    NO job was parked — there is no process, no task, no future for
    ``wait_for_job`` / ``kill_job`` to key off. Advertising a ``job_id`` here
    (which the two provider sites did until the follow-up to 6bd042b)
    reproduced the very ``no background job with id '…'`` symptom the
    ParkedJobRegistry fix removed for real parks. This notice states the
    honest truth: the tool did not start, no job exists, the model should
    retry.

    NEVER an error (POLICIES §7.1). See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.
    """
    return (
        f"Tool '{tool_name}' did not start within {wait_seconds:g}s "
        f"(the CLI announced the call but the control request never arrived). "
        f"status: not_started. No background job was created. "
        f"The turn ends; retry the tool call."
    )


class _ParkedToolJob:
    """A tool whose execution outlived the bound. Its work keeps running (it
    was NOT cancelled); ``wait_for_job`` / ``kill_job`` pick it up later by its
    canonical ``job_N`` id, or by the originating ``tool_call_id`` alias.

    ``job_id`` is a canonical ``job_N`` minted from ``ParkedJobRegistry`` — it
    is NOT the ``tool_call_id``. Advertising the ``tool_call_id`` as a job id
    is precisely the internal defect (2026-07-31): ``wait_for_job`` could never
    resolve it.
    """

    __slots__ = ("job_id", "tool_name", "task", "tool_call_id")

    def __init__(
        self,
        job_id: str,
        tool_name: str,
        task: "asyncio.Task[Any]",
        tool_call_id: str = "",
    ) -> None:
        self.job_id = job_id
        self.tool_name = tool_name
        self.task = task
        self.tool_call_id = tool_call_id


class BoundedToolCallMiddleware(AgentMiddleware):
    """Single source of truth for the per-tool-call timeout (POLICIES.md §7.1).

    Wrap order: this middleware MUST be the OUTERMOST tool-call wrapper so the
    bound covers every inner tool-call middleware too. See ``graph.py``.
    """

    def __init__(self, wait_seconds: float | None = None) -> None:
        super().__init__()
        self.wait_seconds = resolve_tool_wait_seconds(wait_seconds)
        # Parked jobs keyed by canonical ``job_N`` (NOT tool_call_id — that is
        # the alias, and conflating the two was an internal defect); the work
        # continues here after the bound fires so it is never lost.
        self._jobs: dict[str, _ParkedToolJob] = {}

    def get_job_registry(self) -> dict[str, _ParkedToolJob]:
        """Snapshot of this middleware's parked tool jobs, keyed by canonical
        ``job_N``.

        A COPY on purpose: an entry disappears the moment its result is served
        to a caller, so handing out the live dict would let a caller iterating
        it hit "dictionary changed size during iteration". (Completion alone no
        longer removes anything — a finished job stays reachable until someone
        collects it.) The shared ``ParkedJobRegistry`` is the source of truth;
        this accessor exists for diagnostics and tests.
        """
        return dict(self._jobs)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Sync path (``invoke``/``stream``). Same single-source bound: run the
        tool on a worker thread and wait at most ``wait_seconds``; on the bound
        elapsing the thread keeps running (work NOT lost) and the model gets the
        SAME NON-error in_progress notice. (Production runs async — see
        ``awrap_tool_call`` — but this keeps the policy hole-free for the sync
        path too. The stage-A dispatch wait is async-only and does not apply
        here, since a sync invocation never drives a CLI proxy provider.)"""
        import concurrent.futures

        tool_call_id = request.tool_call.get("id") or ""
        tool_name = request.tool_call.get("name") or "tool"
        args = request.tool_call.get("args") or {}
        wait_seconds = resolve_effective_tool_wait_seconds(
            tool_name, args, default=self.wait_seconds
        )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, request)
        try:
            result = future.result(timeout=wait_seconds)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            # Work continues on the thread (NOT cancelled). Do not shut the
            # executor down with wait=True — that would block on the timeout.
            #
            # Park under a canonical ``job_N`` exactly like the async path, so
            # the advertised id is always resolvable and the two paths cannot
            # drift apart again. NOTE: a thread cannot be cancelled, so
            # ``kill_job`` on such a job can only report that it is still
            # running — see ``ParkedJob.handle`` (a concurrent Future).
            job_id = get_parked_job_registry().park(
                future,
                alias=tool_call_id or None,
                tool_name=tool_name,
            ).job_id
            return ToolMessage(
                content=build_in_progress_notice(
                    tool_name, job_id, wait_seconds
                ),
                tool_call_id=tool_call_id,
                status="success",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_call_id = request.tool_call.get("id") or ""
        tool_name = request.tool_call.get("name") or "tool"
        args = request.tool_call.get("args") or {}
        wait_seconds = resolve_effective_tool_wait_seconds(
            tool_name, args, default=self.wait_seconds
        )

        thread_id = (
            request.runtime.config.get("configurable", {}).get("thread_id")
            if request.runtime is not None and request.runtime.config is not None
            else None
        )

        # Run BOTH stages under ONE bound:
        #   A) provider dispatch wait (CLI providers): block until the control
        #      request arrives and yields authoritative args. No-op for normal
        #      providers (no waiter registered).
        #   B) tool execution via the inner handler.
        # The task is shielded so the bound fires WITHOUT cancelling the work
        # (it keeps running and is parked as a job).
        async def _dispatch_then_execute() -> ToolMessage | Command:
            waiter = get_dispatch_waiter(thread_id)
            if waiter is not None:
                authoritative_args = await waiter(tool_call_id)
                if authoritative_args is not None:
                    # Authoritative args from the control channel override the
                    # announced args (internal fix 2026-05-26).
                    request.tool_call["args"] = authoritative_args
            return await handler(request)

        task: asyncio.Task[ToolMessage | Command] = asyncio.ensure_future(
            _dispatch_then_execute()
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=wait_seconds
            )
        except asyncio.TimeoutError:
            # Park the live task (NOT cancelled, NOT lost) and inform the
            # model. NEVER an error.
            #
            # The advertised job_id MUST be a canonical ``job_N`` from the
            # shared registry — the same namespace ``wait_for_job`` /
            # ``kill_job`` resolve. Handing back ``tool_call_id`` (a
            # ``toolu_…``) was the internal defect: the model got an id no
            # registry could honour and lost control of live background work.
            # ``ToolMessage.tool_call_id`` below still carries the
            # ``toolu_…`` because that field is the LangChain protocol — it
            # is NOT a job id, and conflating the two caused the bug.
            job_id = self._park_task(task, tool_name, tool_call_id)
            return ToolMessage(
                content=build_in_progress_notice(
                    tool_name, job_id, wait_seconds
                ),
                tool_call_id=tool_call_id,
                status="success",
            )

    def _park_task(
        self,
        task: "asyncio.Task[Any]",
        tool_name: str,
        tool_call_id: str,
    ) -> str:
        """Register a still-running tool task under a canonical ``job_N`` and
        return that id. The originating ``tool_call_id`` is recorded as an
        alias so a model that echoes it still reaches the job.

        When the work finishes the job is RETIRED, not dropped — event-driven
        via ``add_done_callback``, never polled. It stays reachable so a caller
        returning after completion still gets the result; the entry is erased
        only once that result has been served.
        """
        parked = _ParkedToolJob(
            job_id="",
            tool_name=tool_name,
            task=task,
            tool_call_id=tool_call_id,
        )
        def _forget(jid: str) -> None:
            self._jobs.pop(jid, None)

        job = get_parked_job_registry().park(
            parked,
            alias=tool_call_id or None,
            tool_name=tool_name,
            on_release=_forget,
        )
        parked.job_id = job.job_id
        self._jobs[job.job_id] = parked
        return job.job_id
