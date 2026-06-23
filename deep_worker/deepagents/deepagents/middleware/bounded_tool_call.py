"""The ONE place where every tool call is time-bounded — for ALL providers.

HARD POLICY (POLICIES.md §7.1, owner 2026-06-17): a tool call may never hang
the agent forever, and the bound is enforced in EXACTLY ONE function —
``BoundedToolCallMiddleware.awrap_tool_call`` — that wraps tool execution in
LangGraph's ToolNode. Because middleware lives on the agent graph (see
``deepagents/graph.py``), this hook wraps EVERY tool of EVERY provider
(deepseek / openai / anthropic / google / vertexai / claude-code / codex /
…) identically. No provider, and no other code path, may wait on a tool.

Contract on the wait elapsing (default 120s, env
``DW_LONG_RUNNING_WAIT_SECONDS``):
  * the tool's work is PARKED in the background — NOT cancelled (``shield``);
  * the model gets a NON-error ``in_progress`` ``ToolMessage`` carrying the
    ``tool_call_id`` as ``job_id`` plus the ``wait_for_job`` / ``kill_job``
    actions;
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
    timeout reads identically wherever the bound fires. NEVER an error."""
    return (
        f"Tool '{tool_name}' is still running after {wait_seconds:g}s. "
        f"status: in_progress. job_id: {job_id}. "
        f"actions: wait_for_job(job_id=\"{job_id}\", timeout=<seconds>) "
        f"or kill_job(job_id=\"{job_id}\")."
    )


class _ParkedToolJob:
    """A tool whose execution outlived the bound. Its task keeps running
    (the work was NOT cancelled); ``wait_for_job`` / ``kill_job`` can pick it
    up later by ``job_id`` (== the originating ``tool_call_id``)."""

    __slots__ = ("job_id", "tool_name", "task")

    def __init__(
        self, job_id: str, tool_name: str, task: "asyncio.Task[Any]"
    ) -> None:
        self.job_id = job_id
        self.tool_name = tool_name
        self.task = task


class BoundedToolCallMiddleware(AgentMiddleware):
    """Single source of truth for the per-tool-call timeout (POLICIES.md §7.1).

    Wrap order: this middleware MUST be the OUTERMOST tool-call wrapper so the
    bound covers every inner tool-call middleware too. See ``graph.py``.
    """

    def __init__(self, wait_seconds: float | None = None) -> None:
        super().__init__()
        self.wait_seconds = resolve_tool_wait_seconds(wait_seconds)
        # Parked jobs keyed by tool_call_id; the work continues here after the
        # bound fires so it is never lost.
        self._jobs: dict[str, _ParkedToolJob] = {}

    def get_job_registry(self) -> dict[str, _ParkedToolJob]:
        return self._jobs

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
            return ToolMessage(
                content=build_in_progress_notice(
                    tool_name, tool_call_id, wait_seconds
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
                    # announced args (observed live, 2026-05-26).
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
            self._jobs[tool_call_id] = _ParkedToolJob(
                job_id=tool_call_id, tool_name=tool_name, task=task
            )
            return ToolMessage(
                content=build_in_progress_notice(
                    tool_name, tool_call_id, wait_seconds
                ),
                tool_call_id=tool_call_id,
                status="success",
            )
