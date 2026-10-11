"""The agent-side client. Imports grpc, stdlib, and nothing heavier.

This module runs inside every agent process, so what it must NOT import
matters as much as what it does. No llama-index, no LanceDB, no torch —
that is the entire point of the service, and
``test_package_boundary`` walks the import graph to keep it true.

Two responsibilities live here rather than in the service:

**Resolving defaults.** The wire address is always explicit (§5): only
the agent knows its own working directory and its own name, so
``repo_path=None`` and ``agent_name=None`` are resolved here and the
service is told exactly which tenant to serve.

**Surviving a restart.** The service is not permanent — it shuts itself
down when idle (§11.9) — so an UNAVAILABLE is an ordinary event, not a
failure. The client re-spawns once and retries. Once: a loop would turn
a service that cannot start into an agent that never answers.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import grpc

from deep_worker_cli.storage_paths import deep_worker_home
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb2
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2_grpc as pb2_grpc

#: Deadlines, per design §4.3. Search is generous because the first
#: search after a cold start also pays the ~7 s model load; a tighter
#: deadline would fail exactly the requests that arrive first.
HEALTH_TIMEOUT_S = 5.0
SEARCH_TIMEOUT_S = 30.0
INDEX_TIMEOUT_S = 120.0
STATUS_TIMEOUT_S = 5.0

MIN_TOP_K = 1
MAX_TOP_K = 20

#: How many tenant names to show in a "no index here" message before
#: summarizing the rest. Long enough to be useful, short enough not to
#: bury the instruction that follows it.
_TENANTS_SHOWN = 10


class SemanticMemoryUnavailable(RuntimeError):
    """The service could not be started or reached.

    Deliberately not silently swallowed: policy §3.7 forbids falling back
    to an in-process import, since a quiet fallback would restore exactly
    the 884 MB cost the service exists to remove — and would do it
    invisibly, so nobody would notice the regression.
    """


def clamp_top_k(top_k: int) -> str | None:
    """Return an error message if *top_k* is out of range, else None.

    Checked before the RPC so a typo costs a round trip instead of a
    model load.
    """
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        return f"Error: top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
    return None


def resolve_project_path(
    runtime: Any,
    working_dir: str | None,
    repo_path: str | Path | None = None,
) -> Path:
    if repo_path is not None:
        return Path(repo_path).resolve()
    state = getattr(runtime, "state", None)
    if state:
        runtime_dir = state.get("working_dir")
        if runtime_dir:
            return Path(runtime_dir).resolve()
    if working_dir is not None:
        return Path(working_dir).resolve()
    return Path.cwd().resolve()


def resolve_history_target(
    runtime: Any,
    agent_name: str | None,
) -> tuple[Path, str]:
    if agent_name is not None:
        path = deep_worker_home() / "agents" / agent_name / "summaries"
        return path.resolve(), agent_name
    state = getattr(runtime, "state", None)
    summaries_dir = state.get("summaries_dir") if state else None
    path = Path(summaries_dir) if summaries_dir else deep_worker_home()
    return path.resolve(), "current agent"


class SemanticMemoryClient:
    """Thin gRPC client with auto-start and a single retry."""

    def __init__(self, supervisor: Any | None = None) -> None:
        self._supervisor = supervisor

    def _get_supervisor(self) -> Any:
        """Build the supervisor lazily.

        Imported inside the method so merely constructing a client — as
        tool registration does — does not reach into ``service``.
        """
        if self._supervisor is None:
            from deep_worker_cli.semantic_memory.service.supervisor import (  # noqa: PLC0415
                ServiceSupervisor,
            )
            from deep_worker_cli.semantic_memory.registry import ServiceRegistry  # noqa: PLC0415
            from deep_worker_cli.semantic_memory.service.__main__ import (  # noqa: PLC0415
                build_serve_argv,
            )

            self._supervisor = ServiceSupervisor(
                registry=ServiceRegistry(),
                spawn_argv=build_serve_argv,
            )
        return self._supervisor

    async def _ensure_socket(self) -> Path:
        """Start the service if needed and return its socket path.

        ``ensure_started`` blocks — it waits on inotify and a pidfd — so
        it goes to a thread. Called on the event loop it would freeze the
        whole agent for the duration of a service start.
        """
        supervisor = self._get_supervisor()
        try:
            return await asyncio.to_thread(supervisor.ensure_started)
        except Exception as exc:
            raise SemanticMemoryUnavailable(str(exc)) from exc

    async def _call(
        self,
        method: str,
        request: Any,
        timeout: float,
    ) -> Any:
        """Issue an RPC, re-spawning once if the service went away.

        The retry exists because idle-shutdown is normal operation: an
        agent that pauses for half an hour and then searches will always
        find the service gone. Retrying exactly once distinguishes that
        from a service that cannot start at all, which must surface as an
        error instead of an infinite spawn loop.
        """
        for attempt in (1, 2):
            socket_path = await self._ensure_socket()
            try:
                async with grpc.aio.insecure_channel(f"unix://{socket_path}") as ch:
                    stub = pb2_grpc.SemanticMemoryStub(ch)
                    return await getattr(stub, method)(request, timeout=timeout)
            except grpc.aio.AioRpcError as exc:
                if exc.code() is not grpc.StatusCode.UNAVAILABLE or attempt == 2:
                    raise
                # Drop the dead record so the next ensure_started spawns
                # instead of handing back the same unusable socket.
                self._get_supervisor().registry.drop()

    # -- search ------------------------------------------------------------

    async def search_project(self, query: str, repo_path: Path, top_k: int) -> str:
        reply = await self._call(
            "SearchProject",
            pb2.ProjectSearchRequest(
                repo_path=str(repo_path), query=query, top_k=top_k
            ),
            SEARCH_TIMEOUT_S,
        )
        return reply.result.content

    async def health(self) -> Any:
        return await self._call("Health", pb2.HealthRequest(), HEALTH_TIMEOUT_S)

    async def status(self) -> Any:
        return await self._call("Status", pb2.StatusRequest(), STATUS_TIMEOUT_S)

    async def session_events(
        self,
        client_id: str,
        stop: asyncio.Event,
    ) -> AsyncIterator[Any]:
        while not stop.is_set():
            socket_path = await self._ensure_socket()
            try:
                async with grpc.aio.insecure_channel(f"unix://{socket_path}") as ch:
                    stub = pb2_grpc.SemanticMemoryStub(ch)
                    stream = stub.AttachSession(pb2.AttachRequest(client_id=client_id))
                    async for event in stream:
                        if event.event_type == pb2.SESSION_EVENT_SHUTDOWN:
                            self._get_supervisor().registry.drop()
                            break
                        yield event
                        if stop.is_set():
                            return
            except grpc.aio.AioRpcError as exc:
                if exc.code() is not grpc.StatusCode.UNAVAILABLE:
                    raise
                self._get_supervisor().registry.drop()

    async def search_history(self, query: str, summaries_dir: Path, top_k: int) -> str:
        reply = await self._call(
            "SearchHistory",
            pb2.HistorySearchRequest(
                summaries_dir=str(summaries_dir), query=query, top_k=top_k
            ),
            SEARCH_TIMEOUT_S,
        )
        return reply.result.content

    async def index_history(self, summaries_dir: Path, clear_first: bool = False) -> Any:
        return await self._call(
            "IndexHistory",
            pb2.IndexHistoryRequest(
                summaries_dir=str(summaries_dir), clear_first=clear_first
            ),
            INDEX_TIMEOUT_S,
        )

    # -- introspection -----------------------------------------------------

    async def index_status(self, kind: int, path: Path) -> Any:
        return await self._call(
            "IndexStatus",
            pb2.IndexStatusRequest(kind=kind, path=str(path)),
            STATUS_TIMEOUT_S,
        )

    async def list_tenants(self, kind: int, limit: int = _TENANTS_SHOWN) -> Any:
        return await self._call(
            "ListTenants",
            pb2.ListTenantsRequest(kind=kind, limit=limit),
            STATUS_TIMEOUT_S,
        )


# ---------------------------------------------------------------------------
# Error rendering (§5.1)
# ---------------------------------------------------------------------------


def render_no_project_index(
    resolved_dir: Path, reply: Any, *, explicit: bool = False
) -> str:
    """Explain a missing index and name the targets that do have one.

    "No results" would be a lie here — nothing was searched. The message
    has to carry the way forward, and the way forward is useless without
    names: "pass repo_path explicitly" is not actionable if the agent has
    no idea which paths would work.

    The two cases need different words. When the target was defaulted, the
    news is that the agent's *own* directory is unindexed and the remedy
    is to name another one. When the agent already named a repository,
    saying "this is your working directory" is simply false, and telling
    it to pass ``repo_path`` is advice it has just followed — so the
    remedy there is to pick a different path from the list.

    That remedy only exists if the list does. Nothing guarantees this host
    has any index at all, so the explicit wording is chosen *after* the
    tenants are rendered: pointing at repositories "below" when nothing
    follows would be the same kind of lie this branch exists to fix.
    """
    listed = format_tenants(reply)
    if explicit:
        lines = [
            f"Project memory: no semantic index for '{resolved_dir}'.",
            "",
            "That repository has not been indexed.",
        ]
        lines += (
            ["Pick one of the indexed repositories below."]
            if listed
            else ["There are no indexed repositories on this host right now."]
        )
    else:
        lines = [
            f"Project memory: no semantic index for '{resolved_dir}'.",
            "",
            "This is the agent's working directory and it has not been indexed.",
            "You can still query any repository that HAS an index by passing it",
            'explicitly, e.g. project_memory(query=..., repo_path="<path-to-repo>").',
        ]
    if listed:
        lines += ["", f"Indexed repositories on this host: {listed}"]
    return "\n".join(lines)


def render_no_history_index(
    agent_label: str, reply: Any, *, explicit: bool = False
) -> str:
    """The history-side twin of :func:`render_no_project_index`."""
    listed = format_tenants(reply)
    if explicit:
        lines = [
            f"History memory: no history index for '{agent_label}'.",
            "",
            "That agent's history has not been indexed.",
        ]
        lines += (
            ["Pick one of the agents below."]
            if listed
            else ["No agent histories are indexed on this host right now."]
        )
    else:
        lines = [
            f"History memory: no history index for '{agent_label}'.",
            "",
            "This agent's own history has not been indexed.",
            "You can still query any agent that HAS an index by passing it",
            'explicitly, e.g. history_memory(query=..., agent_name="<name>").',
        ]
    if listed:
        lines += ["", f"Agents with an indexed history: {listed}"]
    return "\n".join(lines)


def format_tenants(reply: Any) -> str:
    """Render the tenant list, noting anything the page left out.

    ``total`` is the true count while ``tenants`` is one page, so a bare
    join would silently present a truncated list as the whole set —
    "these are your options" when it is really "some of them".
    """
    if reply is None:
        return ""
    names = list(getattr(reply, "tenants", []))
    if not names:
        return ""
    total = int(getattr(reply, "total", len(names)))
    listed = ", ".join(names)
    hidden = total - len(names)
    if hidden > 0:
        listed += f" … (+{hidden} more)"
    return listed


def index_history_sync(summaries_dir: Path, clear_first: bool = False) -> Any:
    async def invoke() -> Any:
        return await SemanticMemoryClient().index_history(summaries_dir, clear_first)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(invoke())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, invoke()).result()
