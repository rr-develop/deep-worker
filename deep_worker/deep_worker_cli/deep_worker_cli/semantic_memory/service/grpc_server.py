"""The gRPC servicer: dispatch, isolation, and the executor boundary.

This module owns the two invariants that keep the service from
deadlocking itself, both inherited from incidents rather than theory.

**Every engine call goes to a dedicated executor** (design §4.2, blocker
B2). LanceDB bridges to its own background event loop through a blocking
``run_coroutine_threadsafe(...).result()``. On a loop thread that call
holds the GIL while it waits, LanceDB's loop can never acquire the GIL to
finish the work being waited on, and the process hangs forever — the
agent_c deadlock of 2026-06-15. Moving the code into a service does not
change that physics, it only changes which process hangs. So the rule
covers ``SearchProject`` and ``SearchHistory`` as much as
``IndexHistory``: searching loads the same model and opens the same
database as indexing does.

**One tenant is served one call at a time.** A LanceDB index is a set of
files owned by a single process; a write concurrent with a read on the
same tenant corrupts or blocks. Each tenant therefore gets an
``asyncio.Lock``, taken before the work is handed to the executor.
Different tenants stay fully parallel — the lock is per tenant, never
global, because a global one would serialize ten repositories behind one
slow reindex.

The two rules compose in one order that matters: acquire the lock on the
loop, then offload. Doing it the other way — offloading a body that then
blocks on the lock — would park an executor thread on a lock that only
the loop can release, and the executor pool would drain.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

import grpc

from deep_worker_cli.semantic_memory.service import engine
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb2
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
SERVICE_VERSION = "0.1.0"

# One worker per concurrent engine call we are willing to run. The model
# is shared and the per-tenant lock already serializes same-tenant work,
# so this bounds cross-tenant parallelism — not a throughput knob to
# raise casually: each concurrent search holds its own torch buffers.
_EXECUTOR_WORKERS = 4

#: Idle time with no attached clients before the service stops itself
#: (policy §3.12). 884 MB is too much to keep resident for an agent that
#: went away hours ago.
_IDLE_SHUTDOWN_S = 30 * 60

#: Server-pushed keepalive cadence on the AttachSession stream. Must stay
#: under gRPC's default 30 s http/2 idle timeout or the stream is torn
#: down and every client looks detached. The LSP service uses 25 s.
_KEEPALIVE_INTERVAL_S = 25.0

#: ``ListTenants`` page size when the request leaves ``limit`` unset.
_DEFAULT_TENANT_LIMIT = 10

_T = TypeVar("_T")


class SemanticMemoryServicer(pb2_grpc.SemanticMemoryServicer):
    """Serves every tenant on this host from one loaded model."""

    def __init__(
        self,
        transport_address: str = "",
        idle_shutdown_s: float = _IDLE_SHUTDOWN_S,
        keepalive_interval_s: float = _KEEPALIVE_INTERVAL_S,
    ) -> None:
        self._transport_address = transport_address
        self._idle_shutdown_s = idle_shutdown_s
        self._keepalive_interval_s = keepalive_interval_s
        # Set whenever a client attaches or detaches. The watchdog blocks
        # on it instead of ticking — the project forbids polling, and a
        # periodic wakeup would burn CPU for the entire idle window.
        self._activity = asyncio.Event()
        #: Outbound queues, one per attached AttachSession stream.
        self._sessions: set[asyncio.Queue[pb2.SessionEvent]] = set()
        self._idle_since: float | None = time.time()
        self._stop_requested = asyncio.Event()
        self._started_at = time.time()
        self._executor = ThreadPoolExecutor(
            max_workers=_EXECUTOR_WORKERS,
            thread_name_prefix="semantic-memory",
        )
        # defaultdict is safe here only because every mutation happens on
        # the event-loop thread: handlers touch it before offloading, and
        # asyncio gives us the mutual exclusion a plain dict would need a
        # lock for.
        self._tenant_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._model_loaded = False
        self._searches_total = 0
        self._index_ops_total = 0
        self._last_activity = self._started_at
        self._attached_clients = 0
        self._warm_tenants: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    async def idle_watchdog(self) -> None:
        """Stop the service once it has been idle long enough.

        Event-driven, never periodic: the task blocks on ``_activity``
        and wakes only on a real attach/detach or on the single deadline
        that matters. Polling is forbidden by project policy, and here it
        would also mean waking up hundreds of times across a half-hour
        window to learn nothing.
        """
        try:
            while True:
                if self._idle_since is None:
                    # Someone is attached — block until a detach.
                    self._activity.clear()
                    await self._activity.wait()
                    continue
                remaining = self._idle_shutdown_s - (time.time() - self._idle_since)
                if remaining <= 0:
                    logger.info(
                        "idle for %.0fs with no attached clients → shutting down",
                        self._idle_shutdown_s,
                    )
                    self._broadcast_shutdown()
                    self._stop_requested.set()
                    return
                self._activity.clear()
                try:
                    await asyncio.wait_for(self._activity.wait(), timeout=remaining)
                except TimeoutError:
                    pass  # deadline reached; re-evaluate at the top
        except asyncio.CancelledError:
            return

    async def wait_for_stop(self) -> None:
        await self._stop_requested.wait()

    def _broadcast_shutdown(self) -> None:
        """Tell attached clients we are going away.

        They would find out anyway when the stream breaks, but a broken
        stream is ambiguous — crash, restart, or planned exit all look
        alike. Saying so explicitly lets a client treat the next
        UNAVAILABLE as expected and re-spawn without alarm (§11.9).
        """
        event = pb2.SessionEvent(
            event_type=pb2.SESSION_EVENT_SHUTDOWN,
            timestamp_ns=time.time_ns(),
        )
        for queue in self._sessions:
            queue.put_nowait(event)

    async def _ensure_model(self) -> None:
        """Resolve the provider on first use.

        Loading is lazy by design (§4.1): an agent that never calls the
        tool must not pay ~7 s and 884 MB for it, so nothing here happens
        at startup.

        ``model_loaded`` means "a request has reached the engine", not
        "the weights are resident": the provider defers the actual load
        to the first ``search``, which happens on the executor just
        after. The flag therefore flips slightly early. It is an operator
        signal — has this service done any real work yet — and precise
        enough for that; making it exact would mean loading the weights
        eagerly here, which is the cost the laziness exists to avoid.
        """
        if self._model_loaded:
            return
        await self._offload(engine.load_model)
        self._model_loaded = True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- executor boundary -------------------------------------------------

    async def _offload(self, fn: Callable[..., _T], *args: Any) -> _T:
        """Run a blocking engine call off the event loop.

        The single chokepoint for blocker B2 — every engine entry point
        reaches the model through here, so ``test_no_lancedb_on_loop`` has
        exactly one thing to pin.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args))

    async def _run_for_tenant(
        self, tenant: str, fn: Callable[..., _T], *args: Any
    ) -> _T:
        """Serialize per tenant, then offload. Order is load-bearing."""
        async with self._tenant_locks[tenant]:
            self._warm_tenants.add(tenant)
            return await self._offload(fn, *args)

    def _touch(self) -> None:
        self._last_activity = time.time()

    # -- admin -------------------------------------------------------------

    async def Health(  # noqa: N802 — gRPC generated name
        self, request: pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> pb2.HealthReply:
        return pb2.HealthReply(
            ready=True,
            protocol_version=PROTOCOL_VERSION,
            service_version=SERVICE_VERSION,
            uptime_seconds=int(time.time() - self._started_at),
            attached_clients=self._attached_clients,
            model_loaded=self._model_loaded,
        )

    async def Status(  # noqa: N802
        self, request: pb2.StatusRequest, context: grpc.aio.ServicerContext
    ) -> pb2.StatusReply:
        return pb2.StatusReply(
            pid=os.getpid(),
            transport_address=self._transport_address,
            uptime_seconds=int(time.time() - self._started_at),
            attached_clients=self._attached_clients,
            model_loaded=self._model_loaded,
            embedding_model=engine.embedding_model_name(),
            searches_total=self._searches_total,
            index_ops_total=self._index_ops_total,
            last_activity_unix=int(self._last_activity),
            warm_tenants=sorted(self._warm_tenants),
        )

    async def AttachSession(  # noqa: N802
        self, request: pb2.AttachRequest, context: grpc.aio.ServicerContext
    ) -> Any:
        """Keep a client attached until it goes away.

        The stream is the keepalive channel for idle-shutdown: while any
        client holds it open the watchdog's countdown is suspended. The
        server pushes the frames, so the client never polls — it just
        reads. The cadence stays under gRPC's 30 s http/2 idle timeout,
        which would otherwise tear the stream down and make every client
        look detached.
        """
        self._attach()
        queue = self._register_session()
        try:
            yield pb2.SessionEvent(
                event_type=pb2.SESSION_EVENT_ATTACHED,
                message=request.client_id,
                timestamp_ns=time.time_ns(),
            )
            while True:
                yield await self._next_session_event(queue)
        finally:
            self._unregister_session(queue)
            self._detach()

    def _register_session(self) -> asyncio.Queue[pb2.SessionEvent]:
        queue: asyncio.Queue[pb2.SessionEvent] = asyncio.Queue()
        self._sessions.add(queue)
        return queue

    def _unregister_session(self, queue: asyncio.Queue[pb2.SessionEvent]) -> None:
        self._sessions.discard(queue)

    async def _next_session_event(
        self, queue: asyncio.Queue[pb2.SessionEvent]
    ) -> pb2.SessionEvent:
        """Next event for this stream, or a keepalive if nothing happens.

        Blocks on the queue rather than sleeping between ticks: a real
        event (a shutdown notice) goes out the instant it is queued
        instead of waiting out the remainder of an interval, and an idle
        stream costs one parked coroutine rather than a wakeup every few
        seconds. The timeout is a floor on traffic, not a poll — it only
        fires when there was genuinely nothing to send.
        """
        try:
            return await asyncio.wait_for(queue.get(), self._keepalive_interval_s)
        except TimeoutError:
            return pb2.SessionEvent(
                event_type=pb2.SESSION_EVENT_KEEPALIVE,
                timestamp_ns=time.time_ns(),
            )

    def _attach(self) -> None:
        self._attached_clients += 1
        self._idle_since = None
        self._activity.set()
        self._touch()

    def _detach(self) -> None:
        self._attached_clients = max(0, self._attached_clients - 1)
        if self._attached_clients == 0:
            # Start the countdown from the moment the LAST client left.
            self._idle_since = time.time()
        self._activity.set()

    # -- search ------------------------------------------------------------

    async def SearchProject(  # noqa: N802
        self, request: pb2.ProjectSearchRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ProjectSearchReply:
        repo_path = await _require_path(request.repo_path, "repo_path", context)
        self._searches_total += 1
        self._touch()
        await self._ensure_model()
        rendered = await self._run_for_tenant(
            _tenant_key("project", repo_path),
            engine.search_project,
            request.query,
            repo_path,
            request.top_k or 5,
        )
        return pb2.ProjectSearchReply(result=_ok(rendered))

    async def SearchHistory(  # noqa: N802
        self, request: pb2.HistorySearchRequest, context: grpc.aio.ServicerContext
    ) -> pb2.HistorySearchReply:
        summaries_dir = await _require_path(request.summaries_dir, "summaries_dir", context)
        self._searches_total += 1
        self._touch()
        await self._ensure_model()
        rendered = await self._run_for_tenant(
            _tenant_key("history", summaries_dir),
            engine.search_history,
            request.query,
            summaries_dir,
            request.top_k or 5,
        )
        return pb2.HistorySearchReply(result=_ok(rendered))

    # -- indexing ----------------------------------------------------------

    async def IndexHistory(  # noqa: N802
        self, request: pb2.IndexHistoryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.IndexHistoryReply:
        summaries_dir = await _require_path(request.summaries_dir, "summaries_dir", context)
        self._index_ops_total += 1
        self._touch()
        await self._ensure_model()
        rendered, files_indexed, chunks_added, lines_read, errors = await self._run_for_tenant(
            _tenant_key("history", summaries_dir),
            engine.index_history,
            summaries_dir,
            request.clear_first,
        )
        return pb2.IndexHistoryReply(
            result=_ok(rendered),
            files_indexed=files_indexed,
            chunks_added=chunks_added,
            lines_read=lines_read,
            errors=errors,
        )

    async def IndexStatus(  # noqa: N802
        self, request: pb2.IndexStatusRequest, context: grpc.aio.ServicerContext
    ) -> pb2.IndexStatusReply:
        path = await _require_path(request.path, "path", context)
        kind = await _require_kind(request.kind, context)

        if kind == pb2.INDEX_KIND_PROJECT:
            exists = await self._run_for_tenant(
                _tenant_key("project", path), engine.project_index_exists, path
            )
            return pb2.IndexStatusReply(exists=exists, complete=exists, pending_lines=0)

        # Under the tenant lock like everything else: ``pending_lines``
        # reads ``line_hashes.json``, which a concurrent IndexHistory on
        # the same tenant rewrites. Reading it unlocked would report a
        # count from a half-written file.
        tenant = _tenant_key("history", path)
        exists = await self._run_for_tenant(tenant, engine.history_index_exists, path)
        pending = await self._run_for_tenant(tenant, engine.history_pending_lines, path)
        return pb2.IndexStatusReply(
            exists=exists,
            complete=exists and pending == 0,
            pending_lines=pending,
        )

    async def ListTenants(  # noqa: N802
        self, request: pb2.ListTenantsRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ListTenantsReply:
        """Enumerate targets that actually carry an index.

        The client cannot do this itself: it would have to scan other
        agents' directories, which is exactly the knowledge the service
        exists to centralize (§16 A2). Only tenants WITH an index are
        listed — an empty directory is not a usable target and offering
        it as one would send the agent into a guaranteed error.
        """
        kind = await _require_kind(request.kind, context)
        tenants = await self._offload(engine.list_tenants, kind == pb2.INDEX_KIND_PROJECT)
        # 0 means "unset" on the wire, and proto3 cannot tell it from an
        # explicit 0 — so it maps to the server default, never to "all".
        # ``or`` would read 0 as falsy and return every tenant, the exact
        # opposite of the documented contract.
        limit = request.limit if request.limit > 0 else _DEFAULT_TENANT_LIMIT
        # ``total`` reports the true count even when the list is truncated,
        # so the caller can say "+N more" instead of implying it saw all.
        return pb2.ListTenantsReply(tenants=tenants[:limit], total=len(tenants))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(rendered: str) -> pb2.ToolResult:
    """Wrap rendered text.

    ``is_error`` describes the TRANSPORT, not the search. A miss, a
    missing index, or a failed query are all successful RPCs carrying
    text the agent should read; marking them as errors would make the
    client raise where it should simply relay (§4.1).
    """
    return pb2.ToolResult(content=rendered, is_error=False)


def _tenant_key(family: str, path: Path) -> str:
    """Lock key. Namespaced by family: a repo and a summaries dir could
    otherwise collide on an identical path and serialize each other."""
    return f"{family}:{path}"


async def _require_path(raw: str, field: str, context: grpc.aio.ServicerContext) -> Path:
    """Resolve a wire path, aborting on an empty one.

    The wire address is always explicit (§5): defaults are resolved in the
    client, where the agent's own working directory and name are known.
    An empty field means the client skipped that resolution, so failing
    loudly here surfaces the bug instead of silently serving the
    service's own cwd — someone else's repository.

    ``grpc.aio``'s ``abort`` is a COROUTINE, unlike its sync counterpart.
    Calling it without ``await`` builds a coroutine object, raises
    nothing, and lets the handler run on with invalid input — the
    validation silently evaporates. Hence ``async``, and hence the
    ``await`` at every call site.
    """
    if not raw:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, f"{field} must not be empty"
        )
    return Path(os.path.realpath(raw))


async def _require_kind(kind: int, context: grpc.aio.ServicerContext) -> int:
    """Reject the proto3 zero value instead of guessing.

    ``INDEX_KIND_UNKNOWN`` exists because proto3 mandates a zero default,
    not because it is a meaningful request. Guessing the family from the
    shape of the path would silently answer about the wrong index; an
    unset field is a client bug and says so.

    Async for the same reason as ``_require_path`` — see there.
    """
    if kind not in (pb2.INDEX_KIND_PROJECT, pb2.INDEX_KIND_HISTORY):
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "kind must be INDEX_KIND_PROJECT or INDEX_KIND_HISTORY",
        )
    return kind


async def serve(socket_path: Path, idle_shutdown_s: float = _IDLE_SHUTDOWN_S) -> None:
    """Run the service on a UDS until it goes idle or is terminated."""
    # The socket is created by gRPC, which honours the ambient umask —
    # inherited from whichever agent happened to spawn us. Narrow it
    # ourselves so a permissive agent umask cannot widen the socket
    # (design §10).
    os.umask(0o077)

    servicer = SemanticMemoryServicer(
        transport_address=f"unix://{socket_path}",
        idle_shutdown_s=idle_shutdown_s,
    )
    server = grpc.aio.server()
    pb2_grpc.add_SemanticMemoryServicer_to_server(servicer, server)
    server.add_insecure_port(f"unix://{socket_path}")
    await server.start()

    watchdog = asyncio.create_task(servicer.idle_watchdog())
    try:
        await servicer.wait_for_stop()
        await server.stop(grace=1.0)
    finally:
        watchdog.cancel()
        servicer.shutdown()
        # The socket file outlives the process that bound it; leaving it
        # behind makes the next supervisor see a socket with nobody
        # listening (registry.is_stale covers the pid, not the file).
        socket_path.unlink(missing_ok=True)
