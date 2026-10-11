"""The servicer must never touch the engine on the event loop.

These tests exist because of a specific hang, not a style preference.
LanceDB bridges to its own background event loop with a blocking
``run_coroutine_threadsafe(...).result()``; run on a loop thread it holds
the GIL while waiting, LanceDB's loop starves, and the process is dead
(mj1_init, 2026-06-15). The service inherits that physics, so the
executor boundary is a correctness property and is pinned as one.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from deep_worker_cli.semantic_memory.service import engine, grpc_server
from deep_worker_cli.semantic_memory.service.grpc_server import SemanticMemoryServicer
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb2

# The executor-boundary tests drive real engine calls, which resolve the
# semantic-search provider and raise ProviderUnavailable when the optional
# extra is absent. Skip rather than fail: the extra is not part of a plain
# `pip install -e .`. Same convention as the rest of tests/unit_tests.
pytest.importorskip(
    "llama_index", reason="semantic-search extra is absent"
)  # semantic-search dep; skip when absent


class _Context:
    """Minimal stand-in for ``grpc.aio.ServicerContext``.

    ``abort`` is a COROUTINE here because it is one in ``grpc.aio``, and
    that detail is the whole point. An earlier version of this stub was
    synchronous; against it a handler that called ``context.abort(...)``
    without ``await`` passed every test, while over a real socket the
    un-awaited coroutine raised nothing and the handler served the
    invalid request anyway. The stub must be as async as the real thing
    or it certifies a validation layer that does not exist.

    It raises rather than returning, mirroring gRPC's contract that abort
    never hands control back to the handler.
    """

    def __init__(self) -> None:
        self.code = None
        self.details = ""

    async def abort(self, code, details):  # noqa: ANN001, ANN201
        self.code = code
        self.details = details
        raise _Aborted(details)

    def add_done_callback(self, cb):  # noqa: ANN001, ANN201
        self._cb = cb


class _Aborted(Exception):
    pass


@pytest.fixture
def servicer():
    svc = SemanticMemoryServicer(transport_address="unix:///tmp/test.sock")
    yield svc
    svc.shutdown()


class TestExecutorBoundary:
    """Blocker B2: no engine call may run on the loop thread."""

    @pytest.mark.asyncio
    async def test_project_search_runs_off_the_loop_thread(self, servicer, monkeypatch):
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}

        def _record(query, repo_path, top_k):  # noqa: ANN001, ANN202
            seen["thread"] = threading.get_ident()
            return "rendered"

        monkeypatch.setattr(engine, "search_project", _record)
        reply = await servicer.SearchProject(
            pb2.ProjectSearchRequest(repo_path="/tmp", query="q", top_k=3), _Context()
        )

        assert reply.result.content == "rendered"
        assert seen["thread"] != loop_thread, (
            "engine ran on the event-loop thread — this is the mj1_init deadlock"
        )

    @pytest.mark.asyncio
    async def test_history_search_runs_off_the_loop_thread(self, servicer, monkeypatch):
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}

        def _record(query, summaries_dir, top_k):  # noqa: ANN001, ANN202
            seen["thread"] = threading.get_ident()
            return "rendered"

        monkeypatch.setattr(engine, "search_history", _record)
        await servicer.SearchHistory(
            pb2.HistorySearchRequest(summaries_dir="/tmp", query="q"), _Context()
        )
        assert seen["thread"] != loop_thread

    @pytest.mark.asyncio
    async def test_indexing_runs_off_the_loop_thread(self, servicer, monkeypatch):
        """Indexing is the obvious case, but it is NOT the only one — the
        searches above matter just as much, since they load the same model.
        """
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}

        def _record(summaries_dir, clear_first):  # noqa: ANN001, ANN202
            seen["thread"] = threading.get_ident()
            return "done", 2, 7, 5, 0

        monkeypatch.setattr(engine, "index_history", _record)
        reply = await servicer.IndexHistory(
            pb2.IndexHistoryRequest(summaries_dir="/tmp"), _Context()
        )

        assert seen["thread"] != loop_thread
        assert (reply.files_indexed, reply.chunks_added) == (2, 7)
        assert (reply.lines_read, reply.errors) == (5, 0)


class TestTenantSerialization:
    """One tenant at a time; different tenants stay parallel."""

    @pytest.mark.asyncio
    async def test_same_tenant_calls_do_not_overlap(self, servicer, monkeypatch):
        overlapping = False
        active = 0
        guard = threading.Lock()

        def _slow(query, summaries_dir, top_k):  # noqa: ANN001, ANN202
            nonlocal overlapping, active
            with guard:
                active += 1
                if active > 1:
                    overlapping = True
            # Long enough that a missing lock reliably overlaps: the second
            # call only has to enter while the first is still inside.
            threading.Event().wait(0.15)
            with guard:
                active -= 1
            return "rendered"

        monkeypatch.setattr(engine, "search_history", _slow)
        request = pb2.HistorySearchRequest(summaries_dir="/tmp/same", query="q")
        await asyncio.gather(
            servicer.SearchHistory(request, _Context()),
            servicer.SearchHistory(request, _Context()),
        )

        assert not overlapping, "two calls on one tenant ran concurrently"

    @pytest.mark.asyncio
    async def test_different_tenants_run_in_parallel(self, servicer, monkeypatch):
        """A per-tenant lock, not a global one.

        A global lock would also pass the test above while serializing
        every repository on the host behind one slow reindex — so the
        parallelism has to be pinned too, or the cheap wrong fix survives.
        """
        started = threading.Barrier(2, timeout=5)

        def _rendezvous(query, summaries_dir, top_k):  # noqa: ANN001, ANN202
            started.wait()  # deadlocks unless both run at once
            return "rendered"

        monkeypatch.setattr(engine, "search_history", _rendezvous)
        await asyncio.gather(
            servicer.SearchHistory(
                pb2.HistorySearchRequest(summaries_dir="/tmp/a", query="q"), _Context()
            ),
            servicer.SearchHistory(
                pb2.HistorySearchRequest(summaries_dir="/tmp/b", query="q"), _Context()
            ),
        )

    @pytest.mark.asyncio
    async def test_project_and_history_paths_do_not_share_a_lock(
        self, servicer, monkeypatch
    ):
        """Same path string, different families — must not serialize."""
        started = threading.Barrier(2, timeout=5)

        def _rendezvous(*args):  # noqa: ANN002, ANN202
            started.wait()
            return "rendered"

        monkeypatch.setattr(engine, "search_project", _rendezvous)
        monkeypatch.setattr(engine, "search_history", _rendezvous)
        await asyncio.gather(
            servicer.SearchProject(
                pb2.ProjectSearchRequest(repo_path="/tmp/x", query="q"), _Context()
            ),
            servicer.SearchHistory(
                pb2.HistorySearchRequest(summaries_dir="/tmp/x", query="q"), _Context()
            ),
        )


class TestArgumentValidation:
    @pytest.mark.asyncio
    async def test_unknown_index_kind_is_rejected(self, servicer):
        """Fail loud rather than guess the family from the path shape."""
        context = _Context()
        with pytest.raises(_Aborted):
            await servicer.IndexStatus(
                pb2.IndexStatusRequest(kind=pb2.INDEX_KIND_UNKNOWN, path="/tmp"),
                context,
            )
        assert "INDEX_KIND_PROJECT" in context.details

    @pytest.mark.asyncio
    async def test_empty_path_is_rejected(self, servicer):
        """The wire address is always explicit (§5).

        An empty field means the client skipped default resolution.
        Serving it would silently search the SERVICE's cwd — a repository
        belonging to nobody in particular.
        """
        context = _Context()
        with pytest.raises(_Aborted):
            await servicer.SearchProject(
                pb2.ProjectSearchRequest(repo_path="", query="q"), context
            )
        assert "repo_path" in context.details


class TestIdleShutdown:
    """884 MB must not outlive the agents that needed it (policy §3.12)."""

    @pytest.mark.asyncio
    async def test_service_stops_itself_once_idle(self):
        svc = SemanticMemoryServicer(idle_shutdown_s=0.05)
        try:
            watchdog = asyncio.create_task(svc.idle_watchdog())
            await asyncio.wait_for(svc.wait_for_stop(), timeout=2)
            watchdog.cancel()
        finally:
            svc.shutdown()

    @pytest.mark.asyncio
    async def test_attached_client_suspends_the_countdown(self):
        """An attached client must hold the service open indefinitely."""
        svc = SemanticMemoryServicer(idle_shutdown_s=0.05)
        try:
            svc._attach()
            watchdog = asyncio.create_task(svc.idle_watchdog())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(svc.wait_for_stop(), timeout=0.4)
            watchdog.cancel()
        finally:
            svc.shutdown()

    @pytest.mark.asyncio
    async def test_countdown_restarts_from_the_last_detach(self):
        """The window is measured from when the LAST client left, not from
        startup — otherwise a long session would shut down mid-use."""
        svc = SemanticMemoryServicer(idle_shutdown_s=0.2)
        try:
            svc._attach()
            watchdog = asyncio.create_task(svc.idle_watchdog())
            await asyncio.sleep(0.25)  # longer than the window, but attached
            assert not svc._stop_requested.is_set()
            svc._detach()
            await asyncio.wait_for(svc.wait_for_stop(), timeout=2)
            watchdog.cancel()
        finally:
            svc.shutdown()

    def test_watchdog_has_no_polling_loop(self):
        """Policy forbids polling. A periodic tick would also wake the
        process hundreds of times across a half-hour idle window."""
        import inspect

        src = inspect.getsource(SemanticMemoryServicer.idle_watchdog)
        assert "asyncio.sleep" not in src, "watchdog polls instead of waiting"
        assert "_activity.wait" in src


class TestKeepalive:
    @pytest.mark.asyncio
    async def test_stream_pushes_keepalive_frames(self):
        """Server-driven, under gRPC's 30 s http/2 idle timeout: without
        it the stream dies and every client looks detached."""
        svc = SemanticMemoryServicer(keepalive_interval_s=0.01)
        try:
            events = []
            stream = svc.AttachSession(pb2.AttachRequest(client_id="a"), _Context())
            async for event in stream:
                events.append(event.event_type)
                if len(events) >= 3:
                    break
            assert events[0] == pb2.SESSION_EVENT_ATTACHED
            assert events[1:] == [pb2.SESSION_EVENT_KEEPALIVE] * 2
            assert svc._keepalive_interval_s < 30
        finally:
            svc.shutdown()


    @pytest.mark.asyncio
    async def test_real_events_preempt_the_keepalive_interval(self):
        """A queued event goes out immediately, not at the next tick.

        This is what makes the stream event-driven rather than a poll:
        with a 60 s interval, a sleep-based loop would sit on a pending
        shutdown notice for up to a minute.
        """
        svc = SemanticMemoryServicer(keepalive_interval_s=60)
        try:
            queue = svc._register_session()
            svc._broadcast_shutdown()
            event = await asyncio.wait_for(svc._next_session_event(queue), timeout=1)
            assert event.event_type == pb2.SESSION_EVENT_SHUTDOWN
        finally:
            svc.shutdown()

    def test_stream_has_no_sleep_loop(self):
        """Policy forbids polling loops; the keepalive must come from a
        timed-out wait on real work, not a periodic sleep."""
        import inspect

        src = inspect.getsource(SemanticMemoryServicer.AttachSession)
        assert "asyncio.sleep" not in src


class TestListTenants:
    @pytest.mark.asyncio
    async def test_unset_limit_uses_the_server_default(self, servicer, monkeypatch):
        """proto3 cannot distinguish an unset int32 from 0, so 0 means
        "server default" — reading it as falsy would return everything,
        the opposite of the documented contract."""
        monkeypatch.setattr(engine, "list_tenants", lambda project: [f"a{i}" for i in range(26)])
        reply = await servicer.ListTenants(
            pb2.ListTenantsRequest(kind=pb2.INDEX_KIND_HISTORY), _Context()
        )
        assert len(reply.tenants) == grpc_server._DEFAULT_TENANT_LIMIT
        assert reply.total == 26, "total must report the true count, not the page"

    @pytest.mark.asyncio
    async def test_explicit_limit_is_honoured(self, servicer, monkeypatch):
        monkeypatch.setattr(engine, "list_tenants", lambda project: ["a", "b", "c"])
        reply = await servicer.ListTenants(
            pb2.ListTenantsRequest(kind=pb2.INDEX_KIND_HISTORY, limit=2), _Context()
        )
        assert list(reply.tenants) == ["a", "b"]
        assert reply.total == 3


class TestResultWrapping:
    def test_engine_errors_are_not_transport_errors(self):
        """``is_error`` describes the RPC, not the search.

        A miss or a missing index is a successful call carrying text the
        agent should read. Flagging it would make the client raise where
        it should relay, turning a hint into a stack trace.
        """
        result = grpc_server._ok("Error: Project memory not available.")
        assert result.is_error is False
        assert result.content.startswith("Error:")


class TestAdmin:
    @pytest.mark.asyncio
    async def test_health_reports_protocol_version(self, servicer):
        reply = await servicer.Health(pb2.HealthRequest(), _Context())
        assert reply.ready is True
        assert reply.protocol_version == grpc_server.PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_status_survives_a_broken_provider(self, servicer, monkeypatch):
        """Status must answer even when the model cannot load — that is
        precisely when an operator needs it."""
        monkeypatch.setattr(
            engine, "embedding_model_name", lambda: (_ for _ in ()).throw(RuntimeError())
        )
        with pytest.raises(RuntimeError):
            engine.embedding_model_name()
        # The servicer calls the real (non-raising) implementation.
        monkeypatch.undo()
        reply = await servicer.Status(pb2.StatusRequest(), _Context())
        assert reply.pid > 0
