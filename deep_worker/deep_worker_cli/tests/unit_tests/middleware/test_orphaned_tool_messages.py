"""Regression: a ToolMessage whose AIMessage.tool_calls is gone must not
reach the provider.

Incident (repo_graph ingest, 2026-09-23): batches failed with

    400 — Messages with role 'tool' must be a response to a preceding
    message with 'tool_calls'

and were classified ``agent_error`` (permanent), losing the batch — though by
nature this is a protocol race, not a permanent fault.

Mechanism. A ``--sum-query`` run wipes the entire message channel with
``RemoveMessage(REMOVE_ALL_MESSAGES)`` (agent_process.py:2529). The wipe is
atomic over what it can SEE, so an ``AIMessage`` and its ``ToolMessage`` go
together. But a tool-call still IN FLIGHT when the wipe lands resolves
afterwards, and ``add_messages`` appends its ``ToolMessage`` onto the
freshly-installed summary state — parent already erased. The next request
therefore carries a ``ToolMessage`` answering nothing.

Note this is the MIRROR of Bug B. ``_rewrite_adjacent_closers`` only ever
walks ``AIMessage.tool_calls``, so an orphan with no parent is invisible to
it — the pre-fix middleware returned ``None`` for exactly the state that
produces the 400. Exposure is shared by every ``sum_query`` opener (linker,
cross_file_update_runner, extractor), which is why the repair lives in the
one entry-node middleware rather than in any consumer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Overwrite

from deep_worker_cli.middleware.dangling_tool_calls import (
    DanglingToolCallMiddleware,
)


def _invoke(mw: DanglingToolCallMiddleware, msgs: list) -> Any:
    return mw.before_model({"messages": msgs})


def _result_messages(res: Any) -> list:
    assert res is not None, "expected the middleware to rewrite the history"
    payload = res["messages"]
    assert isinstance(payload, Overwrite), (
        "must Overwrite: appending would leave the orphan in state"
    )
    return payload.value


def _summary_shape() -> HumanMessage:
    """What the sum_query wipe installs in place of the whole history."""
    return HumanMessage(content="Here is a summary of the conversation to date")


class TestOrphanedToolMessages:
    def test_orphan_after_a_wipe_is_dropped(self) -> None:
        """The exact production state: summary + a late tool result."""
        mw = DanglingToolCallMiddleware()
        msgs = [
            _summary_shape(),
            ToolMessage(
                content='{"accepted": 6}',
                tool_call_id="call_submit_links",
                name="submit_links",
            ),
        ]

        out = _result_messages(_invoke(mw, msgs))

        assert not any(isinstance(m, ToolMessage) for m in out), (
            "an orphaned ToolMessage must not survive — it is what the "
            "provider rejects with the 400"
        )
        assert len(out) == 1 and isinstance(out[0], HumanMessage)

    def test_a_paired_tool_message_is_kept(self) -> None:
        """NEGATIVE CONTROL: the sweep must not eat legitimate results.

        Without this, a middleware that dropped EVERY ToolMessage would pass
        the test above while destroying normal tool use.
        """
        mw = DanglingToolCallMiddleware()
        msgs = [
            HumanMessage(content="do it"),
            AIMessage(
                content="",
                tool_calls=[{"name": "submit_links", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="ok", tool_call_id="c1", name="submit_links"),
        ]

        assert _invoke(mw, msgs) is None, (
            "a correctly paired history needs no repair"
        )

    def test_only_the_orphan_is_dropped(self) -> None:
        """A mixed history keeps the paired result and loses only the orphan."""
        mw = DanglingToolCallMiddleware()
        msgs = [
            _summary_shape(),
            ToolMessage(content="stale", tool_call_id="gone", name="submit_links"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {}, "id": "live"}],
            ),
            ToolMessage(content="body", tool_call_id="live", name="read_file"),
        ]

        out = _result_messages(_invoke(mw, msgs))

        kept = [m.tool_call_id for m in out if isinstance(m, ToolMessage)]
        assert kept == ["live"], f"expected only the paired result, got {kept}"

    def test_orphan_and_dangling_call_are_both_repaired(self) -> None:
        """Both directions in one history — each gets its own treatment.

        The orphan is DROPPED (nothing can answer for it); the dangling call
        is CLOSED with a warning closer (the existing Bug-B behaviour).
        """
        mw = DanglingToolCallMiddleware()
        msgs = [
            _summary_shape(),
            ToolMessage(content="stale", tool_call_id="gone", name="submit_links"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {}, "id": "unanswered"}],
            ),
        ]

        out = _result_messages(_invoke(mw, msgs))

        tool_ids = [m.tool_call_id for m in out if isinstance(m, ToolMessage)]
        assert tool_ids == ["unanswered"], (
            "the orphan must go and the dangling call must gain a closer"
        )

    def test_dropping_is_idempotent(self) -> None:
        """Re-feeding the repaired history must not churn further."""
        mw = DanglingToolCallMiddleware()
        msgs = [
            _summary_shape(),
            ToolMessage(content="stale", tool_call_id="gone", name="submit_links"),
        ]

        out = _result_messages(_invoke(mw, msgs))

        assert _invoke(mw, out) is None, "second pass must be a no-op"
