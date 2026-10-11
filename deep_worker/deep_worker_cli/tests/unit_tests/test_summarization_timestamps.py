"""Every summary message goes into the history with its own ``created_at``.

Owner requirement (2026-08-29): in the web UI history a summarization
block must not have an empty timestamp — "this is also important for
pseudo-summarization".

The web UI's dialog renderer prints a bare ``--- Summarization ---``
header when a message has no ``created_at``. That is the correct
degradation for old checkpoints, and it must not be fixed on the reader
side — otherwise the reader would start inventing a time. So the
invariant has to hold on the WRITER side: a summary message is born
already stamped.

All five summary-building paths (``before_model`` / ``abefore_model``,
requested summarization, pseudo-summarization and the empty-history
branch) converge on a single helper — ``_build_new_messages_with_path``.
The tests hit exactly that helper: it is the node every summary message
must pass through, so one check covers all entry points at once.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware.message_timestamps import CREATED_AT_KEY
from deepagents.middleware.summarization import SummarizationMiddleware


def _middleware() -> SummarizationMiddleware:
    """Middleware with a fake model: the summary-building helper never calls it.

    The model is needed only so the constructor can be assembled —
    ``_build_new_messages_with_path`` itself touches neither the model nor
    the backend, so the substitution masks nothing.
    """
    return SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="x")])),
        backend=None,
    )


def _only_message(built: list[Any]) -> HumanMessage:
    assert len(built) == 1, f"expected exactly one summary message, got {len(built)}"
    msg = built[0]
    assert isinstance(msg, HumanMessage)
    return msg


def _parse(raw: Any) -> datetime:
    assert isinstance(raw, str) and raw.strip(), f"created_at is empty: {raw!r}"
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None, f"created_at has no timezone: {raw!r}"
    return parsed


class TestSummaryMessageIsStamped:
    """A summary never goes into the history without a time — on any path."""

    def test_summary_with_jsonl_reference_is_stamped(self) -> None:
        """Regular summarization with the history dumped to JSONL."""
        built = _middleware()._build_new_messages_with_path(
            "some summary", "/tmp/agent.jsonl"
        )
        msg = _only_message(built)
        assert msg.additional_kwargs["lc_source"] == "summarization"
        _parse(msg.additional_kwargs.get(CREATED_AT_KEY))

    def test_summary_without_file_path_is_stamped(self) -> None:
        """Empty-history branch: there is no JSONL reference, the stamp is still there.

        A separate test, because this is a DIFFERENT content-building
        branch (``file_path is None``) — not a cosmetic variant of the
        first one. It is the branch ``apply_pseudo_summary`` calls when
        the filtered history is empty.
        """
        built = _middleware()._build_new_messages_with_path("summary", None)
        msg = _only_message(built)
        assert msg.additional_kwargs["lc_source"] == "summarization"
        _parse(msg.additional_kwargs.get(CREATED_AT_KEY))

    @pytest.mark.parametrize("file_path", ["/tmp/a.jsonl", None])
    def test_stamp_is_utc_and_sane(self, file_path: str | None) -> None:
        """The stamp is UTC and close to "now", not the epoch and not a placeholder.

        The renderer converts to the local zone, trusting that the input
        is UTC. A lying stamp reads as plausible — so we check the zone
        and the order of magnitude, not just "the field is not empty".
        """
        before = datetime.now(timezone.utc)
        built = _middleware()._build_new_messages_with_path("s", file_path)
        after = datetime.now(timezone.utc)

        stamped = _parse(_only_message(built).additional_kwargs.get(CREATED_AT_KEY))
        assert stamped.utcoffset() == timezone.utc.utcoffset(None)
        assert before <= stamped.astimezone(timezone.utc) <= after

    def test_later_summary_never_goes_back_in_time(self) -> None:
        """A second summary does not end up EARLIER than the first.

        Deliberately ``>=``, not ``>``: ``_build_new_messages_with_path``
        creates a fresh ``MonotonicUtcClock()`` on every call, and the
        +1 µs bump lives inside a single instance. The helper does not
        promise strict monotonicity across calls, and demanding it here
        would fail on correct code if two calls land in the same
        microsecond. What must really hold is that time does not go
        backwards: otherwise summaries in the UI would be ordered wrongly.
        """
        mw = _middleware()
        first = _parse(
            _only_message(mw._build_new_messages_with_path("1", None))
            .additional_kwargs.get(CREATED_AT_KEY)
        )
        second = _parse(
            _only_message(mw._build_new_messages_with_path("2", None))
            .additional_kwargs.get(CREATED_AT_KEY)
        )
        assert second >= first
