"""Stage-A dispatch timeout: the CLI announced a tool call but the control
request never arrived — no work was ever started, no job was parked.

Prior to this fix the two provider sites (claude / codex) reused
``build_in_progress_notice`` and promised the model
``wait_for_job(job_id="toolu_…")`` / ``kill_job``. The id could never
resolve (there is no parked job to key off), which is the very
``no background job with id '…'`` symptom the whole internal defect thread
is about. The new ``build_dispatch_timeout_notice`` states the honest truth:
the tool did not start, no job exists, retry.

See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.
"""

from __future__ import annotations

from deepagents.middleware.bounded_tool_call import (
    build_dispatch_timeout_notice,
    build_in_progress_notice,
)


# ── the honest dispatch-timeout notice ───────────────────────────────────


def test_notice_has_no_job_id_or_wait_kill_promise() -> None:
    text = build_dispatch_timeout_notice("execute", 120.0)
    lower = text.lower()
    assert "job_id" not in lower, (
        "stage-A dispatch timeout must NOT advertise a job_id — no job was "
        "parked, and any id would collide with the internal defect this "
        "series is fixing"
    )
    assert "wait_for_job" not in lower, text
    assert "kill_job" not in lower, text
    # Belt-and-braces: no ``job_N`` pattern either.
    import re

    assert re.search(r"\bjob_\d+\b", lower) is None, text


def test_notice_includes_tool_name_and_seconds() -> None:
    text = build_dispatch_timeout_notice("execute", 42.0)
    assert "execute" in text, "tool name must be surfaced"
    assert "42" in text, "wait bound must be surfaced so the model sees it"


def test_notice_marks_status_not_started_not_in_progress() -> None:
    text = build_dispatch_timeout_notice("execute", 120.0).lower()
    assert "not_started" in text, (
        "status must be not_started so the model treats this as a failure "
        "to start, not a backgrounded job"
    )
    assert "in_progress" not in text, text
    # Not an error either (POLICIES §7.1: timeout is never ``is_error``).
    assert "error" not in text, text


def test_notice_asks_the_model_to_retry() -> None:
    text = build_dispatch_timeout_notice("execute", 120.0).lower()
    assert "retry" in text, (
        "the model must be told what to do next — retrying the tool call "
        "is the correct recovery for a lost control request"
    )


# ── regression: real parks still get the honest job handle ───────────────


def test_in_progress_notice_still_advertises_job_id_and_actions() -> None:
    """``build_in_progress_notice`` is used by the middleware AFTER a real
    park (deep_worker/deepagents/middleware/bounded_tool_call.py:186,242).
    If this contract regresses we would silently break the very fix that
    6bd042b landed."""
    text = build_in_progress_notice("execute", "job_7", 120.0)
    assert "job_7" in text
    assert "wait_for_job" in text
    assert "kill_job" in text
    assert "in_progress" in text
