"""Mid-turn context compaction must be observable to the turn's caller.

Consumer requirement REQ-90: "summarization in the course of processing a
document is an incident, not normal operation". The consumer cannot
detect it on its own — only deep_worker knows a compaction happened — and the
interrupted turn arrives looking like an ordinary success: ``success: True``,
channel intact, nothing raised. Accepting it as ordinary means filing a
half-extracted document as "done", and nothing ever revisits it.

The contract agreed with the consumer:

  * ``context_compacted: True`` at the TOP level of the invoke reply, beside
    ``success``/``recovered`` — NOT inside ``response``, whose ``content`` the
    consumer reads as text.
  * ``context_summarized`` carries the detail (occurred/observable/reason/
    tokens/messages/history_path/at_utc), ``at_utc`` in UTC with a ``Z``.
  * An ABSENT key means "no compaction". Only an explicit flag counts: a report
    on every turn would fail every document of every run.

The two paths that MUST stay silent are the point of most of these tests:

  * pseudo-summarization (``--sum-query``) is the consumer's own per-document
    context reset — the NORMAL mode, and the first half of REQ-90. A flag here
    would fail EVERY document of EVERY run (the consumer named this the main risk
    of this change).
  * ``apply_requested_summary`` (``force=True``) is an explicit operator
    command with its own report (``agent_process.py``
    ``request_summarization``), not a size incident.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    _describe_trigger,
    _utc_now_iso,
)
from deepagents.backends.protocol import BackendProtocol
from deep_worker_cli.process.agent_process import AgentProcess
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers: build the REAL middleware, bypassing its model/backend wiring.
# A hand-rolled stand-in would prove nothing about production (a self-made stub
# shaped "close enough" invents its own conclusions), so these tests drive the
# real class and only replace the collaborators a unit test cannot have.
# ---------------------------------------------------------------------------
def _middleware(**over: Any) -> SummarizationMiddleware:
    sm = object.__new__(SummarizationMiddleware)
    sm.last_turn_compaction = None
    sm.trigger = over.get("trigger", ("fraction", 0.85))
    sm._cached_token_signature = None
    sm._cached_token_total = 0
    return sm


def _history(n: int) -> list[Any]:
    out: list[Any] = []
    for i in range(n):
        out.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
        out.append(AIMessage(content=f"a{i}", id=f"a{i}"))
    return out


def _pseudo_ready_middleware() -> tuple[SummarizationMiddleware, list[str]]:
    """A real middleware able to run ``apply_pseudo_summary`` end to end.

    Only the filesystem is faked. The offload helper itself is NOT stubbed:
    ``_aoffload_to_backend`` is shared with the size path, so a stamp placed in
    it would mark pseudo-summarization too, and a test that replaced the helper
    could never see that.
    """
    written: list[str] = []

    class _Written:
        error = None

    class _Backend:
        """Implements only what the REAL offload calls: ``adownload_files``
        (miss → fresh file) and ``awrite``. Shape taken from the helper, not
        invented."""

        async def adownload_files(self, paths: Any) -> Any:
            raise FileNotFoundError("no history yet")

        async def awrite(self, path: str, content: str) -> Any:
            written.append(path)
            return _Written()

        async def aedit(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
            raise AssertionError("fresh file must take the awrite path")

    # ``apply_pseudo_summary`` rejects anything that is not a BackendProtocol
    # (Gap-4: factory backends need a ToolRuntime). BackendProtocol is an ABC,
    # so register rather than subclass — subclassing would drag in every
    # abstract method the offload never calls.
    BackendProtocol.register(_Backend)

    sm = _middleware()
    sm._backend = _Backend()
    sm._on_summarized = None
    sm._history_path_prefix = "/summaries"
    # The real offload stamps provenance, which interrogates the model. A bare
    # object is enough: the helpers fall back when nothing is recognisable.
    sm.model = object()
    sm._hot_cache_ttl_overrides = None
    return sm, written


class _Result:
    """Stands in for ``RequestedSummaryResult`` (a frozen dataclass)."""

    def __init__(self, summarized: bool, replacement: list[Any] | None, path: str | None) -> None:
        self.summarized = summarized
        self.replacement_messages = replacement
        self.history_path = path


def _process() -> AgentProcess:
    return object.__new__(AgentProcess)


# ---------------------------------------------------------------------------
# The real size-based path stamps a report.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_size_trigger_during_turn_is_recorded() -> None:
    sm = _middleware()
    incoming = _history(40)
    tail = _history(2)

    async def _compaction(messages, runtime, **kw):
        assert kw["force"] is False, "the size path must not force"
        return _Result(True, tail, "/summaries/agent.jsonl")

    sm._arun_compaction = _compaction  # type: ignore[method-assign]
    sm._cached_total_tokens = lambda msgs: 1000 * len(msgs)  # type: ignore[method-assign]

    await SummarizationMiddleware.abefore_model(sm, {"messages": incoming}, None)

    report = sm.last_turn_compaction
    assert report is not None, "a real size-based compaction must be observable"
    assert report["occurred"] is True
    assert report["observable"] is True
    assert report["reason"] == "size_trigger"
    assert report["messages_before"] == len(incoming)
    assert report["messages_after"] == len(tail)
    assert report["tokens_before"] > report["tokens_after"]
    assert report["history_path"] == "/summaries/agent.jsonl"


@pytest.mark.asyncio
async def test_normal_turn_records_nothing() -> None:
    """No trigger → no report. Absence is the "nothing happened" signal."""
    sm = _middleware()

    async def _compaction(messages, runtime, **kw):
        return _Result(False, None, None)

    sm._arun_compaction = _compaction  # type: ignore[method-assign]
    sm._cached_total_tokens = lambda msgs: 10  # type: ignore[method-assign]

    await SummarizationMiddleware.abefore_model(sm, {"messages": _history(3)}, None)

    assert sm.last_turn_compaction is None


@pytest.mark.asyncio
async def test_arg_truncation_alone_is_not_a_compaction() -> None:
    """Truncation returns replacement messages WITHOUT summarizing.

    This is the shape that makes ``RemoveMessage(REMOVE_ALL_MESSAGES)`` in the
    returned update useless as a detector: the truncation-only branch emits the
    exact same shape. Only ``summarized`` separates them.
    """
    sm = _middleware()
    truncated = _history(5)

    async def _compaction(messages, runtime, **kw):
        return _Result(False, truncated, None)

    sm._arun_compaction = _compaction  # type: ignore[method-assign]
    sm._cached_total_tokens = lambda msgs: 10  # type: ignore[method-assign]

    out = await SummarizationMiddleware.abefore_model(sm, {"messages": _history(5)}, None)

    assert out is not None, "truncation still rewrites the channel"
    assert sm.last_turn_compaction is None, "truncation is not a compaction"


# ---------------------------------------------------------------------------
# THE MAIN RISK: the two silent paths.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pseudo_summary_leaves_no_report() -> None:
    """``apply_pseudo_summary`` is the operator's per-document reset.

    This is the consumer's NORMAL mode (one prompt = one document, history never
    accumulates) and is itself the first half of REQ-90. A report here would
    mark EVERY document of EVERY run as an incident — the consumer named this
    the main risk of the change, so the path is driven for real rather than
    inspected as text.

    Runs the real method on the real class: only the backend (which a unit
    test cannot have) is replaced, and it is replaced by the middleware's own
    ``BackendProtocol``, not by a hand-shaped look-alike.
    """
    sm, written = _pseudo_ready_middleware()

    new_messages, path = await SummarizationMiddleware.apply_pseudo_summary(
        sm, "agent", "EXTRACT THIS DOCUMENT", _history(20)
    )

    # The reset really happened: the REAL offload helper ran and wrote history.
    # That matters for what this test can catch — the offload helper is SHARED
    # with the size path (``_arun_compaction``), so a stamp placed there would
    # mark pseudo too. Stubbing the helper out would hide exactly that.
    assert written, "the real offload must have written history"
    assert path.endswith(".jsonl")
    assert len(new_messages) == 1
    assert new_messages[0].content == "EXTRACT THIS DOCUMENT", (
        "the sum-query must reach the model byte for byte"
    )
    assert sm.last_turn_compaction is None, (
        "pseudo-summarization is the normal per-document reset, not a size "
        "emergency — a report here fails every document of every run"
    )


@pytest.mark.asyncio
async def test_pseudo_summary_reply_carries_no_flag() -> None:
    """End of the wire: a turn whose only compaction was pseudo is clean.

    Pins what the consumer actually sees, not just the middleware field.
    """
    sm, _written = _pseudo_ready_middleware()

    await SummarizationMiddleware.apply_pseudo_summary(
        sm, "agent", "EXTRACT THIS DOCUMENT", _history(12)
    )

    proc = _process()
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]
    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert "context_compacted" not in reply
    assert "context_summarized" not in reply


def test_requested_summary_is_not_an_incident() -> None:
    """``force=True`` is an operator command, not a size incident.

    ``_arun_compaction`` is SHARED between the size path and the requested one,
    so the stamp deliberately lives in ``abefore_model`` rather than in the
    shared body. This test pins that placement: if the stamp migrated into
    ``_arun_compaction``, an explicit ``compact_history`` would start looking
    like an emergency.
    """
    inspect = __import__("inspect")
    shared = inspect.getsource(SummarizationMiddleware._arun_compaction)
    assert "last_turn_compaction" not in shared, (
        "the stamp must not live in the body shared with apply_requested_summary"
    )
    requested = inspect.getsource(SummarizationMiddleware.apply_requested_summary)
    assert "last_turn_compaction" not in requested


# ---------------------------------------------------------------------------
# agent_process: reset, read, and the flag on ALL reply sites.
# ---------------------------------------------------------------------------
def test_report_does_not_survive_into_the_next_turn() -> None:
    """One compaction must not mark every later document of the run."""
    proc = _process()
    sm = _middleware()
    sm.last_turn_compaction = {"occurred": True, "observable": True}
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    assert proc._turn_compaction_report() is not None
    proc._reset_turn_compaction_report()
    assert proc._turn_compaction_report() is None


def test_flag_rides_at_top_level_not_inside_response() -> None:
    proc = _process()
    sm = _middleware()
    sm.last_turn_compaction = {
        "occurred": True,
        "observable": True,
        "reason": "size_trigger",
        "at_utc": _utc_now_iso(),
    }
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert reply["context_compacted"] is True
    assert reply["context_summarized"]["occurred"] is True
    # The consumer reads response.content as text; nothing may be added there.
    assert set(reply["response"]) == {"role", "content"}
    assert reply["response"]["content"] == "DONE"


def test_normal_turn_carries_no_key_at_all() -> None:
    """Absence is the contract for "no compaction", not ``False``."""
    proc = _process()
    sm = _middleware()
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert "context_compacted" not in reply
    assert "context_summarized" not in reply


def test_missing_middleware_never_fails_a_turn() -> None:
    """Summarization can legitimately be absent; observability must not raise."""
    proc = _process()

    def _raise():
        raise RuntimeError("No SummarizationMiddleware found in the compiled agent graph")

    proc._find_summarization_middleware = _raise  # type: ignore[method-assign]

    proc._reset_turn_compaction_report()  # must not raise
    assert proc._turn_compaction_report() is None
    reply = proc._with_compaction_flag({"success": True, "response": {}})
    assert "context_compacted" not in reply


def test_every_success_reply_site_is_stamped() -> None:
    """A flag on one path out of three is an incident that reproduces every
    other run. ``_handle_invoke_owned`` has three success returns (plain,
    streaming, TaskManager); each must go through the wrapper.
    """
    inspect = __import__("inspect")
    body = inspect.getsource(AgentProcess._handle_invoke_owned)
    successes = len(re.findall(r'"success": True,', body))
    wrapped = body.count("_with_compaction_flag(")
    assert successes == wrapped == 3, (
        f"{successes} success replies but {wrapped} wrapped — every success "
        "reply site must carry the flag"
    )
    assert "_reset_turn_compaction_report()" in body, (
        "the turn must clear the previous turn's report before it can return"
    )


# ---------------------------------------------------------------------------
# Timestamp and trigger rendering.
# ---------------------------------------------------------------------------
def test_timestamp_is_utc_with_z() -> None:
    stamp = _utc_now_iso()
    assert stamp.endswith("Z"), f"UTC is a hard project rule: {stamp}"
    assert "+00:00" not in stamp
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", stamp), stamp


def test_trigger_is_rendered_for_the_operator() -> None:
    assert _describe_trigger(("fraction", 0.85)) == "fraction:0.85"
    assert _describe_trigger(("tokens", 170000)) == "tokens:170000"
    assert _describe_trigger([("fraction", 0.85), ("tokens", 170000)]) == (
        "fraction:0.85,tokens:170000"
    )


def test_no_sleep_or_polling_on_the_observability_path() -> None:
    """The signal rides the turn reply; it is never waited for."""
    inspect = __import__("inspect")
    for fn in (
        AgentProcess._with_compaction_flag,
        AgentProcess._turn_compaction_report,
        AgentProcess._reset_turn_compaction_report,
    ):
        src = inspect.getsource(fn)
        assert "sleep" not in src
        assert "while " not in src


# ---------------------------------------------------------------------------
# The STANDING key: ``context_compaction_observable: false``.
#
# Decision (B), agreed with the consumer. A session-stateful provider keeps history on
# its own side and can compact it without telling us, so on those providers
# "no report" cannot be read as "no compaction". All three of the consumer's
# extraction agents run on codex today, so this is the live configuration, not
# an edge case: without the key the detector is blind exactly where it is needed
# and still reports "no summarization".
#
# The key is a STANDING property of the provider, arriving on every turn. It is
# NOT the per-turn report, and the two must never be conflated: read as an
# incident it would fail every document of every run forever.
# ---------------------------------------------------------------------------
def _process_on(provider: str | None) -> AgentProcess:
    """A process whose provider identity is ``provider``.

    Reaches through ``config.provider`` — the same attribute production reads
    (``agent_process.py`` passes ``self.config.provider`` to the model factory),
    so this is the real axis, not a parallel one invented for the test.
    """
    proc = _process()

    class _Config:
        pass

    cfg = _Config()
    cfg.provider = provider  # type: ignore[attr-defined]
    proc.config = cfg  # type: ignore[attr-defined]
    return proc


@pytest.mark.parametrize(
    "provider", ["codex", "acp", "claude-code", "claude-code-sdk", "grok"]
)
def test_standing_key_rides_every_turn_on_unobservable_providers(provider: str) -> None:
    """Including a turn with NO compaction: the key is a provider property.

    Asserts the key's VALUE, not merely its presence — ``False`` is the whole
    message, and a key present with ``True`` would tell the consumer the
    opposite of the truth.
    """
    proc = _process_on(provider)
    sm = _middleware()
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert reply["context_compaction_observable"] is False
    # Standing property, NOT an incident: the per-turn keys stay absent, so the
    # consumer logs loudly and does not fail the document.
    assert "context_compacted" not in reply
    assert "context_summarized" not in reply


def test_standing_key_is_absent_on_observable_providers() -> None:
    """The guard against the mutation "emit it always".

    The consumer's warning, in its dangerous direction for this side: a test that
    only checks "the key appeared" passes even when the key is emitted
    unconditionally, which would tell the consumer that deep_worker can never
    see its own summarization.
    """
    proc = _process_on("deepseek")
    sm = _middleware()
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert "context_compaction_observable" not in reply, (
        "the in-process path IS observable; claiming otherwise makes the "
        "consumer log a blind spot on every turn of every run"
    )


def test_standing_key_is_never_shaped_as_a_per_turn_report() -> None:
    """The consumer's explicit request: two keys, two meanings.

    The standing property must NOT arrive as
    ``context_summarized: {"observable": False}``. That shape is the per-turn
    incident report, and the receiver routes it to "fail the document and
    re-extract"; sent on every turn it is a permanent outage rather than a
    safety margin.
    """
    proc = _process_on("codex")
    sm = _middleware()
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert "context_summarized" not in reply, (
        "the standing key must not masquerade as the per-turn report"
    )
    assert reply["context_compaction_observable"] is False


def test_real_compaction_on_an_unobservable_provider_still_reports_the_incident() -> None:
    """The two keys coexist and keep their separate meanings.

    A codex-backed agent whose middleware DID observe a size compaction must
    send both: the incident (fail this document) and the standing property
    (this provider can also compact unseen).
    """
    proc = _process_on("codex")
    sm = _middleware()
    sm.last_turn_compaction = {
        "occurred": True,
        "observable": True,
        "reason": "size_trigger",
        "at_utc": _utc_now_iso(),
    }
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]

    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert reply["context_compacted"] is True
    assert reply["context_summarized"]["occurred"] is True
    # The report's own ``observable`` describes the event we DID see; the
    # standing key describes the provider. Different subjects, both true.
    assert reply["context_summarized"]["observable"] is True
    assert reply["context_compaction_observable"] is False


@pytest.mark.asyncio
async def test_pseudo_summary_on_codex_sends_only_the_standing_key() -> None:
    """The most expensive possible outcome of this change, pinned.

    The consumer's agents run on codex AND use ``--sum-query`` per document. If
    pseudo-summarization leaked a per-turn report here, every document of every
    run would be failed and re-extracted forever. The standing key is expected;
    the incident keys are not.
    """
    sm, written = _pseudo_ready_middleware()

    await SummarizationMiddleware.apply_pseudo_summary(
        sm, "agent", "EXTRACT THIS DOCUMENT", _history(12)
    )

    proc = _process_on("codex")
    proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]
    reply = proc._with_compaction_flag(
        {"success": True, "response": {"role": "assistant", "content": "DONE"}}
    )

    assert written, "the real offload must have run"
    assert reply["context_compaction_observable"] is False
    assert "context_compacted" not in reply, (
        "a per-turn incident on the per-document reset fails every document "
        "of every run — the exact outcome the consumer warned about"
    )
    assert "context_summarized" not in reply


def test_unknown_provider_does_not_claim_a_blind_spot() -> None:
    """A missing/garbled identity must not be reported as unobservable.

    Same reasoning as ``_is_session_stateful``: answering "is it NOT stateless?"
    made every unknown provider and every mock look session-stateful. Only a
    positively identified CLI-session provider gets the key.
    """
    sm = _middleware()
    for provider in (None, "", "   ", "not-a-real-provider"):
        proc = _process_on(provider)
        proc._find_summarization_middleware = lambda: sm  # type: ignore[method-assign]
        reply = proc._with_compaction_flag({"success": True, "response": {}})
        assert "context_compaction_observable" not in reply, provider


def test_observability_uses_the_single_source_of_truth() -> None:
    """No second hand-maintained list of provider names.

    The property "history lives on the provider's side" IS the session-stateful
    classification, and ``SESSION_STATEFUL_PROVIDERS`` is its one definition. A
    copy here would drift the first time a provider is added — silently in the
    direction of claiming observability we do not have.
    """
    inspect = __import__("inspect")
    src = inspect.getsource(AgentProcess._compaction_is_observable)
    assert "SESSION_STATEFUL_PROVIDERS" in src
    for name in ("codex", "claude-code", "grok", "acp"):
        assert f'"{name}"' not in src, (
            f"{name} is hardcoded here instead of being resolved from "
            "SESSION_STATEFUL_PROVIDERS"
        )


def test_standing_key_reaches_every_success_reply_site() -> None:
    """Both keys ride the same wrapper, so one wiring proves all three sites.

    Pinned explicitly: if the standing key were ever stamped somewhere other
    than ``_with_compaction_flag``, the three-site guarantee proved by
    ``test_every_success_reply_site_is_stamped`` would stop covering it.
    """
    inspect = __import__("inspect")
    wrapper = inspect.getsource(AgentProcess._with_compaction_flag)
    assert "context_compaction_observable" in wrapper

    body = inspect.getsource(AgentProcess._handle_invoke_owned)
    assert "context_compaction_observable" not in body, (
        "the standing key must be stamped by the shared wrapper only"
    )
