"""RED tests for per-turn usage SOURCE selection (DESIGN_PER_TURN_USAGE_MAPPING.md).

These tests pin WHICH provider object the token record is built from. They
deliberately do NOT test arithmetic: ``_from_anthropic`` / ``_from_codex``
are already correct and a hand-made dict fed straight into them passes
today, so such a test cannot express this defect.

The defect pinned: codex emits ``turn.completed.usage``, which carries ``total_token_usage``
semantics whenever one ``codex exec`` process spans several model calls
(handle reuse across tool round-trips, ``chat_model.py:604-614``). Live:
our record wrote ``prompt=3736695`` while that turn's own spend was
``113457`` — 32.9x inflation.

Live evidence backing every constant here is quoted in
``docs/DESIGN_PER_TURN_USAGE_MAPPING.md`` §2.1-bis / §2.1-ter.

⚠ SEATING (reviewer must-fix, 2026-08-22). Every test below drives a
PRODUCTION channel:
  - codex: the rollout lookup the emit path will call, located through the
    same ``CODEX_HOME`` glob production uses (mirroring
    ``resolve_model_from_session_log``), NOT a file dropped next to
    ``token_stats.jsonl``; plus a thin test that ``:855`` actually consults
    it instead of ``event.usage``.
An earlier revision of this file set ``_last_assistant_usage`` — an
attribute no product code writes or reads — and dropped rollouts into the
recorder's tmp dir. Both were false seams: a "fix" could green them while
production still emitted 770 / 3736695. Do not reintroduce either.

⚠ WIRING PINS (reviewer must-fix #2, 2026-08-22). ``TestProductionWiring``
reads the call out of production source, on the AST (the repo pattern from
``test_agent_process_turn_ending_diagnostics.py``) so formatting churn
cannot break the pin and a renamed variable cannot slip past it.

⚠ CODEX PLACEMENT (reviewer must-fix #2b). The reviewer flagged a
doc/test contradiction: ``TestCodexEmitSiteUsesLookup`` drives
``_emit_usage_event`` directly, so it can only ever green if the lookup
lives INSIDE that method, while the design doc put it in
``_consume_turn``'s ``TurnCompletedEvent`` branch at ``:855``. Resolved as
option (b) — the lookup belongs in codex ``_emit_usage_event``, which
already receives ``session_id``/``turn_id`` (chat_model.py:344-352) and so
needs no new plumbing; ``:855`` keeps passing ``event.usage`` and the
terminal emit overrides it from the rollout, failing closed to
``no_usage``. ``docs/DESIGN_PER_TURN_USAGE_MAPPING.md`` §2.1-ter has been
updated to match, so doc and test now describe one design.

⚠ FIXTURE INVARIANT: a single-model-call fixture is USELESS for the codex
tests. When an exec process makes exactly one model call, TOTAL == LAST and
the bug is invisible — that is exactly why the live ``item_0`` row
(``92375``) looked correct while being produced by the same defective line,
and why the author's first probe wrongly concluded "already per-turn".
Every codex fixture below therefore has TOTAL != LAST.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.stats.token_stats import (
    STATUS_NO_USAGE,
    STATUS_OK,
    TokenUsageRecorder,
    read_stats_file,
)

# ---------------------------------------------------------------------------
# Live-derived constants (docs/DESIGN_PER_TURN_USAGE_MAPPING.md §2.1-bis)
# rollout-2026-08-22T17-54-51-01a02a2e-968d-7103-9d31-c36afe005682.jsonl
# ---------------------------------------------------------------------------

SESSION_UUID = "01a02a2e-968d-7103-9d31-c36afe005682"
TURN_ID = "item_40"

# What ``turn.completed.usage`` actually delivers on a multi-call exec: the
# process TOTAL. This is what we wrongly persist today.
CODEX_TOTAL_BLOB = {
    "input_tokens": 3736695,
    "cached_input_tokens": 3600640,
    "cache_write_input_tokens": 0,
    "output_tokens": 5212,
    "reasoning_output_tokens": 0,
}

# The turn's OWN spend, from ``token_count.info.last_token_usage`` at
# 18:38:33.188Z — 11 ms before that exec's ``task_complete`` at 18:38:33.192Z.
CODEX_LAST_INPUT = 113457
CODEX_LAST_CACHED = 112384
CODEX_LAST_OUTPUT = 83

# A LATER token_count line (18:39:45.741Z) belonging to a SUBSEQUENT exec in
# the same append-only rollout. Reading "the latest line" yields this —
# wrong by a small, plausible margin that passes every magnitude rail.
CODEX_LATER_INPUT = 114460

# Emit instant of the real record (34 ms after the token_count above).
EMIT_TS = "2026-08-22T18:38:33.222Z"


# ---------------------------------------------------------------------------
# Rollout fixture — real shape, real CODEX_HOME layout
# ---------------------------------------------------------------------------


def _write_rollout(
    codex_home: Path,
    *,
    entries: list[tuple[str, str, int, int]],
    session_uuid: str = SESSION_UUID,
) -> Path:
    """Write a rollout JSONL into a real ``CODEX_HOME/sessions`` tree.

    ``entries`` is a list of ``(kind, timestamp, last_in, total_in)`` where
    ``kind`` is ``"token_count"`` or ``"task_complete"``. Mirrors the live
    shape: ``{"timestamp", "type": "event_msg", "payload": {...}}`` with
    ``payload.info.last_token_usage`` / ``total_token_usage``.

    Path layout matches production date-sharding
    (``sessions/<yyyy>/<mm>/<dd>/rollout-<ts>-<uuid>.jsonl``) so a lookup
    that globs by uuid finds it, and one that builds today's date does not.
    """
    day_dir = codex_home / "sessions" / "2026" / "08" / "22"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"rollout-2026-08-22T17-54-51-{session_uuid}.jsonl"

    lines = []
    for kind, ts, last_in, total_in in entries:
        if kind == "token_count":
            payload = {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": total_in,
                        "cached_input_tokens": CODEX_LAST_CACHED,
                        "output_tokens": CODEX_LAST_OUTPUT,
                        "total_tokens": total_in + CODEX_LAST_OUTPUT,
                    },
                    "last_token_usage": {
                        "input_tokens": last_in,
                        "cached_input_tokens": CODEX_LAST_CACHED,
                        "output_tokens": CODEX_LAST_OUTPUT,
                        "total_tokens": last_in + CODEX_LAST_OUTPUT,
                    },
                    "model_context_window": 258400,
                },
            }
        else:
            payload = {"type": "task_complete"}
        lines.append(
            json.dumps({"timestamp": ts, "type": "event_msg", "payload": payload})
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _lookup():
    """The production helper the codex emit path will call.

    Contract (design §2.1-ter): given the session uuid and the emit instant,
    return that turn's ``token_count.info.last_token_usage`` — the one
    immediately preceding the last ``task_complete`` at-or-before the emit
    instant — or ``None`` (fail-closed) when it cannot be determined.

    Imported lazily so the failure names the symbol to add rather than
    breaking collection for the whole file.
    """
    from deep_worker.providers.codex_provider.chat_model import (
        last_token_usage_from_rollout,
    )

    return last_token_usage_from_rollout


# ===========================================================================
# §6.1 tests 3-5 — codex: last_token_usage via task_complete correlation
# ===========================================================================


class TestCodexRolloutLookup:
    """The helper the emit path calls: rollout -> this turn's last_token_usage.

    Seated on the production lookup (CODEX_HOME glob by uuid), not on a file
    beside ``token_stats.jsonl``. These fail today with ImportError until
    ``last_token_usage_from_rollout`` exists — the honest RED for a function
    the design says must be added.
    """

    def test_reads_last_token_usage_not_total_blob(self, tmp_path):
        """RED (test 3): must return LAST (113457), never TOTAL (3736695)."""
        _write_rollout(
            tmp_path,
            entries=[
                ("token_count", "2026-08-22T18:38:30.110Z", 112887, 3623238),
                ("token_count", "2026-08-22T18:38:33.188Z",
                 CODEX_LAST_INPUT, CODEX_TOTAL_BLOB["input_tokens"]),
                ("task_complete", "2026-08-22T18:38:33.192Z", 0, 0),
            ],
        )

        usage = _lookup()(SESSION_UUID, EMIT_TS, codex_home=tmp_path)

        assert usage is not None
        assert usage["input_tokens"] != 3736695, (
            "returned the TOTAL blob — the live defect "
            "(32.9x inflation, 2026-08-22T18:38:33.222Z)"
        )
        assert usage["input_tokens"] == CODEX_LAST_INPUT
        assert usage["output_tokens"] == CODEX_LAST_OUTPUT
        assert usage["cached_input_tokens"] == CODEX_LAST_CACHED

    def test_fail_closed_when_rollout_unavailable(self, tmp_path):
        """RED (test 4): no rollout at all -> None (fail-closed)."""
        assert _lookup()(SESSION_UUID, EMIT_TS, codex_home=tmp_path) is None

    def test_fail_closed_when_no_task_complete(self, tmp_path):
        """RED (test 4b): token_count present but no terminal -> None.

        A turn still in flight must not borrow a neighbouring turn's number.
        """
        _write_rollout(
            tmp_path,
            entries=[
                ("token_count", "2026-08-22T18:38:33.188Z",
                 CODEX_LAST_INPUT, CODEX_TOTAL_BLOB["input_tokens"]),
            ],
        )
        assert _lookup()(SESSION_UUID, EMIT_TS, codex_home=tmp_path) is None

    def test_ignores_token_count_after_task_complete(self, tmp_path):
        """RED (test 5): the recency trap (§2.1-ter).

        The rollout is append-only for the WHOLE session across many exec
        processes, so "read the latest token_count" picks up a LATER turn's
        value: live 114460 instead of 113457. That error is small and
        plausible — it survives every magnitude/window sanity check, which
        makes it more dangerous than the 32.9x one.
        """
        _write_rollout(
            tmp_path,
            entries=[
                ("token_count", "2026-08-22T18:38:33.188Z",
                 CODEX_LAST_INPUT, CODEX_TOTAL_BLOB["input_tokens"]),
                ("task_complete", "2026-08-22T18:38:33.192Z", 0, 0),
                # Subsequent exec in the same rollout — must be ignored.
                ("token_count", "2026-08-22T18:39:45.741Z",
                 CODEX_LATER_INPUT, 456511),
            ],
        )

        usage = _lookup()(SESSION_UUID, EMIT_TS, codex_home=tmp_path)

        assert usage is not None
        assert usage["input_tokens"] != CODEX_LATER_INPUT, (
            "read the LATEST token_count instead of the one preceding this "
            "turn's task_complete — silently wrong by a plausible margin"
        )
        assert usage["input_tokens"] == CODEX_LAST_INPUT


async def _emit_codex_terminal(tmp_path: Path, *, usage: dict):
    """Drive the codex provider's own terminal emit, then read the record."""
    from deep_worker.providers.codex_provider.chat_model import CodexChatModel

    model = CodexChatModel.__new__(CodexChatModel)

    recorder = TokenUsageRecorder(tmp_path)
    cb = recorder.make_usage_event_callback(
        provider="codex", kind="agent", model_name="gpt-5.6-sol"
    )
    object.__setattr__(model, "_usage_event_callback", cb)
    object.__setattr__(model, "_resolved_model_name", "gpt-5.6-sol")

    await model._emit_usage_event(
        event="terminal",
        status=STATUS_OK,
        usage=usage,
        session_id=SESSION_UUID,
        turn_id=TURN_ID,
    )

    read = read_stats_file(tmp_path / "token_stats.jsonl")
    assert len(read.records) == 1, f"expected exactly 1 record, got {len(read.records)}"
    return read.records[0]


class TestCodexEmitSiteUsesLookup:
    """``chat_model.py:855`` must emit the looked-up usage, not ``event.usage``.

    Thin seating test: without it a correct helper could exist while the
    emit site still hands over the TOTAL blob.
    """

    @pytest.mark.asyncio
    async def test_emit_site_prefers_rollout_lookup_over_event_usage(
        self, tmp_path, monkeypatch
    ):
        """RED: the record must be 113457, not the 3736695 in ``event.usage``."""
        from deep_worker.providers.codex_provider import chat_model as cm

        monkeypatch.setattr(
            cm,
            "last_token_usage_from_rollout",
            lambda *a, **k: {
                "input_tokens": CODEX_LAST_INPUT,
                "cached_input_tokens": CODEX_LAST_CACHED,
                "output_tokens": CODEX_LAST_OUTPUT,
            },
            raising=False,
        )

        r = await _emit_codex_terminal(tmp_path, usage=CODEX_TOTAL_BLOB)

        assert r.prompt_tokens == CODEX_LAST_INPUT, (
            f"emit site persisted {r.prompt_tokens}: it is still using "
            "event.usage (the TOTAL blob) instead of the rollout lookup"
        )

    @pytest.mark.asyncio
    async def test_emit_site_records_no_usage_when_lookup_fails(
        self, tmp_path, monkeypatch
    ):
        """RED: floor — lookup returns None => ``no_usage``, never the blob."""
        from deep_worker.providers.codex_provider import chat_model as cm

        monkeypatch.setattr(
            cm, "last_token_usage_from_rollout", lambda *a, **k: None, raising=False
        )

        r = await _emit_codex_terminal(tmp_path, usage=CODEX_TOTAL_BLOB)

        assert r.prompt_tokens != 3736695, (
            "fail-closed violated: fell back to the TOTAL blob"
        )
        assert r.status == STATUS_NO_USAGE
        assert r.prompt_tokens is None


# ===========================================================================
# Wiring pins — production source, read on the AST
# ===========================================================================


def _fn_ast(fn):
    """Parse one production function into an AST tree.

    Same approach as ``test_agent_process_turn_ending_diagnostics.py``:
    reading the AST (not a substring) means reformatting cannot break the
    pin, and a renamed local cannot satisfy it by accident.
    """
    import ast as ast_mod
    import inspect
    import textwrap

    return ast_mod, ast_mod.parse(textwrap.dedent(inspect.getsource(fn)))


def _calls_to(ast_mod, tree, attr: str) -> list:
    """Every ``<something>.<attr>(...)`` call node in the tree."""
    return [
        node
        for node in ast_mod.walk(tree)
        if isinstance(node, ast_mod.Call)
        and isinstance(node.func, ast_mod.Attribute)
        and node.func.attr == attr
    ]


class TestProductionWiring:
    """The behavioural tests cannot see the call site; these can.

    Without these, a fix that only teaches ``_emit_usage_event`` a new
    keyword greens the behavioural tests while production still emits
    3736695 — test green, prod broken.
    """

    def test_codex_terminal_emit_consults_the_rollout_helper(self):
        """RED: codex ``_emit_usage_event`` must call the rollout lookup.

        Option (b): the lookup lives in the terminal emit (which already
        has ``session_id``), so ``:855`` may keep passing ``event.usage``.
        This pins that the helper is actually consulted; without it the
        monkeypatched behavioural tests above could be greened by a stub
        that never reaches production.
        """
        from deep_worker.providers.codex_provider.chat_model import CodexChatModel

        ast_mod, tree = _fn_ast(CodexChatModel._emit_usage_event)

        called = [
            node
            for node in ast_mod.walk(tree)
            if isinstance(node, ast_mod.Call)
            and (
                (isinstance(node.func, ast_mod.Name)
                 and node.func.id == "last_token_usage_from_rollout")
                or (isinstance(node.func, ast_mod.Attribute)
                    and node.func.attr == "last_token_usage_from_rollout")
            )
        ]
        assert called, (
            "codex _emit_usage_event never calls last_token_usage_from_rollout: "
            "the terminal emit still persists the TurnCompletedEvent TOTAL "
            "blob (3736695, 32.9x inflation) handed to it at :855"
        )
