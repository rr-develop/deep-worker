"""Every shipped provider must report its context size. No exceptions.

POLICIES §13 (owner directive 2026-08-25): the guard and the summarizer
decide on the provider's number, and a fallback needs explicit approval
that no provider has. A provider that returns an unstamped message is
therefore a defect — it leaves both consumers with UNKNOWN, i.e. unguarded.

This file is the census. It checks the mechanism each provider uses, so a
new provider (or a refactor that drops a stamp) fails here rather than in
production at 3am.
"""

from __future__ import annotations

import inspect

import pytest


class TestProvidersStamp:
    """Session-holding providers stamp from their CLI's own gauge."""

    def test_acp_stamps_from_usage_update(self) -> None:
        from deep_worker.providers.acp import chat_model

        src = inspect.getsource(chat_model)
        assert "_stamp_context_observation" in src
        assert "annotate_context_observation" in src

    def test_codex_stamps_its_rollout_reading(self) -> None:
        from deep_worker.providers.codex_provider import chat_model

        src = inspect.getsource(chat_model)
        assert "_stamp_provider_context" in src
        # The value comes from the live rollout gauge, not from a guess.
        assert "last_provider_context" in src

class TestEveryProviderReportsPerModelCall:
    """A reading per model call — reporting at the end of the turn is a defect.

    POLICIES §13 "Cadence". One provider turn is not one model call: grok
    has been measured at 98 inferences in a single turn (18525 -> 97368
    tokens and beyond, ``~/.grok/logs/unified.jsonl``, 2026-08-25). A gauge
    that moves once per turn is frozen for the whole time the agent is
    actually working, and the guard that reads it decides on a number that
    is one turn stale.

    These are structural pins, not behaviour: the behaviour lives in
    ``tests/acp_provider/test_grok_live_context.py`` and
    ``tests/unit_tests/test_codex_live_context.py``. The value of a
    census is that a NEW provider — or a refactor that quietly reverts one
    to end-of-turn reporting — fails here.
    """

    def test_acp_watches_its_cli_log_for_the_whole_turn(self) -> None:
        from deep_worker.providers.acp import chat_model

        src = inspect.getsource(chat_model)
        assert "_emit_context_snapshot" in src, (
            "acp has no per-call snapshot emitter"
        )
        turn = inspect.getsource(chat_model.AcpChatModel._consume_turn)
        assert "_start_live_context_watch" in turn, (
            "the acp gauge is not started at the head of the turn: readings "
            "would only exist for whatever the final message happens to carry"
        )

    def test_codex_watches_its_rollout_for_the_whole_turn(self) -> None:
        from deep_worker.providers.codex_provider import chat_model

        src = inspect.getsource(chat_model)
        assert "_emit_context_snapshot" in src, (
            "codex has no per-call snapshot emitter"
        )
        turn = inspect.getsource(chat_model.CodexChatModel._consume_turn)
        assert "await self._start_context_watch(" in turn, (
            "codex never starts its rollout watch inside the turn it is "
            "meant to measure: the gauge can only move on turn.completed, "
            "i.e. once per turn"
        )
        assert "await self._stop_context_watch(" in turn, (
            "codex never stops the watch: the final drain that carries the "
            "last model call of a short turn never runs"
        )

    def test_there_is_only_one_gauge_implementation(self) -> None:
        """Codex must not own a second copy of the watcher.

        It did, and the copy drifted exactly where copies do: its stop
        had no final drain, so a turn ending inside the change-stream's
        debounce reported nothing at all, and its drain emitted only the
        newest reading, so two model calls landing between two wake-ups
        counted as one. Both were fixed once in the shared class and were
        still broken in the copy — found live on 2026-08-26, months after
        the shared one was correct.
        """
        from deep_worker.providers.codex_provider.context_watch import (
            RolloutContextWatcher,
        )
        from deep_worker_cli.context_limit.live_gauge import LiveContextWatcher

        assert issubclass(RolloutContextWatcher, LiveContextWatcher), (
            "codex has its own watcher again — the next fix to the shared "
            "gauge will silently miss it"
        )
        for method in ("start", "stop", "_drain", "_run"):
            assert method not in vars(RolloutContextWatcher), (
                f"codex overrides {method!r}: the shared lifecycle is being "
                "forked again, which is how the missing final drain survived"
            )

    @pytest.mark.parametrize(
        "module_name",
        [
            "deep_worker.providers.acp.chat_model",
            "deep_worker.providers.codex_provider.chat_model",
        ],
    )
    def test_a_reading_is_never_billed_as_spend(self, module_name: str) -> None:
        """The snapshot rides the usage callback; forwarding the usage blob
        with it would count one turn's tokens once per model call."""
        import ast
        import importlib
        import textwrap

        module = importlib.import_module(module_name)
        fn = None
        for cls_name in dir(module):
            cls = getattr(module, cls_name)
            fn = getattr(cls, "_emit_context_snapshot", None)
            if callable(fn):
                break
        assert fn is not None, f"{module_name} has no _emit_context_snapshot"

        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "usage":
                    continue
                assert isinstance(kw.value, ast.Constant) and kw.value.value is None, (
                    f"{module_name}._emit_context_snapshot forwards a usage "
                    "payload: every gauge reading would be persisted as a "
                    "billable turn"
                )


def _used_call(fn, usage):
    return fn(usage)


class TestStandardProvidersNeedNoStamp:
    """LangChain already carries the provider's own prompt count for every
    standard API provider; the reader accepts it directly."""

    def test_usage_metadata_is_accepted_as_a_report(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from deep_worker_cli.context_limit.observed_context import (
            provider_context_tokens,
        )

        msgs = [
            HumanMessage(content="hi", id="h1"),
            AIMessage(
                content="ok",
                id="a1",
                usage_metadata={
                    "input_tokens": 12_345,
                    "output_tokens": 7,
                    "total_tokens": 12_352,
                },
            ),
        ]
        assert provider_context_tokens(msgs) == 12_345


@pytest.mark.parametrize(
    "module_name",
    [
        "deep_worker.providers.acp.chat_model",
        "deep_worker.providers.codex_provider.chat_model",
    ],
)
def test_no_provider_estimates_its_own_context(module_name: str) -> None:
    """A provider must report what its CLI says, never compute a substitute."""
    module = __import__(module_name, fromlist=["*"])
    src = inspect.getsource(module)
    assert "estimate_request_tokens" not in src
    assert "// 4" not in src.replace("//", "// ", 0) or "token" not in src
