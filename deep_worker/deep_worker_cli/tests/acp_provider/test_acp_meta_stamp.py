"""grok reports its context in the prompt reply, not in usage_update.

Measured live on grok 1.0.5 (2026-08-25): a whole turn produces only
``user_message_chunk``, ``agent_thought_chunk``, ``agent_message_chunk`` and
``available_commands_update``. There is NO ``usage_update`` — the ACP gauge
the stamp used to depend on. Consequence, verified across every grok agent
in this deployment: zero context stamps ever, so the guard and the
summarizer saw UNKNOWN on every turn and an agent running ``-l 250k`` had
no context protection at all.

The number does exist and we already parse it: ``session/prompt``'s reply
carries ``_meta.inputTokens`` — the prompt grok actually read. Measured over
three consecutive turns of one session: 14144 → 14190 → 14335, matching
grok's own ``inference_done.prompt_tokens`` exactly, and larger than
``cachedReadTokens`` (11648 / 14080), i.e. it is the FULL prompt including
the cached prefix, which is what "context size" means.

That is a provider report, not an estimate (POLICIES §13): it comes from
the provider's own tokenizer, in the same reply that carries its usage.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker_cli.context_limit.observed_context import context_observation


def _model() -> AcpChatModel:
    return object.__new__(AcpChatModel)


def _meta(input_tokens: int | None, cached: int = 0) -> dict:
    meta: dict = {"cachedReadTokens": cached, "outputTokens": 12}
    if input_tokens is not None:
        meta["inputTokens"] = input_tokens
    return meta


class TestStampFromPromptMeta:
    def test_meta_input_tokens_becomes_the_stamp(self) -> None:
        state = ThreadState()
        msg = _model()._stamp_context_observation(
            state, AIMessage(content="ok"), meta=_meta(14335, cached=14080)
        )
        obs = context_observation(msg)
        assert obs is not None
        assert obs["used"] == 14335

    def test_usage_update_still_wins_when_the_cli_sends_one(self) -> None:
        """claude-code does send usage_update, and it measures the whole
        session the CLI holds — richer than one prompt's input count."""
        state = ThreadState()
        state.usage_snapshot = {"used": 468_146, "size": 1_000_000}
        msg = _model()._stamp_context_observation(
            state, AIMessage(content="ok"), meta=_meta(9_000)
        )
        obs = context_observation(msg)
        assert obs is not None
        assert obs["used"] == 468_146
        assert obs["size"] == 1_000_000

    def test_no_snapshot_and_no_meta_means_no_stamp(self) -> None:
        """UNKNOWN stays unknown — never invented (POLICIES §13)."""
        msg = _model()._stamp_context_observation(
            ThreadState(), AIMessage(content="ok"), meta=None
        )
        assert context_observation(msg) is None

    def test_garbage_meta_is_not_a_report(self) -> None:
        for bad in ({"inputTokens": 0}, {"inputTokens": -3}, {"inputTokens": "big"}, {}):
            msg = _model()._stamp_context_observation(
                ThreadState(), AIMessage(content="ok"), meta=bad
            )
            assert context_observation(msg) is None

    def test_the_finalizer_passes_the_meta_through(self) -> None:
        """Wiring guard: the stamp is only as good as its call site."""
        import inspect

        src = inspect.getsource(AcpChatModel._finalize_response)
        assert "meta=" in src, (
            "_finalize_response must hand the prompt reply's _meta to the "
            "stamp; without it grok has no context report at all"
        )
