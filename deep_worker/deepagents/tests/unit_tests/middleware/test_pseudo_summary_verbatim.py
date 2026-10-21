"""A pseudo-summary (``sum_query``) reaches the model BYTE FOR BYTE.

Incident 2026-10-01. the code-graph server opens its stage-2 agent
(``rg_cross_file_updater``) via ``sum_query`` carrying the full task. The
middleware wrapped that task in ``<summary>`` tags, prefixed "You are in the
middle of a conversation that has been summarized", and appended "The full
conversation history has been saved to …/summaries/<agent>.jsonl should you
need to refer back to it". deepseek read its own task as a recap of the past
and spent the turn trying to inspect that file — 20 consecutive ``execute``
calls (``ls -la …/summaries/ && wc -c …/rg_cross_file_updater.jsonl``) for a
tool it does not even have. The pass added zero links. The delta linker
(``rg_linker``) broke the same way on 2026-09-25.

Crucially this was never size-based summarization: 23 164 tokens against a
170 000 threshold. The sum-query IS the agent's first user request (turn 1
drives the graph on an empty ``message_list``), so any framing is a lie about
what the message is.

These tests pin the sum-query path only. Real size-based summarization keeps
its framing — ``test_real_summarization_framing_is_unchanged`` is the guard
that this fix did not leak into it.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AnyMessage, HumanMessage

from deepagents.middleware.summarization import SummarizationMiddleware

_JSONL_PATH = "/conversation_history/rg_cross_file_updater.jsonl"

# The verbatim task text, deliberately multi-line and containing characters
# that a naive wrapper would mangle.
_TASK = """Stage 2 — cross-file links.

For each symbol in the batch call submit_semantic. Do NOT call execute.
Budget: 40 symbols. Report <done> when the batch is exhausted."""


def _bare_middleware() -> SummarizationMiddleware:
    """A middleware instance without running __init__.

    ``_build_new_messages_with_path`` touches neither the model nor the
    backend, so the construction cost (and its required deps) is irrelevant
    here — the sibling integration test covers the wired path.
    """
    return SummarizationMiddleware.__new__(SummarizationMiddleware)


class TestPseudoSummaryIsVerbatim:
    @pytest.mark.parametrize("file_path", [_JSONL_PATH, None])
    def test_content_equals_sum_query_byte_for_byte(
        self, file_path: str | None
    ) -> None:
        """Requirement (a): content == sum_query, exactly.

        Parametrized over both branches: an offload DID happen (file_path set)
        and did not (empty history). The old code wrapped BOTH — the second
        with a "Here is a summary of the conversation to date:" prefix — so
        neither was verbatim.
        """
        built = _bare_middleware()._build_new_messages_with_path(
            _TASK, file_path, verbatim=True
        )
        assert len(built) == 1
        assert built[0].content == _TASK

    @pytest.mark.parametrize("file_path", [_JSONL_PATH, None])
    def test_no_wrapper_no_history_file_reference(
        self, file_path: str | None
    ) -> None:
        """Requirement (b): no tags, no summaries path, no history phrase."""
        content = _bare_middleware()._build_new_messages_with_path(
            _TASK, file_path, verbatim=True
        )[0].content

        assert "<summary>" not in content
        assert "</summary>" not in content
        assert ".jsonl" not in content
        assert not re.search(r"summaries/", content)
        assert "full conversation history has been saved" not in content
        assert "has been summarized" not in content
        assert "Here is a summary of the conversation" not in content

    def test_the_exact_strings_that_misled_deepseek_are_absent(self) -> None:
        """The incident's literal triggers, named so a partial fix is caught.

        A fix that dropped only the ``<summary>`` tags but kept the
        history-file sentence would still reproduce the defect: it is that
        sentence the model acted on.
        """
        content = _bare_middleware()._build_new_messages_with_path(
            _TASK, _JSONL_PATH, verbatim=True
        )[0].content

        for forbidden in (
            "You are in the middle of a conversation that has been summarized.",
            "should you need to refer back to it for details.",
            "A condensed summary follows:",
            _JSONL_PATH,
        ):
            assert forbidden not in content, f"still present: {forbidden!r}"

    def test_summarization_marker_and_timestamp_survive(self) -> None:
        """Verbatim must not cost the message its metadata.

        ``lc_source='summarization'`` is what ``_filter_summary_messages``
        keys on, and an unstamped message renders without a time in the web
        UI. Both are orthogonal to the content and must stay.
        """
        msg = _bare_middleware()._build_new_messages_with_path(
            _TASK, _JSONL_PATH, verbatim=True
        )[0]
        assert msg.additional_kwargs.get("lc_source") == "summarization"
        stamp = [k for k in msg.additional_kwargs if "created" in k]
        assert stamp, f"no created_at stamp: {msg.additional_kwargs!r}"


class TestRealSummarizationUnaffected:
    """Requirement (c): size-based summarization behaves exactly as before."""

    def test_real_summarization_framing_is_unchanged(self) -> None:
        """Without the flag, the full framing is still produced.

        This is the blast-radius guard. ``_build_new_messages_with_path`` is
        shared: the pseudo path calls it twice (summarization.py:1338, :1354)
        and real summarization twice more (:1441 sync, :1742 async). Only the
        first pair opts in.
        """
        content = _bare_middleware()._build_new_messages_with_path(
            "a genuine recap", _JSONL_PATH
        )[0].content

        assert content.startswith(
            "You are in the middle of a conversation that has been summarized."
        )
        assert "<summary>\na genuine recap\n</summary>" in content
        assert _JSONL_PATH in content
        assert "should you need to refer back to it for details." in content

    def test_real_summarization_empty_history_branch_unchanged(self) -> None:
        content = _bare_middleware()._build_new_messages_with_path(
            "a genuine recap", None
        )[0].content
        assert content == (
            "Here is a summary of the conversation to date:\n\na genuine recap"
        )

    def test_verbatim_defaults_to_off(self) -> None:
        """The flag is opt-in: a caller that forgets it gets the old shape.

        Defaulting to verbatim would silently strip the framing from real
        summarization, where the model genuinely needs to know it is looking
        at a recap and where the history file lives.
        """
        content = _bare_middleware()._build_new_messages_with_path(
            "recap", _JSONL_PATH
        )[0].content
        assert "<summary>" in content


class TestPseudoSummaryEndToEnd:
    """Through ``apply_pseudo_summary``, not just the private helper.

    The helper being right is not the same as the right thing reaching the
    helper — the defect lived exactly in that gap on the ACP profile line.
    """

    @pytest.mark.asyncio
    async def test_apply_pseudo_summary_returns_verbatim_message(
        self, tmp_path
    ) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        fake = __import__(
            "langchain_core.language_models.fake_chat_models",
            fromlist=["GenericFakeChatModel"],
        ).GenericFakeChatModel(messages=iter([HumanMessage(content="x")]))
        mw = SummarizationMiddleware(
            model=fake,
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            trigger=("messages", 5),
            keep=("messages", 2),
            history_path_prefix="/conversation_history",
        )

        history: list[AnyMessage] = [
            HumanMessage(content="old 1", id="h1"),
            HumanMessage(content="old 2", id="h2"),
        ]
        new_msgs, file_path = await mw.apply_pseudo_summary(
            "rg_cross_file_updater", _TASK, history
        )

        # The offload is real — it is simply not advertised to the model.
        assert file_path == "/conversation_history/rg_cross_file_updater.jsonl"
        assert (
            tmp_path / "conversation_history" / "rg_cross_file_updater.jsonl"
        ).exists()

        assert new_msgs[0].content == _TASK
        assert ".jsonl" not in new_msgs[0].content
