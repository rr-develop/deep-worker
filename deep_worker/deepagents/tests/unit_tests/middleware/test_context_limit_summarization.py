"""Live-path tests for operator ``context_limit`` on summarization defaults.

Design: docs/DESIGN_CLI_CONTEXT_LIMIT_FLAG.md §4.2 / tests 3-6.

Pins:
  * ``_compute_summarization_defaults(model, context_limit=L)`` builds the
    trigger disjunction and derives keep + truncate_args from L.
  * All three ``SummarizationMiddleware`` construction sites in
    ``create_deep_agent`` receive those values (GP subagent, user subagent,
    main agent).
  * Fractions are the hardcoded 0.85 / 0.10 (not a silent other ratio).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepagents.graph import create_deep_agent
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    _compute_summarization_defaults,
)


def _model_with_profile(max_input_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(profile={"max_input_tokens": max_input_tokens})


def _model_without_profile() -> SimpleNamespace:
    return SimpleNamespace(profile=None)


class TestComputeSummarizationDefaultsContextLimit:
    def test_no_limit_leaves_profile_defaults_untouched(self) -> None:
        defaults = _compute_summarization_defaults(_model_with_profile(1_000_000))
        assert defaults["trigger"] == ("fraction", 0.85)
        assert defaults["keep"] == ("fraction", 0.10)
        assert defaults["truncate_args_settings"]["trigger"] == ("fraction", 0.85)
        assert defaults["truncate_args_settings"]["keep"] == ("fraction", 0.10)

    def test_limit_below_model_default_adds_token_trigger(self) -> None:
        """Test 4 — L=200k on a 1M model: limit-side trigger is 170k tokens."""
        defaults = _compute_summarization_defaults(
            _model_with_profile(1_000_000), context_limit=200_000
        )
        trigger = defaults["trigger"]
        assert isinstance(trigger, list), (
            "operator limit must be OR-ed with the model default, not replace it"
        )
        assert ("tokens", 170_000) in trigger
        assert ("fraction", 0.85) in trigger

    def test_limit_above_model_default_still_keeps_model_trigger(self) -> None:
        """Test 3 — L=2M on a 200k-profile model: model default remains in the list."""
        defaults = _compute_summarization_defaults(
            _model_with_profile(200_000), context_limit=2_000_000
        )
        trigger = defaults["trigger"]
        assert isinstance(trigger, list)
        assert ("tokens", 1_700_000) in trigger
        assert ("fraction", 0.85) in trigger

    def test_keep_derives_from_limit_not_model_window(self) -> None:
        """Test 5 — keep is 10% of L, not 10% of the model window."""
        defaults = _compute_summarization_defaults(
            _model_with_profile(1_000_000), context_limit=200_000
        )
        assert defaults["keep"] == ("tokens", 20_000)

    def test_truncate_args_recomputed_from_limit(self) -> None:
        defaults = _compute_summarization_defaults(
            _model_with_profile(1_000_000), context_limit=200_000
        )
        assert defaults["truncate_args_settings"]["trigger"] == ("tokens", 170_000)
        assert defaults["truncate_args_settings"]["keep"] == ("tokens", 20_000)

    def test_no_profile_limit_does_not_or_fixed_170k(self) -> None:
        """``-l`` must not OR the 170k ACP-dataclass fallback (incident 2026-08-19)."""
        defaults = _compute_summarization_defaults(
            _model_without_profile(), context_limit=100_000
        )
        trigger = defaults["trigger"]
        assert trigger == ("tokens", 85_000)
        assert defaults["keep"] == ("tokens", 10_000)

    def test_dataclass_profile_counts_as_profile(self) -> None:
        profile = SimpleNamespace(max_input_tokens=1_000_000)
        defaults = _compute_summarization_defaults(
            SimpleNamespace(profile=profile), context_limit=400_000
        )
        trigger = defaults["trigger"]
        assert isinstance(trigger, list)
        assert ("tokens", 340_000) in trigger
        assert ("fraction", 0.85) in trigger
        assert ("tokens", 170_000) not in trigger


class TestAllThreeMiddlewareSitesReceiveContextLimit:
    """Test 6 — mute any of the 3 graph.py sites and this goes red."""

    def test_gp_user_subagent_and_main_all_get_limit_derived_settings(self) -> None:
        captured: list[dict[str, Any]] = []
        orig_init = SummarizationMiddleware.__init__

        def _spy(self: SummarizationMiddleware, *args: Any, **kwargs: Any) -> None:
            captured.append(
                {
                    "trigger": kwargs.get("trigger"),
                    "keep": kwargs.get("keep"),
                    "truncate_args_settings": kwargs.get("truncate_args_settings"),
                }
            )
            return orig_init(self, *args, **kwargs)

        model = FakeListChatModel(responses=["ok"])
        model.profile = {"max_input_tokens": 1_000_000}

        with patch.object(SummarizationMiddleware, "__init__", _spy):
            create_deep_agent(
                model=model,
                context_limit=200_000,
                subagents=[
                    {
                        "name": "researcher",
                        "description": "research helper",
                        "system_prompt": "You research.",
                    }
                ],
            )

        assert len(captured) == 3, (
            f"expected 3 SummarizationMiddleware sites (GP, user subagent, "
            f"main); got {len(captured)}"
        )
        for i, seen in enumerate(captured):
            trigger = seen["trigger"]
            assert isinstance(trigger, list), f"site {i} trigger is not a list: {trigger!r}"
            assert ("tokens", 170_000) in trigger, f"site {i} missing limit trigger: {trigger!r}"
            assert ("fraction", 0.85) in trigger, f"site {i} missing model default: {trigger!r}"
            assert seen["keep"] == ("tokens", 20_000), f"site {i} keep={seen['keep']!r}"
            tas = seen["truncate_args_settings"]
            assert tas["trigger"] == ("tokens", 170_000), f"site {i} truncate trigger={tas!r}"
            assert tas["keep"] == ("tokens", 20_000), f"site {i} truncate keep={tas!r}"
