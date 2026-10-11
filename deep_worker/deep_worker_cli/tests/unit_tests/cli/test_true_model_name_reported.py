"""The reported model name must be the one the agent actually runs on.

Two distinct defects, both reported by the operator on 2026-09-01, both
of which made the web UI status bar name a model the agent was NOT
talking to.

DEFECT 1 — ``-pp`` is ignored on first launch
---------------------------------------------
``-pp``/``--preferred-provider`` is documented to override BOTH the
preferred provider AND the active provider. That override lives in
``merge_restart_config()``, which only runs on ``--restart``. On a FIRST
launch there is no saved config, so ``merge_restart_config`` never runs
and ``build_launch_config()`` reads ``args.provider`` — still sitting at
the parser default ``"deepseek"``. The operator asks for
``-pp or-nemotron-ultra`` and silently lands on deepseek, with
deepseek's default model persisted alongside it.

DEFECT 2 — a stale guess outranks the provider's own answer
-----------------------------------------------------------
``exposed_get_metadata`` resolved the display model as::

    model = config_model or dynamic_model or resolved_model

``config_model`` is what was WRITTEN DOWN at launch; ``dynamic_model``
is what the provider ACTUALLY REPORTED for this session
(``set_dynamic_display_model``, fed by the provider's resolved-model
callback). Ordering the guess first means that after a failover restart
onto a different provider the old launch_config model wins over the live
fact, producing pairs like ``or-nemotron-ultra / z-ai/glm-5.2:free``.

``model_can_belong_to_provider`` does not save us here: for
config-declared API providers it deliberately returns True for any model
(the vendor validates, not us),
so the foreign string passes the filter untouched. Tightening that
predicate is NOT the fix — it would resurrect an earlier incident
where a valid model was rejected and silently replaced. The fix is to
prefer the observed fact over the recorded guess.

Contract under test: when the provider has told us which model it is
running, that name is what gets reported — never a value inferred from
a provider the agent is no longer on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from deep_worker_cli.cli.agent_command import (
    apply_preferred_provider_override,
    build_launch_config,
    create_parser,
)


def _first_launch_args(argv: list[str]) -> argparse.Namespace:
    """Resolve ``argv`` exactly as a FIRST launch does.

    ``main()`` normalises the parsed namespace (``normalize_model_arg``,
    the ``-pp`` provider override) before any launch path reads it, and
    there is no saved config, so ``merge_restart_config`` never runs.
    ``sys.argv`` is set because the override consults it to tell an
    explicit ``--provider`` from the parser default.
    """
    with patch.object(sys, "argv", ["deep-worker", *argv]):
        args = create_parser().parse_args(argv)
        apply_preferred_provider_override(args)
    return args


def _args_as_main_normalizes_them(argv: list[str]) -> argparse.Namespace:
    """Run the real ``main()`` and capture the namespace it produced.

    Everything above exercises the normalisation helpers directly, which
    proves the helpers but NOT that ``main()`` still calls them. Deleting
    either call site left the whole ``tests/unit_tests/cli/`` suite green
    (506 passed) — the fix could be unhooked from the entry point and no
    test would notice.

    ``--list`` is the cheapest real path through ``main()``: it is handled
    a few lines after the normalisation block and before ``daemonize()``,
    so the command handler receives exactly the namespace every launch
    path would receive. Spying on that handler reads the wiring rather
    than re-deriving it.
    """
    from deep_worker_cli.cli import agent_command as ac

    captured: dict[str, Any] = {}

    def _spy(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    with (
        patch.object(sys, "argv", ["deep-worker", *argv]),
        patch.object(ac, "list_threads_command", _spy),
        # ``main`` wraps the handler in ``asyncio.run``; the spy is sync.
        patch.object(ac.asyncio, "run", lambda coro: coro),
    ):
        rc = ac.main()

    assert rc == 0, f"main() exited {rc}"
    assert "args" in captured, "the --list handler was never reached"
    return captured["args"]


class TestMainWiresUpTheNormalisers:
    """The entry point must actually apply the fixes.

    One test per normaliser, both reached through the real ``main()``.
    ``normalize_model_arg`` is covered here for the same reason: it is the
    immediate neighbour, it has the same wiring gap, and removing ITS call
    site was likewise invisible to the suite.
    """

    def test_main_applies_the_preferred_provider_override(self) -> None:
        args = _args_as_main_normalizes_them(["--list", "-pp", "codex"])
        assert args.provider == "codex", (
            "main() did not apply apply_preferred_provider_override — the "
            "first-launch -pp fix is unhooked from the entry point"
        )

    def test_main_heals_the_poisoned_model_sentinel(self) -> None:
        args = _args_as_main_normalizes_them(
            ["--list", "--model", "unknown"]
        )
        assert args.model is None, (
            "main() did not apply normalize_model_arg — the legacy "
            '"unknown" poison reaches the launch path'
        )

    def test_main_leaves_an_ordinary_invocation_alone(self) -> None:
        """Neither normaliser may disturb a launch that needs no repair."""
        args = _args_as_main_normalizes_them(
            ["--list", "--provider", "grok", "--model", "grok-4"]
        )
        assert (args.provider, args.model) == ("grok", "grok-4")


class TestPreferredProviderOnFirstLaunch:
    """DEFECT 1: ``-pp`` must select the provider on first launch too."""

    def test_pp_selects_provider_on_first_launch(self) -> None:
        args = _first_launch_args(["a1", "-pp", "codex", "--dir", "/tmp"])
        cfg = build_launch_config(args)
        assert cfg["provider"] == "codex", (
            "-pp must set the ACTIVE provider on a first launch; got "
            f"{cfg['provider']!r} (the parser default silently won)"
        )

    def test_long_form_selects_provider_on_first_launch(self) -> None:
        args = _first_launch_args(
            ["a1", "--preferred-provider", "codex", "--dir", "/tmp"]
        )
        cfg = build_launch_config(args)
        assert cfg["provider"] == "codex"

    def test_pp_does_not_persist_a_foreign_default_model(self) -> None:
        """The model must not be the *other* provider's default.

        This is the visible symptom: -pp lands on deepseek AND writes
        deepseek's default model, so the UI names a model from a
        provider the operator never asked for.
        """
        args = _first_launch_args(["a1", "-pp", "codex", "--dir", "/tmp"])
        cfg = build_launch_config(args)
        assert cfg["model"] != "deepseek-v4-flash", (
            "persisted the default model of the WRONG provider"
        )

    def test_explicit_provider_still_wins_over_pp(self) -> None:
        """--provider is the active-provider flag; -pp must not fight it."""
        args = _first_launch_args(
            ["a1", "--provider", "codex", "-pp", "codex", "--dir", "/tmp"]
        )
        cfg = build_launch_config(args)
        assert cfg["provider"] == "codex"

    def test_no_pp_keeps_the_parser_default(self) -> None:
        """Absent -pp, nothing changes — the default provider still applies."""
        args = _first_launch_args(["a1", "--dir", "/tmp"])
        cfg = build_launch_config(args)
        assert cfg["provider"] == "deepseek"

    def test_pp_is_recorded_as_the_preference(self) -> None:
        args = _first_launch_args(["a1", "-pp", "codex", "--dir", "/tmp"])
        cfg = build_launch_config(args)
        assert cfg["preferred_provider"] == "codex"

    def test_restart_is_left_to_merge_restart_config(self) -> None:
        """The override must NOT fire on the restart path.

        Under a temporary failover the agent runs on one provider while
        ``preferred_provider`` names another, and ``merge_restart_config``
        has already resolved which wins. Re-applying the override here
        would pull the agent back onto its preferred provider and undo the
        failover — the regression this guard exists to prevent.
        """
        args = argparse.Namespace(
            restart=True,
            provider="deepseek",           # where failover put us
            preferred_provider="claude-code-sdk",   # long-term preference
        )
        with patch.object(sys, "argv", ["deep-worker", "a", "--restart"]):
            apply_preferred_provider_override(args)
        assert args.provider == "deepseek", (
            "the override fired on a restart and undid the failover"
        )

    def test_explicit_provider_beats_pp_even_when_it_is_the_default(
        self,
    ) -> None:
        """``--provider deepseek`` is a choice, not an absence of one.

        Presence is read from argv precisely so that naming the default
        provider outright still counts as selecting it.
        """
        args = _first_launch_args(
            ["a1", "--provider", "deepseek", "-pp", "codex", "--dir", "/tmp"]
        )
        assert args.provider == "deepseek"


class _FakeConfig:
    """Stand-in for AgentConfig with the two model sources separated."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        dynamic: str | None,
        resolved: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.working_dir = Path("/tmp")
        self._dynamic = dynamic
        self._resolved = resolved if resolved is not None else model

    @property
    def dynamic_display_model(self) -> str | None:
        return self._dynamic

    @property
    def resolved_model(self) -> str | None:
        return self._resolved


def _report(config: Any, launch_config: dict | None = None) -> str | None:
    """The metadata model-resolution rule, exercised directly.

    Mirrors ``AgentRpycService.exposed_get_metadata``'s selection so the
    ordering contract is testable without standing up an RPyC service.
    """
    from deep_worker_cli.rpyc_service.agent_service import (
        select_reported_model,
    )

    return select_reported_model(config, launch_config)


class TestObservedModelOutranksRecordedGuess:
    """DEFECT 2: the provider's own answer beats the launch-time guess."""

    def test_live_model_wins_over_stale_launch_config(self) -> None:
        """The operator's exact report: failover left a foreign model."""
        config = _FakeConfig(
            provider="codex",
            model="z-ai/glm-5.2:free",      # written at launch, now stale
            dynamic="gpt-5.5-codex",        # what codex actually reported
        )
        assert _report(config) == "gpt-5.5-codex", (
            "the stale launch_config model outranked the live provider "
            "answer — this is the 'or-nemotron-ultra / z-ai/glm-5.2:free' bug"
        )

    def test_launch_config_used_when_provider_said_nothing(self) -> None:
        """No live answer yet ⇒ the recorded model is the best we have."""
        config = _FakeConfig(
            provider="codex", model="gpt-5.5", dynamic=None
        )
        assert _report(config) == "gpt-5.5"

    def test_matching_values_are_stable(self) -> None:
        config = _FakeConfig(
            provider="codex", model="gpt-5.5", dynamic="gpt-5.5"
        )
        assert _report(config) == "gpt-5.5"

    def test_foreign_recorded_model_is_not_reported(self) -> None:
        """A model from a provider we are no longer on must never surface.

        With no live answer the stale value is all that is left; naming it
        would be a guess presented as fact, so nothing is reported and the
        caller falls through to its own (token-stats) source.
        """
        config = _FakeConfig(
            provider="grok",
            model="deepseek-v4-flash",   # impossible for grok
            dynamic=None,
            resolved=None,
        )
        assert _report(config) != "deepseek-v4-flash"

    def test_unpinned_agent_reports_what_the_provider_resolved(self) -> None:
        """No --model at launch: the live answer is the ONLY true source."""
        config = _FakeConfig(
            provider="codex",
            model=None,
            dynamic="gpt-5.5-codex",
            resolved=None,
        )
        assert _report(config) == "gpt-5.5-codex"

    def test_nothing_known_reports_nothing(self) -> None:
        """Never invent a name — return None and let the caller decide."""
        config = _FakeConfig(
            provider="codex", model=None, dynamic=None, resolved=None
        )
        assert _report(config) is None

    def test_model_from_a_provider_we_left_is_not_reported(self) -> None:
        """The operator's exact shape, before the provider has spoken.

        The pair reported on 2026-09-01 was a config-declared provider
        (``or-nemotron-ultra``) still showing ``z-ai/glm-5.2:free``. For
        such providers ``model_can_belong_to_provider`` accepts ANY model
        by design (the vendor validates, not us), so only
        ``model_provider`` reveals that the recorded model was chosen for
        a provider the agent has since left.

        ``codex`` stands in for the declared provider here: its branch of
        the predicate likewise accepts this model
        (``"z-ai/glm-5.2:free"`` contains none of the foreign markers), so
        the guard is the ONLY thing that can suppress it. Using a
        config-declared name instead would make the test pass for the
        wrong reason — this suite isolates ``~/.deep_worker``, so
        ``load_provider_specs()`` is empty and such a provider is rejected
        by the predicate before the guard is ever consulted.
        """
        from deep_worker_cli.config import model_can_belong_to_provider

        assert model_can_belong_to_provider("codex", "z-ai/glm-5.2:free"), (
            "premise broken: the predicate must accept this pair, "
            "otherwise the guard is not what makes this test pass"
        )

        config = _FakeConfig(
            provider="codex",
            model="z-ai/glm-5.2:free",
            dynamic=None,
        )
        launch_config = {"model_provider": "z-ai"}
        assert _report(config, launch_config) is None, (
            "reported a model belonging to a provider the agent has left"
        )

    def test_matching_model_provider_is_still_reported(self) -> None:
        """The guard must not fire when the record is self-consistent."""
        config = _FakeConfig(
            provider="codex", model="gpt-5.5", dynamic=None
        )
        launch_config = {"model_provider": "codex"}
        assert _report(config, launch_config) == "gpt-5.5"

    def test_observed_model_survives_a_provider_mismatch(self) -> None:
        """A live answer outranks the mismatch guard.

        The guard exists to suppress a stale *guess*; it must never
        suppress what the provider itself just reported.
        """
        config = _FakeConfig(
            provider="codex",
            model="z-ai/glm-5.2:free",
            dynamic="gpt-5.5-codex",
        )
        launch_config = {"model_provider": "z-ai"}
        assert _report(config, launch_config) == "gpt-5.5-codex"

    def test_absent_launch_config_falls_back_cleanly(self) -> None:
        """No launch_config ⇒ the guard is simply not applied."""
        config = _FakeConfig(
            provider="codex", model="gpt-5.5", dynamic=None
        )
        assert _report(config, None) == "gpt-5.5"

    def test_unknown_is_never_reported_as_a_model(self) -> None:
        """The legacy poison sentinel is not a model name."""
        config = _FakeConfig(
            provider="codex",
            model="unknown",
            dynamic=None,
            resolved="unknown",
        )
        assert _report(config) != "unknown"

