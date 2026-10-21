"""CLI-surface wiring for the claude-code provider.

These are REGRESSION tests for two defects that the 131 green ACP tests all
missed, because every one of them imported the provider packages directly and
so never exercised the paths a real ``deep-worker`` invocation takes:

1. ``--provider claude-code`` was rejected by argparse: the ``choices``
   list in ``agent_command`` was a hardcoded copy that nobody updated, even
   though ``AgentConfig`` validation, failover chain, token stats and the
   model factory all already knew the provider.

2. Startup then died with ``unknown provider package:
   'deep_worker.providers.claude_code_provider'`` — neither ACP package was
   registered in ``PROVIDER_DESCRIPTORS``, so ``resolve_provider()`` raised
   before the chat model could ever be constructed.

The lesson encoded here: assert the provider is reachable from the OUTERMOST
surface (argparse) all the way down to the provider-origin resolver, not just
that its module imports.
"""

import argparse

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.cli.agent_command import create_parser
from deep_worker_cli.provider_paths import (
    PROVIDER_DESCRIPTORS,
    resolve_provider,
)
from deep_worker_cli.runtime.provider_chain import (
    SUPPORTED_PROVIDERS,
    all_provider_names,
)

ACP_PROVIDER = "claude-code"

# Both packages the provider needs at runtime. ``acp`` is the transport layer
# and ``claude_code_provider`` the concrete profile/chat-model; a missing
# descriptor for EITHER one is a fatal startup error.
ACP_PACKAGES = (
    "deep_worker.providers.acp",
    "deep_worker.providers.claude_code_provider",
)


class TestArgparseAcceptsAcpProvider:
    """Defect 1: the CLI must actually accept the provider."""

    def test_provider_flag_accepts_claude_code_acp(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["ag", "--provider", ACP_PROVIDER])
        assert args.provider == ACP_PROVIDER

    def test_reproduces_original_failing_command(self) -> None:
        """The exact invocation from the bug report must parse."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "agent_g",
                "-sp",
                "--dir",
                "<repo>",
                "--provider",
                ACP_PROVIDER,
            ]
        )
        assert args.provider == ACP_PROVIDER
        assert args.skip_permissions is True

    def test_unknown_provider_still_rejected(self) -> None:
        """Widening the choices must not turn the flag into a free-text field."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ag", "--provider", "not-a-real-provider"])


class TestSingleSourceOfTruth:
    """The split that caused defect 1 must be structurally impossible now."""

    def test_argparse_choices_are_the_canonical_list(self) -> None:
        """Choices come from ``all_provider_names()`` — the built-in tuple
        PLUS any config-declared API provider — never a hand-maintained
        duplicate.

        Asserting equality with ``SUPPORTED_PROVIDERS`` alone would fail on
        any machine that actually has a ``~/.deep_worker/providers.json``
        (and would forbid the config-driven providers the owner asked for),
        so the invariants are: every built-in is offered, and the offered
        list is exactly what the single source of truth returns.
        """
        parser = create_parser()
        action = next(
            a
            for a in parser._actions  # noqa: SLF001 - inspecting parser wiring is the point
            if isinstance(a, argparse.Action) and a.dest == "provider"
        )
        choices = list(action.choices)
        assert choices == list(all_provider_names())
        # Built-ins must never be dropped by the config merge.
        for builtin in SUPPORTED_PROVIDERS:
            assert builtin in choices
        assert len(choices) == len(set(choices)), "duplicate provider choices"

    @pytest.mark.integration  # AgentConfig validates creds: needs GOOGLE_API_KEY
    def test_every_supported_provider_passes_config_validation(self) -> None:
        """argparse and AgentConfig must not disagree in EITHER direction."""
        for provider in SUPPORTED_PROVIDERS:
            cfg = AgentConfig(working_dir="/tmp", provider=provider)
            assert cfg.provider == provider

    def test_chain_steps_must_name_a_configured_provider(self, tmp_path) -> None:
        """Replaces "chain ⊆ SUPPORTED_PROVIDERS", which the owner revoked.

        That invariant barred config-driven providers from failover entirely —
        none of the ``providers.json`` names are in ``SUPPORTED_PROVIDERS``, so
        any chain containing one failed it. Chains may now name ANY configured
        provider (owner directive 2026-09-03).

        The rule that survives is weaker on membership but stronger on
        coverage: it validates the operator's actual config, which the old
        assertion never looked at.
        """
        import json

        import pytest

        from deep_worker_cli.runtime.failover_chains import (
            FailoverChainConfigError,
            chains_path,
            load_chains,
        )

        path = chains_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # A built-in that is NOT in the old chain constant is accepted.
        path.write_text(json.dumps({"chains": {"default": {"steps": ["openai"]}}}))
        assert [s.provider for s in load_chains(tmp_path)["default"].steps] == ["openai"]

        # A provider that exists nowhere is still rejected — the check must not
        # be relaxed into something vacuous.
        path.write_text(json.dumps({"chains": {"default": {"steps": ["no-such-provider"]}}}))
        with pytest.raises(FailoverChainConfigError, match="unknown provider"):
            load_chains(tmp_path)

    def test_supported_providers_has_no_duplicates(self) -> None:
        """A duplicate would show twice in ``--help`` and skew the choices list."""
        assert len(SUPPORTED_PROVIDERS) == len(set(SUPPORTED_PROVIDERS))

    def test_cli_presentation_order_is_stable(self) -> None:
        """Pins the documented contract: this tuple is a stable user-facing menu.

        It is intentionally NOT derived from any failover order (see the comment
        on ``SUPPORTED_PROVIDERS``). Retuning failover priority — now an edit to
        ``failover_chains.json``, not to Python — must not reshuffle ``--help``,
        so the order is asserted literally here; changing it is a deliberate UX
        decision that should fail this test first.
        """
        assert SUPPORTED_PROVIDERS == (
            "deepseek",
            "gemini",
            "openai",
            "claude-code-sdk",
            "claude-code",
            "codex",
            "grok",
        )

    def test_acp_provider_is_in_canonical_list(self) -> None:
        assert ACP_PROVIDER in SUPPORTED_PROVIDERS


class TestProviderPackagesAreResolvable:
    """Defect 2: runtime provider-origin resolution must know the ACP packages."""

    @pytest.mark.parametrize("pkg", ACP_PACKAGES)
    def test_package_has_a_descriptor(self, pkg: str) -> None:
        assert pkg in {d.pkg for d in PROVIDER_DESCRIPTORS}

    @pytest.mark.parametrize("pkg", ACP_PACKAGES)
    def test_resolve_provider_does_not_raise(self, pkg: str) -> None:
        """This is the exact call that raised ValueError during startup."""
        resolution = resolve_provider(pkg)
        assert resolution is not None

    def test_unregistered_package_still_raises(self) -> None:
        """The registry must stay a whitelist, not silently accept anything."""
        with pytest.raises(ValueError, match="unknown provider package"):
            resolve_provider("deep_worker.providers.definitely_not_a_provider")

    @pytest.mark.parametrize("pkg", ACP_PACKAGES)
    def test_descriptor_source_subpath_exists(self, pkg: str) -> None:
        """A descriptor pointing at a non-existent tree fails only at runtime."""
        from pathlib import Path

        descriptor = next(d for d in PROVIDER_DESCRIPTORS if d.pkg == pkg)
        # tests/acp_provider → tests → deep_worker_cli → deep_worker → repo root
        repo_root = Path(__file__).resolve().parents[4]
        assert (repo_root / descriptor.source_subpath).is_dir()
