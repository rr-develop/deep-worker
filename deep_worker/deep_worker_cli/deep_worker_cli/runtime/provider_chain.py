# Provider identity and the canonical provider menu.
#
# NOTE: failover ORDER does not live here any more. Chains are named and
# operator-configured in ``~/.deep_worker/failover_chains.json``; hard-coding a
# chain in Python is forbidden (owner directive 2026-09-03). See
# ``runtime/failover_chains.py`` and
# ``docs/DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md``.
#
# What remains here is deliberately NOT order: the stable ``--help`` menu of
# built-in providers, and the account identity a usage cap is recorded against.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-cycle-free: failover_chains imports this module
    from pathlib import Path

    from deep_worker_cli.runtime.failover_chains import FailoverChain, FailoverStep

# Canonical set of provider names accepted anywhere in the CLI.
#
# SINGLE SOURCE OF TRUTH. The argparse ``--provider`` choices and
# ``AgentConfig.__post_init__`` validation both derive from this tuple, so a
# newly added provider cannot be accepted by one layer and rejected by the
# other (that split is exactly how ``claude-code`` shipped unreachable
# from the command line while every downstream layer already supported it).
#
# Ordering is the historical CLI presentation order shown in ``--help``, with
# ``claude-code`` slotted next to its sibling ``claude-code-sdk``. It is
# DELIBERATELY NOT a failover order, and failover chains are NOT constrained to
# it: an operator may build a chain from any configured provider, including the
# config-driven ones in ``providers.json`` that never appear in this menu
# (owner directive 2026-09-03). The old invariant "chain ⊆ SUPPORTED_PROVIDERS"
# is gone with the chain constant; ``failover_chains`` validates steps against
# ``all_provider_names()`` instead, which is what "the provider exists" means.
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "deepseek",
    "gemini",
    "openai",
    "claude-code-sdk",
    "claude-code",
    "codex",
    "grok",
)

# Account affinity (DESIGN_ACP_PROVIDER_LAYER.md §7, review R2): a usage cap
# is ACCOUNT-level, not provider-wrapper-level. Failing over from
# ``claude-code`` to ``claude-code-sdk`` would re-hit the SAME Claude
# subscription immediately. Providers that share an account must therefore
# be skipped when the exhausted provider's account is the same.
PROVIDER_ACCOUNT_GROUPS: dict[str, str] = {
    "claude-code": "claude-subscription",
    "claude-code-sdk": "claude-subscription",
    "codex": "codex-account",
    "grok": "grok-account",
    "deepseek": "deepseek-account",
}


def account_group_for(provider: str, home: "Path | None" = None) -> str:
    """The billing account ``provider`` draws on.

    A usage cap applies to the ACCOUNT, not to the wrapper we happen to
    talk through, so this is the correct identity for anything that tracks
    "is this exhausted": ``claude-code`` and ``claude-code-sdk`` are two
    implementations of one Claude subscription and must share one verdict
    (owner directive 2026-08-17).

    Resolution order:

    1. ``PROVIDER_ACCOUNT_GROUPS`` — the built-in providers.
    2. ``account_group`` from ``providers.json`` — operator-declared groups
       for config-driven providers, so two entries pointing at the same
       vendor account can be grouped without a code change.
    3. The provider name itself — an unknown provider is its own account.
       Never a shared default: guessing "same account" would make one
       vendor's cap silently block an unrelated one.

    A malformed ``providers.json`` must not break limit tracking, so config
    errors degrade to step 3 rather than propagating.
    """
    builtin = PROVIDER_ACCOUNT_GROUPS.get(provider)
    if builtin is not None:
        return builtin

    try:
        from deep_worker_cli.runtime.provider_registry import (  # noqa: PLC0415
            load_provider_specs,
        )

        for spec in load_provider_specs(home):
            if spec.name == provider and spec.account_group:
                return spec.account_group
    except Exception:  # noqa: BLE001 — config problems must not break tracking
        pass

    return provider


def failover_candidates(
    after_provider: str,
    chain: "FailoverChain | None" = None,
    model: str | None = None,
    home: "Path | None" = None,
) -> list["FailoverStep"]:
    """Steps to consider after ``after_provider`` exhausted its account.

    The order comes from ``chain`` — an operator-configured, named chain. When
    ``chain`` is omitted the agent's ``default`` chain is loaded, so callers
    that do not care about naming keep working.

    Steps sharing ``after_provider``'s account group are EXCLUDED **unless the
    step pins a different model**. A usage cap is normally account-level, so
    retrying the same subscription fails again — but a cap can also be
    per-model, and an operator who wrote ``opus`` then ``sonnet`` is asking
    precisely for the second case (owner directive 2026-09-03: "a model's limit is
    not always the same as the account's limit"). Pinning a model is how they say so; a step
    with no model cannot be that, and stays filtered.

    This is availability logic, not composition policy — the operator's chain
    is never edited, only walked past.

    Unlike the previous constant-based implementation, a provider that is
    ABSENT from the chain no longer yields ``[]``. It now starts at the first
    step (see :meth:`FailoverChain.candidates_after`), which is what lets a
    config-driven provider fail over at all; returning ``[]`` stranded such an
    agent in ERROR permanently.

    Returns steps, not names: a step carries the model the operator pinned to
    that position, which the caller passes to ``--restart``.
    """
    from deep_worker_cli.runtime.failover_chains import (  # noqa: PLC0415
        get_chain,
    )

    resolved_chain = chain if chain is not None else get_chain(home=home)
    tail = resolved_chain.candidates_after(after_provider, model)

    group = account_group_for(after_provider, home)

    def _worth_trying(step: "FailoverStep") -> bool:
        if account_group_for(step.provider, home) != group:
            return True
        # Same account. Only a step pinned to a DIFFERENT model can still
        # succeed — that is an operator saying "the cap was per-model".
        return bool(step.model) and step.model != model

    return [step for step in tail if _worth_trying(step)]


def all_provider_names(home: "Path | None" = None) -> tuple[str, ...]:
    """Built-in providers plus any declared in ``~/.deep_worker/providers.json``.

    ``SUPPORTED_PROVIDERS`` above stays the stable, user-facing list of
    providers that ship with deep-worker. Operators can add standard API
    providers (OpenAI-compatible endpoints) purely through config, and those
    must be accepted everywhere a provider name is accepted — argparse
    ``--preferred-provider`` choices and ``AgentConfig`` validation — without
    any code change (owner directive 2026-08-15).

    ``home`` redirects the config root, so a caller validating chains against a
    test tree sees that tree's providers rather than the real ones.

    Config errors are NOT swallowed here: a malformed ``providers.json``
    raises ``ProviderConfigError`` so the operator gets an actionable message
    instead of a confusing "invalid choice" from argparse.
    """
    from deep_worker_cli.runtime.provider_registry import load_provider_specs

    extra = tuple(spec.name for spec in load_provider_specs(home))
    return SUPPORTED_PROVIDERS + extra
