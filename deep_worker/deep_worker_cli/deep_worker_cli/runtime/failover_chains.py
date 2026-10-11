# Named, operator-configurable provider failover chains.
#
# DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md. Owner directive 2026-09-03: a chain
# is NEVER hard-coded in Python. This module reads
# ``~/.deep_worker/failover_chains.json`` and is the single runtime source of
# failover order.
#
# The one Python literal here is _DEFAULT_SEED: it initialises the config file
# the first time it is needed and is never consulted afterwards. The runtime
# path always goes through the file, so an operator who edits ``default``
# changes behaviour — which is the whole point of the directive.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_worker_cli.storage_paths import deep_worker_home

_logger = logging.getLogger(__name__)

CHAINS_FILE_NAME = "failover_chains.json"

DEFAULT_CHAIN_NAME = "default"

# One-shot seed for materialising the config file (design §3.3). NOT a runtime
# fallback: once the file exists this value is dead. ``claude-code-sdk`` is
# deliberately absent — it and ``claude-code`` are two forms of one provider on
# one subscription, so a step between them lands in the same exhausted account
# (owner directive 2026-09-03).
_DEFAULT_SEED: tuple[str, ...] = ("claude-code", "codex", "grok", "deepseek")

_SEED_DESCRIPTION = (
    "Quality-first order across subscription accounts. "
    "Edit freely: this file is the only source of failover order."
)


class FailoverChainConfigError(RuntimeError):
    """``failover_chains.json`` is malformed or names an unknown provider.

    Raised at load time with an actionable message naming the chain and the
    offending step, so the operator can fix the file without reading code.
    """


@dataclass(frozen=True)
class FailoverStep:
    """One position in a chain: a provider, optionally pinned to a model.

    ``model`` may be an exact id (``claude-opus-5``) or a family
    (``opus``). Family resolution happens at failover time against the
    provider's own profile — never against a table in this module
    (feedback_no_model_hardcode_2026_08_15).
    """

    provider: str
    model: str | None = None


@dataclass(frozen=True)
class FailoverChain:
    """A named, ordered list of failover steps."""

    name: str
    steps: tuple[FailoverStep, ...]
    description: str | None = None

    def index_of(self, provider: str, model: str | None = None) -> int | None:
        """Position of the step the agent currently occupies, or ``None``.

        Matching prefers the exact ``(provider, model)`` pair so a chain that
        lists one provider at several model tiers advances correctly. When the
        agent's model is unknown or matches no step, the first step naming the
        provider wins — a chain position is better than no position.

        ``None`` means the provider is absent from this chain entirely; see
        :meth:`candidates_after` for what that implies.
        """
        if model:
            for i, step in enumerate(self.steps):
                if step.provider == provider and step.model == model:
                    return i
        for i, step in enumerate(self.steps):
            if step.provider == provider:
                return i
        return None

    def candidates_after(
        self, provider: str, model: str | None = None
    ) -> tuple[FailoverStep, ...]:
        """Steps to try after ``provider`` exhausted its account.

        Two cases, deliberately different (design §4.5):

        * The provider IS in this chain — return the steps strictly after it,
          the classic contract.
        * The provider is NOT in this chain — return the WHOLE chain. This is
          the fix for the config-driven dead end: before chains were
          configurable, an off-chain provider got ``[]`` and the agent could
          neither leave it nor be sent to it, so a 429 stranded it in ERROR
          forever.

        Account-group filtering is NOT applied here. It is availability logic,
        not composition policy: the caller consults the tracker, which already
        keys verdicts by account. Filtering here would silently drop steps the
        operator wrote, which the owner has forbidden.
        """
        idx = self.index_of(provider, model)
        if idx is None:
            return self.steps
        return self.steps[idx + 1:]


def chains_path(home: Path | None = None) -> Path:
    """Absolute path of ``failover_chains.json`` in the runtime tree."""
    return deep_worker_home(home) / CHAINS_FILE_NAME


def _seed_document() -> dict[str, Any]:
    return {
        "version": 1,
        "_comment": (
            "Provider failover chains. See "
            "deep_worker/deep_worker_cli/docs/"
            "DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md. A step is "
            '{"provider": "<name>"} with an optional "model" (exact id or '
            "family, e.g. \"opus\"). Any provider from providers.json may be "
            "used."
        ),
        "chains": {
            DEFAULT_CHAIN_NAME: {
                "description": _SEED_DESCRIPTION,
                "steps": [{"provider": name} for name in _DEFAULT_SEED],
            }
        },
    }


def materialise_default(home: Path | None = None) -> Path:
    """Create ``failover_chains.json`` with the seeded ``default`` chain.

    No-op when the file already exists — the operator owns it from then on,
    including the right to redefine ``default`` itself. Written atomically so
    a concurrent reader never sees a partial file.
    """
    path = chains_path(home)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_seed_document(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    _logger.info(f"[FAILOVER] Created {path} with the seeded '{DEFAULT_CHAIN_NAME}' chain")
    return path


def _parse_step(raw: Any, chain_name: str, index: int) -> FailoverStep:
    if isinstance(raw, str):
        # Sugar: a bare provider name. Chains are hand-edited and the common
        # case carries no model.
        provider = raw.strip()
        model = None
    elif isinstance(raw, dict):
        provider = str(raw.get("provider", "")).strip()
        raw_model = raw.get("model")
        model = str(raw_model).strip() if raw_model is not None else None
        if model == "":
            model = None
    else:
        raise FailoverChainConfigError(
            f"{CHAINS_FILE_NAME}: chain {chain_name!r} step #{index} must be a "
            f"provider name or an object with a 'provider' field, got "
            f"{type(raw).__name__}."
        )

    if not provider:
        raise FailoverChainConfigError(
            f"{CHAINS_FILE_NAME}: chain {chain_name!r} step #{index} has no "
            f"'provider'."
        )
    return FailoverStep(provider=provider, model=model)


def _parse_chain(name: str, raw: Any) -> FailoverChain:
    if isinstance(raw, list):
        raw_steps: Any = raw
        description = None
    elif isinstance(raw, dict):
        raw_steps = raw.get("steps")
        raw_desc = raw.get("description")
        description = str(raw_desc) if raw_desc is not None else None
    else:
        raise FailoverChainConfigError(
            f"{CHAINS_FILE_NAME}: chain {name!r} must be a list of steps or an "
            f"object with a 'steps' field, got {type(raw).__name__}."
        )

    if not isinstance(raw_steps, list) or not raw_steps:
        raise FailoverChainConfigError(
            f"{CHAINS_FILE_NAME}: chain {name!r} must declare a non-empty "
            f"'steps' list."
        )

    steps = tuple(
        _parse_step(entry, name, i) for i, entry in enumerate(raw_steps)
    )
    return FailoverChain(name=name, steps=steps, description=description)


def _validate_providers(
    chains: dict[str, FailoverChain], home: Path | None = None
) -> None:
    """Every step must name a provider that actually exists.

    This is the ONLY membership rule (design §4.3, owner directive): no
    ordering constraints, no account-group restrictions, no ban on
    config-driven providers. We check existence purely so a chain cannot send
    an agent to an endpoint that was never configured.

    Credentials are NOT checked here — a missing env var must not stop the CLI
    from starting. That check happens at failover time, where it can actually
    prevent a doomed restart.
    """
    from deep_worker_cli.runtime.provider_chain import (  # noqa: PLC0415
        all_provider_names,
    )

    known = set(all_provider_names(home))
    for chain in chains.values():
        for i, step in enumerate(chain.steps):
            if step.provider not in known:
                raise FailoverChainConfigError(
                    f"{CHAINS_FILE_NAME}: chain {chain.name!r} step #{i} names "
                    f"unknown provider {step.provider!r}. Known providers: "
                    f"{', '.join(sorted(known))}. Add it to providers.json or "
                    f"fix the chain."
                )


def load_chains(home: Path | None = None) -> dict[str, FailoverChain]:
    """All configured chains, keyed by name.

    Materialises the file with the seeded ``default`` when it is absent, so a
    fresh install behaves exactly like the previous hard-coded chain.

    Raises :class:`FailoverChainConfigError` on malformed JSON, a malformed
    chain, an unknown provider, or a missing ``default``. A missing ``default``
    is an error rather than a silent fallback: an agent that asked for a chain
    by name must never be handed a different one.
    """
    path = chains_path(home)
    if not path.exists():
        materialise_default(home)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FailoverChainConfigError(
            f"{path}: invalid JSON ({exc}). Fix the file or delete it to have "
            f"the default chain regenerated."
        ) from exc
    except OSError as exc:
        raise FailoverChainConfigError(f"{path}: cannot be read ({exc}).") from exc

    if not isinstance(raw, dict):
        raise FailoverChainConfigError(
            f"{path}: top level must be an object with a 'chains' field."
        )

    raw_chains = raw.get("chains")
    if not isinstance(raw_chains, dict) or not raw_chains:
        raise FailoverChainConfigError(
            f"{path}: 'chains' must be a non-empty object mapping chain names "
            f"to step lists."
        )

    chains = {
        str(name): _parse_chain(str(name), body)
        for name, body in raw_chains.items()
    }

    if DEFAULT_CHAIN_NAME not in chains:
        raise FailoverChainConfigError(
            f"{path}: no {DEFAULT_CHAIN_NAME!r} chain. Every agent falls back "
            f"to it when none is named, so it must exist."
        )

    _validate_providers(chains, home)
    return chains


def get_chain(name: str | None = None, home: Path | None = None) -> FailoverChain:
    """The chain called ``name``, or ``default`` when ``name`` is falsy.

    Raises :class:`FailoverChainConfigError` when the name is unknown — an
    agent configured for a chain that no longer exists must fail loudly, not
    silently inherit someone else's failover order.
    """
    chains = load_chains(home)
    wanted = (name or DEFAULT_CHAIN_NAME).strip() or DEFAULT_CHAIN_NAME
    try:
        return chains[wanted]
    except KeyError:
        raise FailoverChainConfigError(
            f"{chains_path(home)}: no chain named {wanted!r}. Configured: "
            f"{', '.join(sorted(chains))}."
        ) from None


def resolve_step_model(step: FailoverStep) -> str | None:
    """The model id to launch ``step`` with, or ``None`` to let the CLI pick.

    ``step.model`` may be a family alias (``opus``) rather than a dated id.
    We do NOT resolve it here, and there is no family table in this module —
    model names belong to the backend (feedback_no_model_hardcode_2026_08_15).

    Passing an alias through verbatim is not a limitation but the correct
    behaviour for the providers where aliases occur: ``claude-code`` and
    ``claude-code-sdk`` accept ``--model opus`` and resolve it against their
    own picker, which is by definition the current model of that family. Every
    other provider gets the string unchanged and rejects a bad id with a real
    error from the vendor.

    Deliberately NOT used here: ``context_limit.provider_max`` can turn an
    alias into a dated id, but its default lookup spawns a transient CLI
    subprocess (``provider_max.py:226``). Failover runs while an agent is
    already failing; spawning a second process to learn a name we can simply
    forward would be both slow and a stranger's answer about our own launch
    (owner directive 2026-09-01). Context-window resolution has its own reason
    to pay that cost — choosing a launch model does not.
    """
    return step.model or None
