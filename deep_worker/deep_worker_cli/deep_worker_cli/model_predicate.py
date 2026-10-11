"""Provider↔model compatibility predicate — the single source of truth.

This module is deliberately a **leaf**: it imports nothing from the rest of
``deep_worker_cli`` and nothing heavy from the outside world. That property is
load-bearing. The web layer needs this predicate but cannot import
``deep_worker_cli.config``, which drags in langchain and friends; before this
module existed the web layer kept a hand-copied duplicate, and the two copies
drifted apart. The duplicate is what let the stale Claude Code whitelist
survive in two places at once.

Past incident: ``deep-worker --restart <agent> --model fable`` booted the
agent on ``sonnet``. The predicate's claude-code-sdk
branch only accepted ``sonnet``/``haiku``/``opus``/``inherit`` plus any string
containing ``claude``, so the documented alias ``fable`` was rejected, the
model was nulled, and the hardcoded default took over — silently.

The authoritative alias list is the "Model aliases" table in
https://code.claude.com/docs/en/model-config.
"""

from __future__ import annotations

# Documented Claude Code short aliases.
# Source: code.claude.com/docs/en/model-config §"Model aliases".
# ``fable`` requires Claude Code >= v2.1.170.
CLAUDE_CODE_SHORT_ALIASES: frozenset[str] = frozenset(
    {
        "default",
        "best",
        "fable",
        "sonnet",
        "opus",
        "haiku",
        "sonnet[1m]",
        "opus[1m]",
        "opusplan",
        # Legacy internal alias: older deep-worker builds persisted this.
        # Kept so existing session_metadata.json files stay loadable.
        "inherit",
    }
)

# Model tokens that belong to some *other* vendor. Used by the
# openai/codex branch, which is defined by exclusion rather than by an
# enumerable list.
_FOREIGN_MARKERS: tuple[str, ...] = (
    "claude",
    "sonnet",
    "haiku",
    "opus",
    "fable",
    "grok",
    "deepseek",
    "gemini",
)

# The legacy poison sentinel. ``"unknown"`` was once written into
# session_metadata.json as a stand-in for "no model"; forwarding it to a
# provider CLI produced an HTTP 400 (observed in a past incident). It is
# never a real model for any provider.
_POISON_SENTINEL = "unknown"


def is_claude_code_model(model_name: str) -> bool:
    """Return True if ``model_name`` is a legitimate Claude Code selector.

    Accepts the documented short aliases and any full ``claude-*`` model ID.
    The full-ID check also covers ``[1m]`` context-window variants such as
    ``claude-sonnet-5[1m]`` — no separate branch is needed for the suffix.

    Args:
        model_name: An already-normalised (stripped, lower-cased) token.
    """
    if model_name in CLAUDE_CODE_SHORT_ALIASES:
        return True
    return model_name.startswith("claude-")


def model_can_belong_to_provider(provider: str | None, model: str | None) -> bool:
    """Return True if ``model`` is a plausible model for ``provider``.

    This is a *compatibility* check, not an existence check: we cannot know
    which models a given account can actually reach, so the goal is to catch
    cross-provider contamination (a grok model surviving a switch to codex)
    and the legacy ``"unknown"`` poison, while never rejecting a token the
    provider's own CLI would have accepted.

    Rejecting a valid token is the more damaging error — it silently
    substitutes a different model — so the claude-code-sdk branch errs toward
    accepting anything the documented alias table or the ``claude-*``
    namespace covers.

    Args:
        provider: Provider name, e.g. ``"claude-code-sdk"``.
        model: Model name or alias, e.g. ``"fable"``.

    Returns:
        True if the pairing is plausible, False otherwise. Non-strings,
        blanks and ``"unknown"`` always return False.
    """
    if not isinstance(provider, str) or not provider.strip():
        return False
    if not isinstance(model, str) or not model.strip():
        return False

    provider_name = provider.strip().lower()
    model_name = model.strip().lower()

    if model_name == _POISON_SENTINEL:
        return False

    if provider_name in ("claude-code-sdk", "claude-code"):
        return is_claude_code_model(model_name)
    if provider_name == "anthropic":
        return "claude" in model_name
    if provider_name in ("google", "vertexai", "gemini"):
        return "gemini" in model_name
    if provider_name == "deepseek":
        return "deepseek" in model_name
    if provider_name == "grok":
        return "grok" in model_name
    if provider_name in ("openai", "codex"):
        return not any(marker in model_name for marker in _FOREIGN_MARKERS)

    # Config-declared API providers (providers.json). Unknown
    # to this leaf module by design, so consult the registry lazily. A
    # provider the operator declared owns whatever model they name for it:
    # the vendor validates the model, not us. Import is deferred and
    # failure-tolerant so this module stays dependency-free for the web
    # layer.
    return _is_config_provider(provider_name)


def _is_config_provider(provider_name: str) -> bool:
    """True when ``provider_name`` is declared in providers.json."""
    try:
        from deep_worker_cli.runtime.provider_registry import load_provider_specs

        return any(spec.name == provider_name for spec in load_provider_specs())
    except Exception:  # noqa: BLE001 — predicate must never raise
        return False
