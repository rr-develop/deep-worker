"""Provider-imposed context-window ceilings.

See ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md`` §3.4.

Every LLM provider has a per-model maximum input token count. The
operator's ``-l`` flag is a HIGH cap (the limit we promise to keep
under); the provider's cap is a LOW cap (the API will 400 if we
exceed it). The guard's pre-call check protects the HIGH cap; this
module protects the LOW cap at boot time.

Source of truth — API lookup, NO assumptions
---------------------------------------------
Per the operator's 2026-08-19 contract:

> "They must be read from the API. I do not want false values!!!!
> The context size can change at any time with the release of a new
> model in the family — that does not give you the right to use old
> values or old models of that family. A family always means the
> latest available model of that family."

Values come from APIs, never from hand-curated hardcodes:

  1. Dated model IDs (``claude-haiku-4-5``, ``gpt-4`` …) are looked
     up in the LangChain SDK's ``data._profiles._PROFILES`` dict.
     The SDK maintainers ship these dicts on every release with
     current values — the right API for this purpose.

  2. Family aliases (``fable``, ``opus``, ``sonnet``, ``haiku``,
     ``mythos``) on ``claude-code`` / ``claude-code-sdk`` are
     resolved through the ACP ``session/new`` PICKER. The Claude
     Code CLI returns a picker entry per supported model with the
     actual model id in the description (e.g. ``REPLACES
     claude-fable-5[1m]``). We parse that to get the resolved
     dated id, then look it up in the SDK.

  3. For providers that don't expose a picker (direct Anthropic
     API, OpenAI, DeepSeek …) family aliases are NOT resolved.
     The CLI cannot know what the current generation is for that
     family under that provider — the picker is the only API
     that exposes this. Silent passthrough.

  4. If the picker lookup fails for ANY reason (adapter missing,
     timeout, REPLACES marker absent, parser error) — silent
     passthrough. NO assumption-based fallback. Per the operator:
     "If you don't know and can't get it via the API, better not
     do any cap at all besides the operator's -l."

Cached lookup: the picker spawn is process-level expensive (one
subprocess spawn per call pre-cache). The cache is per-(provider,
model) and module-level. Tests can override the picker via
``set_picker_lookup_for_testing``.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Awaitable, Callable, Final


# ── SDK-shipped profile data ──────────────────────────────────────────────
#
# Thin wrapper over each provider's ``data._profiles._PROFILES`` dict
# so that a missing or broken SDK import fails LOUDLY at module
# import (rather than silently treating every model as unknown at
# runtime). The wrapping dict is the contract for the lookup helpers
# below.
def _load_anthropic_profiles() -> dict[str, int]:
    """Pull ``max_input_tokens`` from ``langchain_anthropic``'s
    shipped profile dict. ``{}`` if SDK is missing."""
    try:
        from langchain_anthropic.data import _profiles as _anthropic_profiles
    except ImportError:
        return {}
    return {
        model_id: profile["max_input_tokens"]
        for model_id, profile in _anthropic_profiles._PROFILES.items()
        if isinstance(profile.get("max_input_tokens"), int)
    }


def _load_openai_profiles() -> dict[str, int]:
    """Pull ``max_input_tokens`` from ``langchain_openai``'s
    shipped profile dict. ``{}`` if SDK is missing."""
    try:
        from langchain_openai.data import _profiles as _openai_profiles
    except ImportError:
        return {}
    return {
        model_id: profile["max_input_tokens"]
        for model_id, profile in _openai_profiles._PROFILES.items()
        if isinstance(profile.get("max_input_tokens"), int)
    }


def _load_gemini_profiles() -> dict[str, int]:
    """Pull ``max_input_tokens`` from ``langchain_google_genai``'s
    shipped profile dict. ``{}`` if SDK is missing."""
    try:
        from langchain_google_genai.data import (  # type: ignore[import-not-found]
            _profiles as _gemini_profiles,
        )
    except ImportError:
        return {}
    return {
        model_id: profile["max_input_tokens"]
        for model_id, profile in _gemini_profiles._PROFILES.items()
        if isinstance(profile.get("max_input_tokens"), int)
    }


_ANTHROPIC_MAX_INPUT: Final[dict[str, int]] = _load_anthropic_profiles()
_OPENAI_MAX_INPUT: Final[dict[str, int]] = _load_openai_profiles()
_GEMINI_MAX_INPUT: Final[dict[str, int]] = _load_gemini_profiles()


# Codex CLI override: GPT-5-Codex runs on the OpenAI API but the
# Codex CLI's effective cap is 400k (not the API's 1.05M). This is
# the ONE documented exception to the SDK-shipped data rule —
# because the SDK data is the API ceiling, not the CLI ceiling.
# Source: openai/codex issue tracker (CLI enforces 400k on
# /v1/chat/completions inside the CLI sandbox).
_CODEX_CLI_MAX_INPUT: Final[dict[str, int]] = {
    "gpt-5-codex": 400_000,
    "gpt-5.3-codex": 400_000,
    "gpt-5.5-codex": 400_000,
}


# ── Family aliases — NEVER statically resolved ────────────────────────────
#
# The set of family aliases that this CLI ACKNOWLEDGES. For
# ``claude-code`` / ``claude-code-sdk``, these are resolved through
# the ACP session/new PICKER (see :func:`resolve_provider_max`). For
# any other provider, these are NOT resolvable — silent
# passthrough.
_FAMILY_ALIASES: Final[frozenset[str]] = frozenset({
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "mythos",
})


# ── Picker description parser ─────────────────────────────────────────────
#
# The Claude Code CLI's picker entries for family aliases look like:
#
#   value='fable', name='Fable',
#   description='Fable (latest Claude Fable, 1M context, released 2026-06).
#                REPLACES claude-fable-5[1m]'
#
# We parse the description to extract the dated id that the alias
# currently resolves to. The ``REPLACES`` marker is the documented
# format from the adapter; if it's missing, we cannot resolve the
# alias and fall through to silent passthrough.
#
# This regex is intentionally conservative — it only matches the
# documented ``REPLACES <model_id>`` shape, with lowercase letters
# and digits/hyphens. Anything outside that shape is ignored
# (no false positives on free-text descriptions).
_VIA_PICKER_RE = re.compile(r"REPLACES\s+([a-z0-9][a-z0-9-]+)")


def _parse_picker_for_alias(reply: dict, alias: str) -> str | None:
    """Parse the picker reply to find the resolved dated id for ``alias``.

    Returns the resolved dated id (e.g. ``"claude-fable-5"``) or
    ``None`` if the picker doesn't have an entry for ``alias`` or
    the entry's description doesn't carry the ``REPLACES`` marker.
    """
    try:
        from deep_worker.providers.claude_code_provider.profile import (
            _picker_entry,
        )
    except ImportError:
        return None

    options, _ = _picker_entry(reply)
    if not options:
        return None
    for opt in options:
        if not isinstance(opt, dict):
            continue
        if opt.get("value") != alias:
            continue
        desc = opt.get("description")
        if not isinstance(desc, str):
            return None
        m = _VIA_PICKER_RE.search(desc)
        if m:
            return m.group(1)
    return None


# ── Picker lookup (the API call) ─────────────────────────────────────────
#
# The default picker lookup spawns a transient AcpClient and reads
# the session/new reply. This is the "API" the operator asked for.
#
# Tests override this via ``set_picker_lookup_for_testing`` to avoid
# spawning a real subprocess. The function returns the full reply
# dict (which may include the picker) or ``None`` on any failure.
# All failures are silent per the operator's contract.

_PickerLookup = Callable[[str], Awaitable[dict | None]]

# Module-level injector. ``None`` → use the default AcpClient spawn.
_picker_lookup_override: _PickerLookup | None = None


def set_picker_lookup_for_testing(lookup: _PickerLookup | None) -> None:
    """Inject a picker lookup for tests. ``None`` restores the default.

    The lookup takes the family alias (e.g. ``"fable"``) and returns
    the full ``session/new`` reply dict (or ``None`` on failure).

    Tests use this to inject a canned reply without spawning a real
    AcpClient / Claude Code CLI subprocess.
    """
    global _picker_lookup_override
    _picker_lookup_override = lookup


async def _default_picker_lookup(alias: str) -> dict | None:
    """Default picker lookup: spawn a transient AcpClient.

    Spawns the Claude Code CLI subprocess, calls ``session/new``,
    returns the full reply dict. Returns ``None`` on any failure
    (adapter missing, subprocess crashed, timeout, etc.) — per the
    operator's contract, NO cap means we don't cap.

    The CLI is the only ACP API that exposes the picker for family
    aliases. The direct Anthropic API does not — for operators
    using ``claude-code`` / ``claude-code-sdk`` without the CLI
    installed, family aliases fall through to silent passthrough.
    """
    # The AcpClient import is heavy (transitive imports of the
    # ACP module). Lazy import so a pure-Python ``import provider_max``
    # does not pull in the Claude Code subprocess infrastructure.
    try:
        from deep_worker.providers.acp.acp_client import AcpClient
        from deep_worker.providers.claude_code_provider.profile import (
            ADAPTER_ENTRY,
        )
    except ImportError:
        return None

    if not ADAPTER_ENTRY.is_file():
        return None

    client: Any = None
    try:
        client = AcpClient(
            cmd=("node", str(ADAPTER_ENTRY)),
            client_name="deep-worker-resolver",
            client_version="0.0.0",
        )
        await client.start()
        # Full ``session/new`` handshake (initialize + authenticate
        # already done by ``start()``). 30s timeout is generous for
        # the picker round-trip on a cold CLI.
        reply = await client._call(
            "session/new",
            {"cwd": "/tmp", "mcpServers": []},
            timeout=30.0,
        )
        return reply
    except Exception:
        return None
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


# ── Module-level resolve cache ────────────────────────────────────────────
#
# The picker lookup is expensive (one subprocess spawn). The
# resolved value is stable for the lifetime of the agent process
# (no model swap mid-run), so cache per (provider, model). Tests
# that need a clean slate call :func:`clear_resolve_cache`.
_RESOLVE_CACHE: dict[tuple[str, str], int | None] = {}


def clear_resolve_cache() -> None:
    """Clear the module-level resolve cache. Test-only."""
    _RESOLVE_CACHE.clear()


# ── Public API ────────────────────────────────────────────────────────────


async def resolve_provider_max(provider: str, model: str) -> int | None:
    """Resolve the provider-max context window via API lookup.

    Resolution rules (per operator contract 2026-08-19):

    1. ``claude-code`` / ``claude-code-sdk`` + family alias (fable,
       opus, sonnet, haiku, mythos): call the picker via
       ``_default_picker_lookup`` (or the test override),
       parse the description for the resolved dated id, then
       look up in the SDK profile data.

    2. ``claude-code`` / ``claude-code-sdk`` + dated id: direct
       SDK profile lookup.

    3. Other providers + any model: direct SDK profile lookup
       (no family-alias resolution — the picker is Anthropic-only).

    Returns ``None`` on any failure. Per the operator contract:
    "If you don't know and can't get it via the API, better not
    do any cap at all besides the operator's -l."

    The result is cached per (provider, model) for the lifetime
    of the agent process.
    """
    cache_key = (provider, model)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]

    result = await _resolve_provider_max_impl(provider, model)
    _RESOLVE_CACHE[cache_key] = result
    return result


async def _resolve_provider_max_impl(provider: str, model: str) -> int | None:
    if provider in ("claude-code", "claude-code-sdk"):
        if model in _FAMILY_ALIASES:
            # Family alias → picker lookup.
            lookup = _picker_lookup_override or _default_picker_lookup
            try:
                reply = await lookup(model)
            except Exception:
                # Picker crashed (subprocess error, timeout, etc.).
                # Silent passthrough per operator contract.
                return None
            if reply is None:
                # Picker unavailable — silent passthrough.
                return None
            resolved_id = _parse_picker_for_alias(reply, model)
            if resolved_id is None:
                # Picker doesn't have the alias (or no REPLACES marker).
                # Silent passthrough.
                return None
            return _ANTHROPIC_MAX_INPUT.get(resolved_id)
        # Direct lookup for dated ids.
        return _ANTHROPIC_MAX_INPUT.get(model)

    if provider == "openai":
        return _OPENAI_MAX_INPUT.get(model)
    if provider == "gemini":
        return _GEMINI_MAX_INPUT.get(model)
    if provider == "codex":
        return _CODEX_CLI_MAX_INPUT.get(model) or _OPENAI_MAX_INPUT.get(model)
    return None


# ── Sync SDK-only helpers (used by non-async callers) ─────────────────────
#
# These are the SYNCHRONOUS path — they only consult the SDK
# profile data, NEVER the picker. For family aliases on
# claude-code they return ``None`` (silent passthrough); the
# async path above is the way to resolve those.
#
# Kept for backward compatibility and for callers that don't have
# an async context (e.g. unit tests asserting direct SDK lookup).


def provider_max_for(provider: str, model: str) -> int | None:
    """Sync SDK-only lookup. Returns ``None`` for family aliases.

    For family aliases on ``claude-code`` / ``claude-code-sdk``,
    this returns ``None`` because the resolved dated id is not
    statically known — use :func:`resolve_provider_max` (async)
    to do the picker lookup.

    For everything else, it returns the SDK profile data value
    or ``None`` if the model isn't in the SDK.
    """
    if model in _FAMILY_ALIASES:
        # Family aliases are NOT resolvable statically — the
        # current generation's id changes with every Anthropic
        # release. Static lookup would silently under-clamp the
        # moment a new generation ships. Use the async resolver
        # to get the actual current value via the picker.
        return None
    if provider in ("claude-code", "claude-code-sdk"):
        return _ANTHROPIC_MAX_INPUT.get(model)
    if provider == "openai":
        return _OPENAI_MAX_INPUT.get(model)
    if provider == "gemini":
        return _GEMINI_MAX_INPUT.get(model)
    if provider == "codex":
        return _CODEX_CLI_MAX_INPUT.get(model) or _OPENAI_MAX_INPUT.get(model)
    return None


def clamp_to_provider_max(
    requested: int | None,
    provider: str,
    model: str,
) -> int | None:
    """Sync clamp. Silent passthrough for family aliases (use
    :func:`resolve_provider_max` for those)."""
    if requested is None:
        return None
    max_input = provider_max_for(provider, model)
    if max_input is None:
        return requested
    return min(requested, max_input)


def warn_if_above_provider_max(
    requested: int | None,
    provider: str,
    model: str,
) -> None:
    """Sync WARN. Silent for family aliases (use the async
    resolver + clamping path for those)."""
    if requested is None:
        return
    max_input = provider_max_for(provider, model)
    if max_input is None:
        return
    if requested > max_input:
        print(
            f"[context-limit] WARN: -l {requested:,} exceeds provider cap "
            f"({provider}/{model} = {max_input:,}). "
            f"Auto-clamping to provider max.",
            file=sys.stderr,
        )
