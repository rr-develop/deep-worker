"""Tests for ``deep_worker_cli.context_limit.provider_max``.

Architecture under test (agent_a 2026-08-18 lesson, then refined):

  - Dated model IDs are sourced from the LangChain SDK's
    ``data._profiles._PROFILES`` dicts — never hand-curated.
  - Family aliases (fable, opus, sonnet, haiku, mythos) on
    ``claude-code`` / ``claude-code-sdk`` are resolved through
    the ACP ``session/new`` PICKER. The CLI returns a picker
    entry per supported model with the actual model id in
    the description (e.g. ``REPLACES claude-fable-5[1m]``).
    We parse that, then look up the dated id in the SDK.
  - For providers that don't expose a picker (direct Anthropic
    API, OpenAI, DeepSeek, …) family aliases are silent
    passthrough — we can't determine the current generation's
    id statically.
  - If the picker lookup fails for ANY reason — silent
    passthrough. Per operator contract 2026-08-19: "If you
    don't know and can't get it via the API, better not do
    any cap at all besides the operator's -l."

These tests pin the architecture so the regression cannot
recur in any direction (hand-curated hardcodes, removing the
picker entirely, OR silently clamping to a wrong default).
"""

from __future__ import annotations

import pytest

from deep_worker_cli.context_limit.provider_max import (
    _ANTHROPIC_MAX_INPUT,
    _CODEX_CLI_MAX_INPUT,
    _FAMILY_ALIASES,
    _GEMINI_MAX_INPUT,
    _OPENAI_MAX_INPUT,
    _VIA_PICKER_RE,
    _parse_picker_for_alias,
    clear_resolve_cache,
    clamp_to_provider_max,
    provider_max_for,
    resolve_provider_max,
    set_picker_lookup_for_testing,
    warn_if_above_provider_max,
)


# ── Architecture: data source is the SDK, not this package ──────────────


def test_provider_max_table_is_built_from_sdk_profiles() -> None:
    """Every entry in the provider_max dicts must trace back to
    a LangChain SDK ``_PROFILES`` dict, not a hand-curated value.

    The (provider, model) row must be present in the SDK's
    ``max_input_tokens`` mapping for the same model id. If we
    ever find a row that is NOT in the SDK, that's a manual
    hardcode — and the agent_a 2026-08-18 regression says
    we don't want to maintain those manually.
    """
    try:
        from langchain_anthropic.data import _profiles as _anthropic_profiles
    except ImportError:
        _anthropic_profiles = None
    try:
        from langchain_openai.data import _profiles as _openai_profiles
    except ImportError:
        _openai_profiles = None
    try:
        from langchain_google_genai.data import (  # type: ignore[import-not-found]
            _profiles as _gemini_profiles,
        )
    except ImportError:
        _gemini_profiles = None

    sdk_anthropic = (
        {m: p["max_input_tokens"] for m, p in _anthropic_profiles._PROFILES.items()}
        if _anthropic_profiles else {}
    )
    sdk_openai = (
        {m: p["max_input_tokens"] for m, p in _openai_profiles._PROFILES.items()}
        if _openai_profiles else {}
    )
    sdk_gemini = (
        {m: p["max_input_tokens"] for m, p in _gemini_profiles._PROFILES.items()}
        if _gemini_profiles else {}
    )

    # Codex CLI: per-model override documented in provider_max.py
    # (CLI enforces 400k on /v1/chat/completions, not the API's 1.05M).
    codex_override = {"gpt-5-codex", "gpt-5.3-codex", "gpt-5.5-codex"}

    for model, _value in _ANTHROPIC_MAX_INPUT.items():
        assert model in sdk_anthropic, (
            f"({model!r}) is in _ANTHROPIC_MAX_INPUT but "
            f"NOT in langchain_anthropic._PROFILES — "
            f"this is a hand-curated hardcode, which is the bug."
        )
        assert _ANTHROPIC_MAX_INPUT[model] == sdk_anthropic[model], (
            f"value for {model!r} disagrees with SDK — "
            f"value is {(_ANTHROPIC_MAX_INPUT[model])}, "
            f"SDK says {sdk_anthropic[model]}."
        )

    for model, _value in _OPENAI_MAX_INPUT.items():
        assert model in sdk_openai, (
            f"({model!r}) is in _OPENAI_MAX_INPUT but "
            f"NOT in langchain_openai._PROFILES."
        )

    for model, _value in _GEMINI_MAX_INPUT.items():
        assert model in sdk_gemini, (
            f"({model!r}) is in _GEMINI_MAX_INPUT but "
            f"NOT in langchain_google_genai._PROFILES."
        )

    # Codex CLI override is documented; not in SDK is fine.
    for model in _CODEX_CLI_MAX_INPUT:
        assert model in codex_override, (
            f"Codex CLI override for {model!r} not in the documented "
            f"exception list {codex_override}."
        )


def test_provider_max_table_has_zero_hand_curated_anthropic_entries() -> None:
    """Equivalently: every Anthropic row in the table is in the
    SDK. This is the strict version of the previous test —
    it catches the case where someone adds a manual ``claude-fable-5:
    1_000_000`` to provider_max.py.
    """
    try:
        from langchain_anthropic.data import _profiles as _anthropic_profiles
    except ImportError:
        pytest.skip("langchain_anthropic not installed")
    sdk_models = set(_anthropic_profiles._PROFILES.keys())

    table_anthropic_models = set(_ANTHROPIC_MAX_INPUT)
    extras = table_anthropic_models - sdk_models
    assert not extras, (
        f"Models in _ANTHROPIC_MAX_INPUT but not in "
        f"langchain_anthropic._PROFILES: {extras}. "
        f"These are hand-curated hardcodes — the agent_a regression."
    )


# ── Family aliases: NEVER statically resolved ────────────────────────────


def test_family_aliases_are_explicitly_listed() -> None:
    """The set of family aliases is maintained as an explicit
    frozenset so the test suite can pin the contract.
    """
    assert _FAMILY_ALIASES == frozenset({"opus", "sonnet", "haiku", "fable", "mythos"})


@pytest.mark.parametrize("alias", ["opus", "sonnet", "haiku", "fable", "mythos"])
def test_provider_max_for_family_alias_is_none(alias: str) -> None:
    """Family aliases are NEVER in the table. The sync SDK-only
    helper returns ``None`` — the async picker resolver is the
    only path that can resolve these.

    Per operator contract (2026-08-19):
    > "Family always means the latest available model in this
    > family. The context size can change with the next family
    > release — you have no right to use old values or old models
    > of this family."
    """
    for provider in ("claude-code", "claude-code-sdk"):
        assert provider_max_for(provider, alias) is None, (
            f"family alias ({provider!r}, {alias!r}) must NOT statically "
            f"resolve to a specific value — the current context window "
            f"depends on the latest model in the family, which is "
            f"dynamic. The operator's ``-l`` must flow through unchanged."
        )


# ── Picker description parser ─────────────────────────────────────────────


def _reply(*options, current="fable"):
    """Build a fake ``session/new`` reply with the given picker options."""
    return {
        "result": {
            "configOptions": [
                {
                    "id": "model",
                    "currentValue": current,
                    "options": [
                            {"value": o["value"], "name": o.get("name", o["value"]),
                             "description": o.get("description", "")}
                            for o in options
                        ],
                }
            ]
        }
    }


def test_picker_regex_extracts_dated_id_from_replaces_marker() -> None:
    """The ``REPLACES <model_id>`` shape in picker descriptions is
    the documented format from the Claude Code CLI adapter. The
    regex must extract the dated id and not match anything else.
    """
    # Happy path.
    assert (
        _VIA_PICKER_RE.search(
            "Fable (latest Claude Fable, 1M context, released 2026-06). "
            "REPLACES claude-fable-5[1m]"
        )
        .group(1)
        == "claude-fable-5"
    )
    # Multiple replaces (e.g. nested); regex takes the first.
    m = _VIA_PICKER_RE.search("REPLACES claude-foo-bar REPLACES claude-baz-99")
    assert m.group(1) == "claude-foo-bar"
    # No marker → no match.
    assert _VIA_PICKER_RE.search("Opus (1M context)") is None
    # Uppercase REPLACES is NOT matched (only lowercase per docs).
    assert _VIA_PICKER_RE.search("REPLACES Claude-Foo-Bar") is None
    # Trailing punctuation is consumed; matching id stops at non-id char.
    assert (
        _VIA_PICKER_RE.search("REPLACES claude-fable-5[1m]").group(1)
        == "claude-fable-5"
    )


def test_parse_picker_for_alias_returns_dated_id_when_present() -> None:
    """The helper returns the dated id from the picker entry's
    description."""
    reply = _reply(
        {"value": "fable", "description": "Fable. REPLACES claude-fable-5[1m]"},
        {"value": "opus", "description": "Opus (1M context)"},
        {"value": "sonnet", "description": "Sonnet. REPLACES claude-sonnet-5[1m]"},
        current="fable",
    )
    assert _parse_picker_for_alias(reply, "fable") == "claude-fable-5"
    assert _parse_picker_for_alias(reply, "sonnet") == "claude-sonnet-5"


def test_parse_picker_for_alias_returns_none_when_no_marker() -> None:
    """If the entry exists but has no ``REPLACES`` marker (e.g.
    Opus' description is just text), we can't resolve — return
    ``None`` so the async resolver falls through to silent
    passthrough.
    """
    reply = _reply(
        {"value": "opus", "description": "Opus (1M context)"},
        current="opus",
    )
    assert _parse_picker_for_alias(reply, "opus") is None


def test_parse_picker_for_alias_returns_none_when_alias_missing() -> None:
    """If the picker doesn't list the alias at all (e.g. the CLI
    version doesn't include ``haiku``), we can't resolve.
    """
    reply = _reply(
        {"value": "fable", "description": "Fable. REPLACES claude-fable-5[1m]"},
        current="fable",
    )
    assert _parse_picker_for_alias(reply, "haiku") is None


def test_parse_picker_for_alias_returns_none_for_empty_reply() -> None:
    """Malformed/empty replies don't crash the parser."""
    assert _parse_picker_for_alias({}, "fable") is None
    assert _parse_picker_for_alias({"result": {}}, "fable") is None
    assert _parse_picker_for_alias({"result": {"configOptions": []}}, "fable") is None


# ── Async picker resolver ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_resolver_cache():
    """Ensure each test sees a clean module-level resolver cache.

    Tests that exercise the picker inject a mock lookup via
    ``set_picker_lookup_for_testing``; this fixture clears any
    leftover cache entries between tests so isolation holds.
    """
    clear_resolve_cache()
    # Reset any test-only override so it doesn't leak between tests.
    set_picker_lookup_for_testing(None)
    yield
    clear_resolve_cache()
    set_picker_lookup_for_testing(None)


def _mock_picker(reply):
    """Build a mock picker lookup that returns the given reply."""
    async def _lookup(alias: str) -> dict | None:
        return reply
    return _lookup


def _mock_picker_fail():
    """Build a mock picker lookup that returns ``None`` (simulating
    CLI subprocess failure, timeout, etc.)."""
    async def _lookup(alias: str) -> dict | None:
        return None
    return _lookup


def _mock_picker_exception():
    """Build a mock picker lookup that raises (simulating a crash)."""
    async def _lookup(alias: str) -> dict:
        raise RuntimeError("simulated AcpClient subprocess crash")
    return _lookup


async def test_resolve_provider_max_dated_id_uses_sdk_directly() -> None:
    """Dated model IDs bypass the picker and go straight to SDK."""
    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "fable", "description": "REPLACES claude-fable-5[1m]"},
        current="fable",
    )))
    # If this went through the picker, the test reply would change the result.
    # Direct SDK lookup is independent.
    assert (
        await resolve_provider_max("claude-code", "claude-haiku-4-5") == 200_000
    )


async def test_resolve_provider_max_family_alias_via_picker() -> None:
    """Family alias on claude-code → picker lookup → SDK value."""
    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "fable", "description": "Fable. REPLACES claude-fable-5[1m]"},
        current="fable",
    )))
    # claude-fable-5 isn't in the SDK yet → resolver returns None.
    # This is the correct behavior: silent passthrough (NO assumed value).
    assert await resolve_provider_max("claude-code", "fable") is None


async def test_resolve_provider_max_family_alias_resolves_when_sdk_has_id() -> None:
    """When the picker resolves to an id that IS in the SDK, the
    resolver returns the SDK value."""
    # Inject a dated id that exists in the SDK.
    sdk_id = next(iter(_ANTHROPIC_MAX_INPUT))
    sdk_value = _ANTHROPIC_MAX_INPUT[sdk_id]
    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "fable", "description": f"Fable. REPLACES {sdk_id}[1m]"},
        current="fable",
    )))
    assert await resolve_provider_max("claude-code", "fable") == sdk_value


async def test_resolve_provider_max_no_marker_returns_none() -> None:
    """Picker entry exists but has no ``REPLACES`` marker → silent
    passthrough.
    """
    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "opus", "description": "Opus (1M context)"},
        current="opus",
    )))
    assert await resolve_provider_max("claude-code", "opus") is None


async def test_resolve_provider_max_picker_failure_returns_none() -> None:
    """Picker lookup returns ``None`` (subprocess crashed, adapter
    missing, timeout) → silent passthrough per operator contract.
    """
    set_picker_lookup_for_testing(_mock_picker_fail())
    assert await resolve_provider_max("claude-code", "fable") is None


async def test_resolve_provider_max_picker_exception_returns_none() -> None:
    """Picker lookup raises an exception → silent passthrough.

    The resolver MUST NOT propagate the exception — boot-time
    failures should be swallowed.
    """
    set_picker_lookup_for_testing(_mock_picker_exception())
    # MUST NOT raise.
    assert await resolve_provider_max("claude-code", "fable") is None


async def test_resolve_provider_max_alias_missing_from_picker() -> None:
    """Picker exists but doesn't list the alias → silent passthrough."""
    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "fable", "description": "REPLACES claude-fable-5[1m]"},
        current="fable",
    )))
    assert await resolve_provider_max("claude-code", "haiku") is None


async def test_resolve_provider_max_other_providers_no_picker_call() -> None:
    """OpenAI/DeepSeek/Gemini don't go through the picker — family
    aliases are silent passthrough because the picker is Anthropic-only.
    """
    async def _tracker(alias: str) -> dict | None:
        raise AssertionError(
            f"picker called for non-anthropic provider (alias={alias!r})"
        )

    set_picker_lookup_for_testing(_tracker)
    # Direct SDK lookup (gpt-4 = 8192).
    assert await resolve_provider_max("openai", "gpt-4") == 8_192
    # Family alias on openai → silent passthrough (picker never invoked).
    assert await resolve_provider_max("openai", "fable") is None
    # Same for gemini (an unknown model id).
    assert await resolve_provider_max("gemini", "gemini-99-future") is None


async def test_resolve_provider_max_codex_uses_codex_override() -> None:
    """Codex CLI uses the documented override; picker is NOT invoked."""
    async def _tracker(alias: str) -> dict | None:
        raise AssertionError("picker called for codex")

    set_picker_lookup_for_testing(_tracker)
    assert await resolve_provider_max("codex", "gpt-5-codex") == 400_000
    # Family alias on codex → silent passthrough (codex has no family
    # alias concept).
    assert await resolve_provider_max("codex", "fable") is None


async def test_resolve_provider_max_caches_result() -> None:
    """The resolver caches results per (provider, model). The picker
    is called at most once for a given (provider, model) per process.
    """
    call_count = 0

    async def _counting_lookup(alias: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        return _reply(
            {"value": "fable", "description": "REPLACES claude-fable-5[1m]"},
            current="fable",
        )

    set_picker_lookup_for_testing(_counting_lookup)
    a = await resolve_provider_max("claude-code", "fable")
    b = await resolve_provider_max("claude-code", "fable")
    c = await resolve_provider_max("claude-code", "fable")
    assert a == b == c
    assert call_count == 1, (
        f"picker should be called once, was called {call_count} times"
    )


async def test_resolve_provider_max_cache_distinct_keys() -> None:
    """Different (provider, model) keys → distinct cache entries.
    Calling for fable and sonnet should each trigger exactly one
    picker call.
    """
    call_count = 0

    async def _counting_lookup(alias: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        return _reply(
            {"value": alias, "description": f"REPLACES claude-{alias}-5[1m]"},
            current=alias,
        )

    set_picker_lookup_for_testing(_counting_lookup)
    await resolve_provider_max("claude-code", "fable")
    await resolve_provider_max("claude-code", "sonnet")
    await resolve_provider_max("claude-code", "fable")  # cached
    assert call_count == 2


async def test_clear_resolve_cache_resets_cached_values() -> None:
    """``clear_resolve_cache`` invalidates the cache so a new picker
    lookup is performed on the next call."""
    call_count = 0

    async def _counting_lookup(alias: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        return None  # always fails

    set_picker_lookup_for_testing(_counting_lookup)
    await resolve_provider_max("claude-code", "fable")
    await resolve_provider_max("claude-code", "fable")
    assert call_count == 1
    clear_resolve_cache()
    await resolve_provider_max("claude-code", "fable")
    assert call_count == 2


async def test_resolve_provider_max_unknown_provider() -> None:
    """Unknown provider → silent passthrough (NO assumption)."""
    set_picker_lookup_for_testing(_mock_picker_exception())
    assert (
        await resolve_provider_max("mystery-provider", "anything") is None
    )


# ── Boot-time clamp behaviour (sync SDK-only path) ───────────────────────


def test_provider_max_for_known_dated_anthropic_id_comes_from_sdk() -> None:
    """A dated Anthropic model id resolves to the SDK's value,
    not a hand-curated one.
    """
    # claude-haiku-4-5 is in the LangChain SDK = 200K.
    assert provider_max_for("claude-code", "claude-haiku-4-5") == 200_000
    assert provider_max_for("claude-code-sdk", "claude-haiku-4-5") == 200_000


def test_provider_max_for_known_openai_id_comes_from_sdk() -> None:
    """OpenAI ``gpt-4`` is 8_192 per LangChain SDK profile data
    (NOT 128k — that's ``gpt-4-turbo``).
    """
    assert provider_max_for("openai", "gpt-4") == 8_192
    assert provider_max_for("openai", "gpt-4-turbo") == 128_000


def test_provider_max_for_unknown_model_is_none() -> None:
    """Unknown models → ``None`` → silent passthrough at the clamp.

    Not "silent clamp to default" — the clamp function returns
    the requested value unchanged. Returning a default would
    silently cap unknown models at 100k, which is the bug
    pattern the family-alias regression was about.
    """
    assert provider_max_for("unknown-provider", "any-model") is None
    assert provider_max_for("grok", "grok-99-future") is None
    # Family aliases are NOT "unknown" — they are an explicit
    # ack-but-passthrough case. Both end up as None but the
    # reasoning is documented above.
    assert provider_max_for("claude-code", "fable") is None


def test_clamp_unchanged_when_unknown() -> None:
    """When the provider_max is unknown, ``clamp_to_provider_max``
    returns the requested value unchanged (silent passthrough).
    """
    assert clamp_to_provider_max(400_000, "unknown-provider", "anything") == 400_000
    assert clamp_to_provider_max(1_000_000, "unknown-provider", "anything") == 1_000_000


def test_clamp_unchanged_for_family_alias() -> None:
    """When the model is a family alias, ``clamp_to_provider_max``
    returns the requested value unchanged. The operator's
    explicit ``-l`` is not silently overridden.

    The async ``resolve_provider_max`` is the path that handles
    family aliases via the picker. The sync helper is the SDK-only
    fallback for non-async callers (e.g. unit tests).
    """
    for alias in ("fable", "opus", "sonnet", "haiku", "mythos"):
        assert clamp_to_provider_max(400_000, "claude-code", alias) == 400_000
        assert clamp_to_provider_max(1_000_000, "claude-code", alias) == 1_000_000


def test_clamp_unchanged_when_requested_is_none() -> None:
    """``None`` ``-l`` → returns ``None`` (no clamp applied)."""
    assert clamp_to_provider_max(None, "claude-code", "fable") is None
    assert clamp_to_provider_max(None, "openai", "gpt-4") is None


def test_clamp_clamps_known_over_provider_max() -> None:
    """Known dated id with ``-l`` over the SDK-published cap
    → clamped down to the SDK value.
    """
    # gpt-4 = 8_192 per SDK profile data.
    assert clamp_to_provider_max(100_000, "openai", "gpt-4") == 8_192


def test_clamp_unchanged_when_at_or_below_provider_max() -> None:
    """``-l`` at or below the SDK cap → unchanged."""
    assert clamp_to_provider_max(8_192, "openai", "gpt-4") == 8_192
    assert clamp_to_provider_max(8_000, "openai", "gpt-4") == 8_000


# ── Boot-time WARN ───────────────────────────────────────────────────────


def test_warn_emitted_when_known_over_cap(capsys: pytest.CaptureFixture[str]) -> None:
    """Over the known SDK cap → one-line WARN to stderr."""
    warn_if_above_provider_max(100_000, "openai", "gpt-4")
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "8,192" in captured.err or "8192" in captured.err


def test_warn_silent_for_family_alias(capsys: pytest.CaptureFixture[str]) -> None:
    """Family alias → silent for the sync path. The async picker
    path is the one that emits a WARN after a successful lookup.

    The sync helper doesn't have access to the picker (it's a
    subprocess spawn); for family aliases it falls through to
    silent passthrough. Per operator contract: "If you don't
    know and can't get it via the API, better not do any cap
    at all besides the operator's -l." Emitting a warn on the
    sync path would be exactly the silent-wrong assumption the
    agent_a regression was about.
    """
    warn_if_above_provider_max(2_000_000, "claude-code", "fable")
    captured = capsys.readouterr()
    assert captured.err == "", (
        "family alias must not trigger a clamp warn on the sync "
        "path — we don't know the current generation's cap, so "
        "warning would be exactly the silent-wrong assumption "
        "the agent_1 regression was about."
    )


def test_warn_silent_when_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown model → silent."""
    warn_if_above_provider_max(999_999_999, "mystery-provider", "mystery-model")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_silent_when_within_cap(capsys: pytest.CaptureFixture[str]) -> None:
    """``-l`` within the SDK-published cap → silent."""
    warn_if_above_provider_max(8_000, "openai", "gpt-4")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_silent_when_requested_is_none(capsys: pytest.CaptureFixture[str]) -> None:
    """No ``-l`` set → silent."""
    warn_if_above_provider_max(None, "openai", "gpt-4")
    captured = capsys.readouterr()
    assert captured.err == ""


# ── Codex CLI override: documented single exception ──────────────────────


def test_codex_cli_override_uses_400k_not_api_cap() -> None:
    """The Codex CLI cap is 400k, not the API's 1.05M. The
    override is documented in provider_max.py as the ONE
    exception to the SDK-shipped data rule.
    """
    assert provider_max_for("codex", "gpt-5-codex") == 400_000
    assert provider_max_for("codex", "gpt-5.3-codex") == 400_000
    assert provider_max_for("codex", "gpt-5.5-codex") == 400_000


# ── agent_a 2026-08-18 incident regression guard ─────────────────────
#
# The original agent_a crash: ``-l 400k`` against
# ``(claude-code, fable)`` was silently clamped to 200k by
# a hand-curated table that assumed all Claude families were
# 200k. The fix path:
#
#   - Sync helper (clamp_to_provider_max): silent passthrough
#     so the operator's ``-l`` flows through unchanged.
#   - Async resolver (resolve_provider_max): asks the picker
#     what ``fable`` currently resolves to, then looks up in
#     the SDK. If the resolved id IS in the SDK, the operator
#     gets clamped to the real cap; otherwise silent
#     passthrough (no fabricated value).


def test_family_alias_400k_passes_through_sync() -> None:
    """The sync helper: ``-l 400k`` against ``fable`` is silent
    passthrough (operator's value respected, not silently
    overridden). The async resolver is the path that can
    actually resolve family aliases.
    """
    assert clamp_to_provider_max(400_000, "claude-code", "fable") == 400_000


def test_family_alias_emits_no_warn_sync(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-l 400k`` against ``fable`` emits no stderr warn on the
    sync path — the operator's `-l` value is respected, not
    silently overridden.
    """
    warn_if_above_provider_max(400_000, "claude-code", "fable")
    captured = capsys.readouterr()
    assert captured.err == ""


async def test_family_alias_async_resolver_uses_picker() -> None:
    """The async resolver exercises the picker for ``fable`` and
    looks up the resolved id in the SDK. If the SDK has the id,
    we return the real value; if not, silent passthrough.

    The test injects a known-good SDK id to verify the lookup
    actually goes: picker → resolved id → SDK lookup.
    """
    sdk_id = next(iter(_ANTHROPIC_MAX_INPUT))
    sdk_value = _ANTHROPIC_MAX_INPUT[sdk_id]

    set_picker_lookup_for_testing(_mock_picker(_reply(
        {"value": "fable", "description": f"Fable. REPLACES {sdk_id}[1m]"},
        current="fable",
    )))

    assert (
        await resolve_provider_max("claude-code", "fable") == sdk_value
    )


# ── SDK-independence: missing SDK must not crash ────────────────────────


def test_load_openai_profiles_returns_empty_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``langchain_openai`` is importable but its
    ``data._profiles`` submodule is missing, ``_load_openai_profiles``
    MUST return ``{}`` (not raise). This is the graceful-degradation
    path the CLI relies on when an SDK is partially installed.

    Implementation note: we patch the inner import inside
    ``_load_openai_profiles`` (not the module-level cache) so
    we exercise the actual ``ImportError`` handler in the
    helper. We do NOT use ``importlib.reload`` here — that
    leaks global state across tests and breaks unrelated
    tests that run later in the same collection.
    """
    # Inject an ImportError for langchain_openai.data._profiles.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai.data" or name == "langchain_openai.data._profiles":
            raise ImportError(f"simulated missing SDK: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Re-import the helper to get a fresh function bound to the
    # patched builtins. The helper references ``__import__``
    # via the standard ``import`` statement, which resolves to
    # ``builtins.__import__`` at call time.
    from deep_worker_cli.context_limit.provider_max import _load_openai_profiles
    result = _load_openai_profiles()
    assert result == {}, (
        "missing SDK must produce empty dict, not raise — "
        "the CLI falls through to silent passthrough for "
        "unresolvable providers."
    )