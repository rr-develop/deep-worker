"""``--model`` must reach the CLI, be validated against the LIVE adapter
picker, and never rely on a hardcoded Python-side model list.

OWNER RULE (memorized in feedback_no_model_hardcode_2026_08_15.md):
  * Hardcoded model names/aliases are forbidden.
  * Short names (fable, opus, sonnet, haiku, ...) MUST be accepted.
  * Empty --model MUST resolve to the adapter's current default class.

The adapter reports its verdict in every ``session/new`` reply, in
``result.configOptions[id="model"]``. Live-verified subtlety (0.68.0): the
adapter INJECTS the pinned value into the picker it returns, so membership
in ``options`` is ALWAYS true and useless as a check. What distinguishes a
model it does not know is the metadata on that injected entry:
``description == "Custom model"`` and ``name == value``. Recognized pins get
a real display name and description (``fable`` → name "Fable"; ``opusplan``
→ "Opus Plan Mode"), get normalized (``Opus`` → currentValue ``opus[1m]``),
or get dropped entirely (``claude-opus-4-20250514`` → currentValue
``default``). Validation therefore keys on the marker, at session/new time,
never on a Python-side list.
"""
from __future__ import annotations

import os

import pytest

from deep_worker.providers.claude_code_provider import profile as ccp_profile
from deep_worker.providers.claude_code_provider.profile import (
    DEFAULT_MODEL_OPTION_VALUE,
    MODEL_CONFIG_OPTION_ID,
    MODEL_ENV_VAR,
    UNKNOWN_MODEL_DESCRIPTION,
    AcpModelError,
    build_profile,
    validate_session_new_reply,
)


class _FakeAdapterEntry:
    """Truthy is_file() so unit tests do not need the real staged adapter."""

    def __init__(self, real: object) -> None:
        self._real = real

    def is_file(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self._real)

    def __fspath__(self) -> str:
        return str(self._real)


@pytest.fixture(autouse=True)
def _adapter_entry_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the on-disk adapter check so validation tests run in any
    checkout (dw4 review round 5 point 2). ``build_profile`` still refuses
    to spawn node; unit tests only exercise the pure Python plumbing."""
    monkeypatch.setattr(
        ccp_profile,
        "ADAPTER_ENTRY",
        _FakeAdapterEntry(ccp_profile.ADAPTER_ENTRY),
    )
    monkeypatch.setattr(
        ccp_profile,
        "_adapter_version_from_package",
        lambda: ccp_profile.PINNED_ADAPTER_VERSION,
    )


def _profile(model: str | None):
    # verify=False: unit tests exercise the profile shape without node.
    return build_profile(model=model, verify=False)


# Verbatim shape of a live 0.68.0 ``session/new`` model picker with no pin.
LIVE_KNOWN_OPTIONS = (
    {"value": "default", "name": "Default (recommended)",
     "description": "Opus (1M context)"},
    {"value": "opus[1m]", "name": "Opus (1M context)",
     "description": "Opus 5 with 1M context \u00b7 Best for everyday, complex tasks"},
    {"value": "claude-fable-5[1m]", "name": "Fable",
     "description": "Fable 5 \u00b7 Most capable for your hardest and longest-running tasks"},
    {"value": "sonnet", "name": "Sonnet",
     "description": "Sonnet 5 \u00b7 Efficient for routine tasks"},
    {"value": "haiku", "name": "Haiku",
     "description": "Haiku 4.5 \u00b7 Fastest for quick answers"},
)

LIVE_KNOWN_VALUES = tuple(o["value"] for o in LIVE_KNOWN_OPTIONS)


def _reply(options, current="default"):
    """Build a session/new reply carrying a model picker."""
    return {
        "result": {
            "sessionId": "sid-1",
            "configOptions": [
                {
                    "id": MODEL_CONFIG_OPTION_ID,
                    "type": "select",
                    "currentValue": current,
                    "options": [dict(o) for o in options],
                }
            ],
        }
    }


def _recognized_reply(pin, *, name=None, description="Recognized model"):
    """What the adapter returns for a pin it KNOWS: the value is injected
    (or substituted) with a real display name and a real description."""
    return _reply(
        list(LIVE_KNOWN_OPTIONS)
        + [{"value": pin, "name": name or pin.title(),
            "description": description}],
        current=pin,
    )


def _unknown_reply(pin):
    """What the adapter returns for a pin it does NOT know: value echoed
    back, name == value, description == the passthrough marker."""
    return _reply(
        list(LIVE_KNOWN_OPTIONS)
        + [{"value": pin, "name": pin,
            "description": UNKNOWN_MODEL_DESCRIPTION.title()}],
        current=pin,
    )


class TestProfilePinsModelVerbatim:
    """No Python-side normalization: whatever the operator supplies (empty
    → no pin, non-empty → verbatim). Validation happens at session/new."""

    def test_none_means_no_pin(self) -> None:
        assert _profile(None).env == ()

    @pytest.mark.parametrize("empty", ["", "   ", "\t"])
    def test_empty_or_whitespace_means_no_pin(self, empty: str) -> None:
        assert _profile(empty).env == ()

    @pytest.mark.parametrize(
        "name",
        [
            "fable",
            "opus",
            "sonnet",
            "haiku",
            "opus[1m]",
            "claude-opus-4-20250514",
            "some-future-model-2027",
        ],
    )
    def test_any_nonempty_name_is_pinned_verbatim(self, name: str) -> None:
        assert dict(_profile(name).env).get(MODEL_ENV_VAR) == name

    def test_pin_is_stripped_of_surrounding_whitespace(self) -> None:
        assert dict(_profile("  fable  ").env).get(MODEL_ENV_VAR) == "fable"

    def test_env_is_hashable_for_cache_key(self) -> None:
        assert isinstance(hash(_profile("opus").env), int)


class TestValidateAgainstAdapterVerdict:
    """The verdict comes from the adapter's session/new reply, NEVER from a
    Python constant. Replies below mirror live 0.68.0 shapes exactly."""

    def test_no_pin_never_raises(self) -> None:
        reply = _reply(LIVE_KNOWN_OPTIONS)
        validate_session_new_reply(reply, env=None)
        validate_session_new_reply(reply, env={})

    def test_short_alias_recognized_by_adapter_is_accepted(self) -> None:
        """THE OWNER'S CASE: --model fable. Live adapter substitutes the
        ``claude-fable-5[1m]`` entry with value ``fable`` keeping the real
        Fable name/description."""
        reply = _reply(
            [
                o if o["value"] != "claude-fable-5[1m]"
                else {"value": "fable", "name": "Fable",
                      "description": o["description"]}
                for o in LIVE_KNOWN_OPTIONS
            ],
            current="fable",
        )
        validate_session_new_reply(reply, env={MODEL_ENV_VAR: "fable"})

    def test_recognized_pin_appended_with_real_metadata_is_accepted(self) -> None:
        """Live case ``opusplan``: appended to the picker with a real name."""
        reply = _recognized_reply(
            "opusplan", name="Opus Plan Mode",
            description="Use Opus in plan mode, Sonnet otherwise",
        )
        validate_session_new_reply(reply, env={MODEL_ENV_VAR: "opusplan"})

    def test_dated_id_recognized_is_accepted(self) -> None:
        """Live case ``claude-sonnet-4-20250514``: known, with a hint desc."""
        reply = _recognized_reply(
            "claude-sonnet-4-20250514", name="Sonnet 4",
            description="Newer version available \u00b7 select Sonnet for Sonnet 5",
        )
        validate_session_new_reply(
            reply, env={MODEL_ENV_VAR: "claude-sonnet-4-20250514"}
        )

    @pytest.mark.parametrize("known", LIVE_KNOWN_VALUES)
    def test_every_offered_value_is_accepted(self, known: str) -> None:
        validate_session_new_reply(
            _reply(LIVE_KNOWN_OPTIONS, current=known),
            env={MODEL_ENV_VAR: known},
        )

    def test_normalized_pin_absent_from_picker_is_accepted(self) -> None:
        """Live case ``Opus`` → currentValue ``opus[1m]``, and ``Opus`` itself
        never appears in options. Absence means resolved, not rejected."""
        validate_session_new_reply(
            _reply(LIVE_KNOWN_OPTIONS, current="opus[1m]"),
            env={MODEL_ENV_VAR: "Opus"},
        )

    def test_dropped_pin_raises(self) -> None:
        """REGRESSION (dw4 review round 7, blocking): live case
        ``claude-opus-4-20250514`` — the adapter DROPS the pin (absent from
        options) and falls back to ``default``. Accepting that reintroduces
        the exact requested-vs-executed divergence this change removes: the
        operator asked for one model and inference proceeds on another. A
        warning is not enough — it must fail before any prompt."""
        with pytest.raises(AcpModelError) as exc:
            validate_session_new_reply(
                _reply(LIVE_KNOWN_OPTIONS, current=DEFAULT_MODEL_OPTION_VALUE),
                env={MODEL_ENV_VAR: "claude-opus-4-20250514"},
            )
        msg = str(exc.value)
        assert "claude-opus-4-20250514" in msg
        assert "DROPPED" in msg
        # The operator must be told what they CAN pass, from the live picker.
        for v in LIVE_KNOWN_VALUES:
            assert v in msg

    def test_explicit_default_pin_is_accepted(self) -> None:
        """``--model default`` is the supported way to ask for the current
        default class — it must NOT be mistaken for a dropped pin."""
        validate_session_new_reply(
            _reply(LIVE_KNOWN_OPTIONS, current=DEFAULT_MODEL_OPTION_VALUE),
            env={MODEL_ENV_VAR: DEFAULT_MODEL_OPTION_VALUE},
        )
        validate_session_new_reply(
            _reply(LIVE_KNOWN_OPTIONS, current=DEFAULT_MODEL_OPTION_VALUE),
            env={MODEL_ENV_VAR: "  DEFAULT  "},
        )

    def test_dropped_pin_detection_needs_the_default_fallback(self) -> None:
        """Absence alone is NOT rejection: ``Opus`` is absent from options
        too, but resolves to a concrete entry. Only absence PLUS a
        ``default`` currentValue means the pin was discarded."""
        validate_session_new_reply(
            _reply(LIVE_KNOWN_OPTIONS, current="opus[1m]"),
            env={MODEL_ENV_VAR: "Opus"},
        )
        with pytest.raises(AcpModelError):
            validate_session_new_reply(
                _reply(LIVE_KNOWN_OPTIONS, current=DEFAULT_MODEL_OPTION_VALUE),
                env={MODEL_ENV_VAR: "Opus"},
            )

    def test_dropped_pin_with_no_current_value_fails_open(self) -> None:
        """No currentValue → no verdict to read; prompt-time is the guard."""
        reply = _reply(LIVE_KNOWN_OPTIONS)
        reply["result"]["configOptions"][0].pop("currentValue")
        validate_session_new_reply(reply, env={MODEL_ENV_VAR: "whatever-xyz"})

    @pytest.mark.parametrize("bogus", ["bogus-9999", "opud", "gpt-4"])
    def test_custom_model_marker_raises(self, bogus: str) -> None:
        with pytest.raises(AcpModelError) as exc:
            validate_session_new_reply(
                _unknown_reply(bogus), env={MODEL_ENV_VAR: bogus}
            )
        assert repr(bogus) in str(exc.value)

    def test_error_lists_only_models_the_adapter_knows(self) -> None:
        with pytest.raises(AcpModelError) as exc:
            validate_session_new_reply(
                _unknown_reply("bogus-9999"), env={MODEL_ENV_VAR: "bogus-9999"}
            )
        msg = str(exc.value)
        for v in LIVE_KNOWN_VALUES:
            assert v in msg, f"live picker value {v!r} missing from error"
        # The passthrough entry must NOT be advertised back as a choice.
        assert "[bogus-9999" not in msg and ", bogus-9999" not in msg

    def test_marker_match_is_case_insensitive_on_both_sides(self) -> None:
        reply = _reply(
            list(LIVE_KNOWN_OPTIONS)
            + [{"value": "BoGuS", "name": "BoGuS",
                "description": "  CUSTOM MODEL  "}],
            current="BoGuS",
        )
        with pytest.raises(AcpModelError):
            validate_session_new_reply(reply, env={MODEL_ENV_VAR: "bogus"})

    def test_picker_missing_fails_open(self) -> None:
        """Older adapter that doesn't ship configOptions must not wedge — the
        residual guard is prompt-time model_not_found from the adapter."""
        for reply in (
            {},
            {"result": {}},
            {"result": {"sessionId": "sid"}},
            {"result": {"sessionId": "sid", "configOptions": []}},
            {"result": {"sessionId": "sid",
                        "configOptions": [{"id": "mode",
                                           "options": [{"value": "auto"}]}]}},
        ):
            validate_session_new_reply(reply, env={MODEL_ENV_VAR: "anything"})

    def test_malformed_options_never_raise_typeerror(self) -> None:
        reply = _reply(LIVE_KNOWN_OPTIONS)
        reply["result"]["configOptions"][0]["options"] = [
            None, 42, {"value": None}, {"name": "no value"},
            {"value": "x", "description": None},
        ]
        validate_session_new_reply(reply, env={MODEL_ENV_VAR: "x"})

    def test_new_model_in_picker_is_accepted_without_code_change(self) -> None:
        """The whole point of the no-hardcode rule: when Anthropic ships a
        new model tomorrow, the adapter knows it and we do not need an edit."""
        validate_session_new_reply(
            _recognized_reply(
                "some-unreleased-model-2027", name="Unreleased",
                description="Ships tomorrow",
            ),
            env={MODEL_ENV_VAR: "some-unreleased-model-2027"},
        )


class TestChatModelPassesModelThrough:
    """End-to-end: ClaudeCodeAcpChatModel → profile.env → client."""

    def test_chat_model_pins_short_name_verbatim(self) -> None:
        from deep_worker.providers.claude_code_provider import (
            ClaudeCodeAcpChatModel,
        )

        # Read ._profile (BaseChatModel uses extra="allow" and the field
        # shadows the ``profile`` property on instances).
        model = ClaudeCodeAcpChatModel(model="fable")
        assert dict(model._profile.env).get(MODEL_ENV_VAR) == "fable"

    def test_chat_model_pins_full_id_verbatim(self) -> None:
        from deep_worker.providers.claude_code_provider import (
            ClaudeCodeAcpChatModel,
        )

        model = ClaudeCodeAcpChatModel(model="claude-opus-4-20250514")
        assert (
            dict(model._profile.env).get(MODEL_ENV_VAR)
            == "claude-opus-4-20250514"
        )

    def test_chat_model_without_model_pins_nothing(self) -> None:
        from deep_worker.providers.claude_code_provider import (
            ClaudeCodeAcpChatModel,
        )

        assert ClaudeCodeAcpChatModel()._profile.env == ()

    def test_profile_carries_the_session_new_validator(self) -> None:
        """The client must be able to call the profile's validator; a lost
        wire-up would silently disable model validation and reintroduce the
        original 'requested vs executed' divergence."""
        p = _profile("fable")
        assert p.session_new_validator is validate_session_new_reply


class TestSharedClientKeyedByEnv:
    """Two models pinned differently must never share one CLI subprocess."""

    def test_different_env_yields_different_clients(self) -> None:
        from deep_worker.providers.acp.acp_client import (
            get_global_acp_client,
            reset_global_acp_clients,
        )

        reset_global_acp_clients()
        try:
            fable = get_global_acp_client(_profile("fable"), None)
            opus = get_global_acp_client(_profile("opus"), None)
            assert fable is not opus, (
                "a client pinned to one model was reused for another — "
                "the second agent would silently run the first model"
            )
        finally:
            reset_global_acp_clients()

    def test_same_env_shares_one_client(self) -> None:
        from deep_worker.providers.acp.acp_client import (
            get_global_acp_client,
            reset_global_acp_clients,
        )

        reset_global_acp_clients()
        try:
            first = get_global_acp_client(_profile("fable"), None)
            second = get_global_acp_client(_profile("fable"), None)
            assert first is second
        finally:
            reset_global_acp_clients()

    def test_client_env_merges_over_os_environ(self) -> None:
        """A bare env dict would strip PATH/HOME and break the CLI."""
        from deep_worker.providers.acp.acp_client import (
            get_global_acp_client,
            reset_global_acp_clients,
        )

        reset_global_acp_clients()
        try:
            client = get_global_acp_client(_profile("fable"), None)
            env = client._env
            assert env is not None
            assert env[MODEL_ENV_VAR] == "fable"
            for inherited in ("PATH", "HOME"):
                if inherited in os.environ:
                    assert env.get(inherited) == os.environ[inherited]
        finally:
            reset_global_acp_clients()

    def test_no_pin_leaves_client_env_unset(self) -> None:
        from deep_worker.providers.acp.acp_client import (
            get_global_acp_client,
            reset_global_acp_clients,
        )

        reset_global_acp_clients()
        try:
            client = get_global_acp_client(_profile(None), None)
            assert client._env is None
        finally:
            reset_global_acp_clients()

    def test_client_receives_the_session_new_validator(self) -> None:
        """Wire-up regression guard: without the validator on the client,
        session/new would skip the picker check and any name would 'work'."""
        from deep_worker.providers.acp.acp_client import (
            get_global_acp_client,
            reset_global_acp_clients,
        )

        reset_global_acp_clients()
        try:
            client = get_global_acp_client(_profile("fable"), None)
            assert client._session_new_validator is validate_session_new_reply
        finally:
            reset_global_acp_clients()


class TestSessionNewInvokesTheValidator:
    """Behavioural guard: the client must CALL the validator inside
    ``session_new`` and let it raise BEFORE the session id is returned.
    Attribute wire-up alone (TestSharedClientKeyedByEnv) would still pass if
    the call site were deleted, silently disabling model validation."""

    @staticmethod
    def _client(validator):
        from deep_worker.providers.acp.acp_client import AcpClient

        client = AcpClient(
            None,
            cmd=("node", "unused"),
            env={MODEL_ENV_VAR: "gpt-4"},
            session_new_validator=validator,
        )
        return client

    @staticmethod
    def _stub_call(client, reply):
        async def _call(method, params):  # noqa: ANN001
            return reply

        client._call = _call

    def test_bad_pin_raises_from_session_new(self) -> None:
        import asyncio

        client = self._client(validate_session_new_reply)
        self._stub_call(client, _unknown_reply("gpt-4"))
        with pytest.raises(AcpModelError):
            asyncio.run(client.session_new(cwd="/tmp", mcp_servers=[]))

    def test_good_pin_returns_session_id(self) -> None:
        import asyncio

        client = self._client(validate_session_new_reply)
        client._env = {MODEL_ENV_VAR: "sonnet"}
        self._stub_call(client, _reply(LIVE_KNOWN_OPTIONS, current="sonnet"))
        sid = asyncio.run(client.session_new(cwd="/tmp", mcp_servers=[]))
        assert sid == "sid-1"

    def test_dropped_pin_raises_from_session_new(self) -> None:
        """The dropped-pin verdict must abort session creation, not just be
        computable in isolation — otherwise the wrong model runs."""
        import asyncio

        client = self._client(validate_session_new_reply)
        client._env = {MODEL_ENV_VAR: "claude-opus-4-20250514"}
        self._stub_call(
            client,
            _reply(LIVE_KNOWN_OPTIONS, current=DEFAULT_MODEL_OPTION_VALUE),
        )
        with pytest.raises(AcpModelError):
            asyncio.run(client.session_new(cwd="/tmp", mcp_servers=[]))

    def test_validator_receives_reply_and_env(self) -> None:
        import asyncio

        seen: list[tuple] = []
        client = self._client(lambda reply, env: seen.append((reply, env)))
        reply = _reply(LIVE_KNOWN_OPTIONS)
        self._stub_call(client, reply)
        asyncio.run(client.session_new(cwd="/tmp", mcp_servers=[]))
        assert seen == [(reply, {MODEL_ENV_VAR: "gpt-4"})]

    def test_absent_validator_is_a_no_op(self) -> None:
        import asyncio

        client = self._client(None)
        self._stub_call(client, _unknown_reply("gpt-4"))
        assert asyncio.run(
            client.session_new(cwd="/tmp", mcp_servers=[])
        ) == "sid-1"


class TestNoHardcodedWhitelistLeaks:
    """Guard against the class of regression the owner had to correct twice
    (rounds 5–6 → this rewrite): if a hardcoded model list ever reappears
    in the profile module, this test fails."""

    def test_profile_module_does_not_expose_a_model_whitelist(self) -> None:
        forbidden = [
            "KNOWN_MODEL_ALIASES",
            "ALLOW_MODEL_PREFIX",
            "_ADAPTER_FAILLOUD_PREFIX",
            "normalize_model",
        ]
        for name in forbidden:
            assert not hasattr(ccp_profile, name), (
                f"claude_code_provider.profile leaks {name!r} — "
                "hardcoded model lists are forbidden (see "
                "feedback_no_model_hardcode_2026_08_15.md). The adapter's "
                "session/new picker is the sole source of truth."
            )
