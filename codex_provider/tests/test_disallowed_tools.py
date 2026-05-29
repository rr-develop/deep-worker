"""Regression tests for ``DEFAULT_DISABLED_FEATURES`` /
``DEFAULT_CONFIG_OVERRIDES``.

Codex CLI 0.130 advertises 17 native tools per turn unless we strip
them via per-feature flags. We verified live (``RUST_LOG=codex_api=trace``)
that the older ``--config disabled_tools=[...]`` channel is silently
ignored: the ``response.create`` websocket payload's ``tools=[...]``
list is identical with or without it.

These tests guard the two replacement channels (feature flags + top-level
config overrides) against accidental drift:

1. **Critical native tools must be removable by name.** Every Codex
   feature flag whose absence would let the model pick a native tool
   over the deep-worker MCP proxy must stay in
   ``DEFAULT_DISABLED_FEATURES``. Otherwise that tool reappears in the
   API tools list and the model freely calls it, bypassing LangGraph
   history. The mapping flag → native tool is verified live and
   documented in ``codex_provider/chat_model.py``.

2. **Top-level config overrides must include ``web_search="disabled"``.**
   Codex 0.130 moved ``web_search`` out of ``[features]`` into a
   top-level string config; passing ``--disable web_search`` only emits a
   deprecation error and the tool stays in the API.

Native tools that Codex 0.130 cannot remove via any config/feature key
(``apply_patch``, ``update_plan``, ``view_image``, ``request_user_input``,
``list_mcp_resources*``) are NOT covered here — there is no flag to
test. Defense in depth for the dangerous one (``apply_patch``) lives in
``CodexChatModel._raise_if_blocked_native_item``.

These tests are pure import-time / argv-building checks. No subprocess,
no live Codex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


# Feature flags whose absence from ``DEFAULT_DISABLED_FEATURES`` would
# let one or more native Codex tools reappear in the ``tools=[...]``
# field of every ``response.create`` API request. Mapping verified live
# against ``codex features list`` on cli 0.130.0:
#
#   shell_tool       -> exec_command (single-call variant)
#   unified_exec     -> exec_command (with write_stdin)
#   multi_agent      -> spawn_agent, send_input, resume_agent,
#                       wait_agent, close_agent
#   image_generation -> image_generation
#   tool_search      -> tool_search
#   apps             -> mcp__codex_apps__*  (Codex Connectors / GitHub app)
#   plugins          -> plugin-contributed tools
_REQUIRED_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "multi_agent",
    "image_generation",
    "tool_search",
    "apps",
    "plugins",
)


@pytest.mark.parametrize("feature", _REQUIRED_DISABLED_FEATURES)
def test_default_disabled_features_contains(feature: str) -> None:
    """Critical Codex feature flags must stay in
    ``DEFAULT_DISABLED_FEATURES`` so their native tools are stripped
    from the API request. See module docstring for the mapping.
    """
    from codex_provider.chat_model import DEFAULT_DISABLED_FEATURES

    assert feature in DEFAULT_DISABLED_FEATURES, (
        f"{feature!r} is missing from DEFAULT_DISABLED_FEATURES. "
        f"Without it, one or more native Codex tools reappear in the "
        f"response.create tools=[...] list and the model is free to "
        f"call them, bypassing the deep-worker MCP proxy and the "
        f"LangGraph history. See codex_provider/chat_model.py."
    )


def test_default_config_overrides_disables_web_search() -> None:
    """``web_search="disabled"`` MUST be in DEFAULT_CONFIG_OVERRIDES.

    Codex 0.130 ignores ``--disable web_search`` (deprecation error) —
    the only working channel is a top-level ``-c 'web_search="disabled"'``
    config override. Without this, the native ``web_search`` tool stays
    in every API request and the model may pick it over deep-worker's
    own ``web_search_delegation_tool``.
    """
    from codex_provider.chat_model import DEFAULT_CONFIG_OVERRIDES

    assert ("web_search", '"disabled"') in DEFAULT_CONFIG_OVERRIDES, (
        "DEFAULT_CONFIG_OVERRIDES must contain ('web_search', '\"disabled\"'). "
        "Otherwise Codex CLI 0.130 keeps web_search in the API tools list "
        "and the model may call it directly instead of deep-worker's MCP "
        "web_search_delegation_tool. See codex_provider/chat_model.py."
    )


def test_disabled_features_threaded_through_to_exec_args() -> None:
    """End-to-end: ``CodexChatModel._disabled_features`` and
    ``._config_overrides`` reach ``ExecArgs``, and ``_build_argv``
    emits one ``--disable <name>`` per feature plus one
    ``--config <key>=<value>`` per override.
    """
    from codex_provider import CodexChatModel
    from codex_provider.chat_model import (
        DEFAULT_CONFIG_OVERRIDES,
        DEFAULT_DISABLED_FEATURES,
    )
    from codex_provider.exec_runner import CodexExecRunner, ExecArgs

    m = CodexChatModel()
    # Model owns the defaults.
    assert m._disabled_features == DEFAULT_DISABLED_FEATURES  # noqa: SLF001
    assert m._config_overrides == DEFAULT_CONFIG_OVERRIDES  # noqa: SLF001

    # And argv carries them through verbatim.
    runner = CodexExecRunner(executable_path="/usr/bin/codex")
    args = ExecArgs(
        prompt="hi",
        disabled_features=("shell_tool", "multi_agent"),
        config_overrides=(("web_search", '"disabled"'),),
    )
    argv = runner.build_argv_for_test(args)
    # Two --disable <name> pairs.
    assert argv.count("--disable") == 2
    assert argv[argv.index("--disable") + 1] in {"shell_tool", "multi_agent"}
    # One --config web_search="disabled" pair.
    assert 'web_search="disabled"' in argv
    # The dead disabled_tools=[...] channel must not be used.
    joined = " ".join(argv)
    assert "disabled_tools=" not in joined
