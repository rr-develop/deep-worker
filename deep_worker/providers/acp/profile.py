"""ACP provider profile — the per-CLI configuration for the shared layer.

The ACP layer is provider-agnostic; everything that differs between CLIs
(Claude Code, Codex, Grok, Gemini CLI, …) lives in a profile:

  * ``cmd`` — the argv tuple that launches the CLI in stdio ACP mode
    (e.g. ``("grok", "agent", "stdio")`` or an absolute node + adapter path).
  * ``vendor_prefix`` — proprietary notification prefix to drop
    (``_x.ai/`` for grok, ``_anthropic/`` for Claude Code, …).
  * ``mcp_dispatch_tool`` — the CLI's built-in name for dispatching an MCP
    tool (``CallMcpTool`` in grok/Claude Code).
  * ``known_builtins`` — the CLI's local built-in tool names recorded via
    channel-B passthrough.
  * ``session_load_supported`` — whether the CLI implements ``session/load``
    (restart resume). If False, a persisted session id is discarded on
    restart (fresh session).
  * ``permission_policy`` — fail-closed permission handling (§4.6).
  * ``extra_limit_markers`` — provider-specific usage-limit regex markers.
  * ``fingerprint`` — ``{adapter, adapter_version, session-capability}``
    used to scope persisted session state (DESIGN §3.5).
  * ``provider`` — the deep-worker provider id (``"claude-code"`` …).

Profiles are immutable dataclasses; the CLI-specific bits (command path,
adapter version probing) are constructed by the provider package.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# Provider-neutral post-``session/new`` hook: ``(reply, env) -> None``,
# raising to reject the session. Declared here so both the profile and the
# client share one definition.
SessionNewValidator = Callable[[dict, "dict[str, str] | None"], None]


@dataclass(frozen=True)
class PermissionPolicy:
    """Two-state permission policy: fail-closed by default, BLANKET
    approval under the operator's ``-sp`` (DESIGN §4.6).

    There is deliberately NO per-tool allowlist. Owner directive
    2026-08-24: ``-sp`` must approve everything the CLI asks about, with no
    list anyone has to keep in sync, because a list that exists eventually
    goes stale and takes the agent's tools down with it — silently.

    That is not hypothetical. It is exactly how grok lost four days of
    shell access (a related internal incident analysis, not included
    here): the policy shipped fail-closed with an EMPTY allowlist and
    ``-sp`` was never wired through, so every ``run_terminal_command``
    permission was rejected and grok terminated the turn — surfacing as
    ``AcpTurnCancelledError`` with nothing in any log. claude-code was one
    CLI release from the same failure: its allowlist defaulted to the
    built-ins known at the time it was written, so the first new built-in
    Anthropic shipped would have been denied under ``-sp``.

    Consequently the ONLY question this policy answers is "did the operator
    pass ``-sp``". The remaining knobs describe the CLI's offered OPTION
    kinds, never its tools: grok offers ``allow-*``/``deny-*``;
    claude-agent-acp offers ``allow_once``/``allow_always``/
    ``reject_once``. Approval selects the FIRST offered option whose kind
    starts with ``allow_kind_prefix``; denial selects one starting with
    ``deny_kind_prefix``. Option ids are opaque and version-specific, so an
    id is never invented — a request offering no allow option is denied
    (loudly), because answering with a made-up id makes the CLI reject the
    response and kill the whole turn.
    """

    auto_approve: bool = False
    allow_kind_prefix: str = "allow"
    deny_kind_prefix: str = "deny"

    def is_allowed_kind(self, kind: str) -> bool:  # noqa: ARG002 — kind is for logging only
        """``True`` iff the operator passed ``-sp``.

        ``kind`` is accepted (and ignored) so callers can log WHAT was
        approved; branching on it is precisely the pattern this policy
        exists to forbid.
        """
        return self.auto_approve


@dataclass(frozen=True)
class AcpProfile:
    """Immutable per-CLI configuration for the shared ACP layer."""

    provider: str
    cmd: tuple[str, ...]
    vendor_prefix: str = ""
    mcp_dispatch_tool: str = "CallMcpTool"
    # How MCP tool calls reach the ACP client:
    #   "dispatch_builtin" — grok wraps them in a dispatch built-in:
    #     CallMcpTool{server, toolName, arguments} on ≤1.0.4,
    #     use_tool{tool_name="<server>__<tool>", tool_input} on 1.0.5+
    #     (recognition is title-agnostic — see acp_types.proxy_dispatch_inner).
    #   "mcp_name_prefix" — claude-agent-acp surfaces them as ordinary
    #     tool_call whose name is mcp__<server>__<tool> (live-verified).
    mcp_dispatch_style: str = "dispatch_builtin"
    known_builtins: frozenset[str] = frozenset()
    # CLI-native built-in names IDENTICAL to deep-worker tool names (grok
    # 1.0.5 went snake_case: read_file, grep, …). They must never be routed
    # into our bound tools (double execution with the CLI's arguments) nor
    # registered as passthrough tools (name shadowing in the ToolNode) —
    # the passthrough layer records them via the GENERIC recorder with the
    # original title preserved, and without the "unknown built-in" warning.
    # Keep DISJOINT from ``known_builtins``.
    colliding_builtins: frozenset[str] = frozenset()
    # Name of a live-context source for CLIs that do NOT report usage per
    # model call on the wire. ``None`` means the CLI already does (claude-
    # agent-acp sends ``usage_update``), so no log watcher is needed.
    # Resolved by ``acp.context_watch.build_live_source``; grok sets
    # ``"grok_unified_log"`` because it reports nothing until the prompt
    # reply, by which point a dozen model calls have gone unrecorded.
    live_context_source: str | None = None
    session_load_supported: bool = True
    # Whether the handshake must call ``authenticate`` after ``initialize``.
    # grok requires it (cached_token); claude-agent-acp authenticates from its
    # own credential store and rejects an authenticate call (or ignores it).
    requires_authenticate: bool = True
    # Where the real tool name lives in a ``tool_call`` update. grok puts it
    # in ``title``; claude-agent-acp puts it in ``_meta.claudeCode.toolName``
    # (``title`` is human-readable prose). Extractor callables live in
    # ``acp_types``; profiles reference them by name here.
    tool_name_from_update: str = "from_title"  # or "from_claude_code_meta"
    permission_policy: PermissionPolicy = field(
        default_factory=PermissionPolicy
    )
    extra_limit_markers: tuple[re.Pattern, ...] = ()
    # Extra environment for the CLI subprocess, MERGED over ``os.environ`` by
    # the client (never replacing it — the CLI needs PATH/HOME and the user's
    # credential env). This is how a provider pins the model when its adapter
    # takes the model from the environment rather than from an ACP field.
    # Part of the shared-client cache key: two models with different env are
    # DIFFERENT CLI identities and must not share one subprocess.
    env: tuple[tuple[str, str], ...] = ()
    # Optional post-session/new validator. The client calls it with
    # ``(reply, env)`` right after a successful ``session/new`` — the
    # provider can consult adapter-supplied fields (e.g. ``configOptions``
    # for the model picker) and raise to reject a mis-pinned session BEFORE
    # any prompt is sent. The signature is provider-NEUTRAL (the shared layer
    # never inspects the reply itself), so the hook stays opaque without
    # giving up static checking.
    session_new_validator: SessionNewValidator | None = None
    # Name surfaced for a built-in the profile does not know, and the args key
    # carrying its real title. Default: derived from ``provider``
    # (``grok`` -> ``grokBuiltin``) with ``_acp_original_title``.
    #
    # Overridable because these two strings are a PUBLIC contract, not an
    # internal detail — they are matched literally elsewhere. grok pins
    # ``GrokBuiltin`` / ``_grok_original_title`` because
    # ``bounded_tool_call.resolve_effective_tool_wait_seconds`` keys on exactly
    # those to grant image generation the 600s bound instead of the default
    # 120s. Letting the derived name win would not raise anything; imagine
    # calls would simply start timing out. They also appear verbatim in every
    # already-checkpointed grok history.
    generic_builtin_name: str = ""
    original_title_arg_key: str = "_acp_original_title"
    adapter: str = ""
    adapter_version: str = ""
    # Session fingerprint {adapter, adapter_version, session-capability}.
    # ``session-capability`` describes what the adapter can do with a
    # persisted session (e.g. "load-v1"); bump it when the resume contract
    # changes. None → fingerprint not used (fresh session every restart).
    session_capability: str | None = "load-v1"

    def fingerprint(self) -> dict[str, Any] | None:
        if not self.adapter:
            return None
        fp: dict[str, Any] = {
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
        }
        if self.session_capability is not None:
            fp["session-capability"] = self.session_capability
        return fp

    # -- session-load capability gate --------------------------------------
    def can_load_session(self) -> bool:
        """Whether a persisted session id may be resumed via ``session/load``.

        Requires the adapter to support session/load AND the current session
        capability to match what was persisted (fingerprint equality is
        checked separately at load time; this is the static gate)."""
        return self.session_load_supported
