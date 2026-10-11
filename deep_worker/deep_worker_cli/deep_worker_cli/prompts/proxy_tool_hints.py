"""Provider-dependent invocation forms for base deep-worker tools.

See design: docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md
Incident:   a related internal incident analysis (not included here)

The base deep-worker tools (``send_message_to_agent``, ``restart_self`` …)
are exposed to the model through an in-process MCP-proxy. The NAME/FORM the
model must use to call them is provider-specific:

* **claude-code-sdk** — static MCP server ``deepworker`` →
  ``mcp__deepworker__<tool>``. The short (bare) name is unstable: after a
  CLI restart/resume the validator stops resolving it and answers
  ``No such tool available`` (see incident).
* **codex** — MCP server name carries the PID (``deepworker_<pid>``) →
  ``mcp__deepworker_<pid>__<tool>``. Bare names are reserved for Codex's
  native tools, so the MCP form is the ONLY way.
* **claude-code** (shared ACP layer) — the in-process proxy is named
  ``deep_worker_acp_proxy_<pid>``; claude-agent-acp surfaces our tools as
  ``mcp__deep_worker_acp_proxy_<pid>__<tool>``.
* **grok** — grok 1.0.5+ lists our tools under the MANGLED name
  ``deep_worker_acp_proxy_<pid>__<tool>`` (no ``mcp__`` prefix) and
  dispatches the call through its ``use_tool`` built-in itself; ≤1.0.4 used
  a ``CallMcpTool{server, toolName}`` wrapper
  (a related internal incident analysis, not included here). Bare names belong
  to grok's NATIVE built-ins (run_terminal_command/read_file/grep/…); our
  tools are reachable ONLY under the mangled name.
* **stateless API providers** (deepseek / openai / anthropic / google /
  gemini / vertexai / minimax / any OpenAI-compatible provider from
  ``providers.json``) — no MCP-proxy; base tools are ordinary LangGraph
  ``BaseTool``s, so the bare name IS the call name.

This module is the SINGLE pure source of truth for that mapping. It does
NOT touch tool metadata or the CLI name-resolution machinery — it only
produces strings for system-prompt / cognitive-hint text.

The note ALWAYS carries a CRITICAL emphasis on
``send_message_to_agent`` because every inter-agent reply must travel
through that channel — a free-text reply to a peer message is delivered
to the human user instead of the peer, silently breaking the
conversation. Different providers have different surface quirks (the
peer message may surface as a user-role message with a ``System:>``
prefix on ACP providers, etc.) but the required action is invariant:
call ``send_message_to_agent`` with the form spelled out below.
"""

from __future__ import annotations

import logging
import os

_LOGGER = logging.getLogger(__name__)

# The three "command" tools the model tends to call from memory (without a
# runtime-supplied schema), which is why they break first. Covered with an
# explicit per-provider form; all other base tools fall under one general
# rule in the rendered note.
COMMAND_TOOLS: tuple[str, ...] = (
    "send_message_to_agent",
    "restart_self",
    "restart_agent",
)


# The concrete tools named in the "ALL base tools (…)" aside.
_BASE_TOOL_SAMPLE: tuple[str, ...] = ("read_file", "execute", "ls")


def _samples_intact(selection: object | None) -> bool:
    """True when every sampled tool survives, i.e. the aside is unchanged.

    Used to keep the ORIGINAL line breaking on the unrestricted path: the
    default prompt must stay byte-identical to its pre-feature text, and
    re-wrapping it would be a silent change to every agent's prompt.
    """
    return all(_permitted(selection, n) for n in _BASE_TOOL_SAMPLE)


def _base_tool_examples(selection: object | None) -> str:
    """The ``(read_file, execute, ls, …)`` aside, narrowed to live tools.

    The aside exists to make "ALL base tools" concrete. Under a selection
    the concrete names must be ones the agent still has, or the sentence
    that teaches the calling convention doubles as a menu of tools the
    agent cannot call. Returns ``""`` when nothing sampled survives, which
    leaves the rule stated without examples.
    """
    sample = [n for n in _BASE_TOOL_SAMPLE if _permitted(selection, n)]
    return f" ({', '.join(sample)}, …)" if sample else ""


def _permitted(selection: object | None, name: str) -> bool:
    """True when *selection* allows *name* (or there is no selection).

    Typed loosely on purpose: this module is deliberately import-light
    (it must render prompts without dragging in the tool-selection
    machinery), and the only thing it needs from a selection is the
    ``permits`` predicate.
    """
    permits = getattr(selection, "permits", None)
    if permits is None:
        return True
    return bool(permits(name))


def _command_tools_for(selection: object | None) -> tuple[str, ...]:
    """The command tools this agent may actually call.

    A per-provider invocation form for a DISABLED tool is worse than
    useless: it is a precise, copy-pasteable instruction for reaching a
    capability the selection removed, sitting in the prompt right where
    the model looks when it wants that capability.
    """
    return tuple(t for t in COMMAND_TOOLS if _permitted(selection, t))

# Static MCP server name for the claude-code-sdk proxy
# (create_sdk_mcp_server(name="deepworker", ...)).
_CLAUDE_SERVER = "deepworker"


def _codex_prefix(pid: int | None) -> str:
    """``mcp__deepworker_<pid>__`` — the codex MCP proxy prefix.

    The literal ``codex_provider/proxy_mcp_server.py`` server name
    (``deepworker_{os.getpid()}``) under the Claude Code-style MCP naming
    scheme codex uses.
    """
    p = pid if pid is not None else os.getpid()
    return f"mcp__deepworker_{p}__"


# Server-name root of the SHARED ACP layer's in-process proxy
# (providers/acp/proxy_mcp_server.py builds ``deep_worker_acp_proxy_<pid>``).
# Kept as a literal so this module stays import-light; a unit test
# cross-checks it against the provider source so the two cannot drift.
ACP_PROXY_SERVER_ROOT = "deep_worker_acp_proxy_"


def _acp_proxy_server(pid: int | None, server_name: str | None = None) -> str:
    """The shared ACP proxy's server name (``deep_worker_acp_proxy_<pid>``).

    ``server_name`` (the LIVE name from ``ProxyMCPServer``) wins when the
    caller has one — the PID-derived fallback is for prompt rendering in
    the agent process, where os.getpid() IS the proxy's pid.
    """
    if server_name:
        return server_name
    p = pid if pid is not None else os.getpid()
    return f"{ACP_PROXY_SERVER_ROOT}{p}"


# Provider → category mapping. Every provider that ships with deep-worker
# MUST fall into one of the four categories handled by
# ``provider_tool_invocation_note``. Stateless-API names are listed here;
# ACP providers are picked up from their category set below; any other
# OpenAI-compatible provider from ``~/.deep_worker/providers.json`` is
# auto-classified as stateless-API on first call.
#
# Owner's directive 2026-08-18: silent regressions when a new provider
# is added are unacceptable. The "automatic" path is: stateless-API
# providers all share a generic-but-strong note (no per-provider block
# needed); ACP providers get the ACP-layer ``_build_tool_contract``
# augmentation; this registry only needs to be touched when a NEW ACP
# category is added. Unknown names log a ``WARNING`` and fall back to
# the stateless-API default, so a missing entry shows up in the logs
# the first time that provider is used.

ACP_PROXY_STATIC: frozenset[str] = frozenset({"claude-code-sdk"})
ACP_PROXY_WITH_PID: frozenset[str] = frozenset({"codex"})
# Shared-ACP-layer providers surfacing our tools as
# ``mcp__deep_worker_acp_proxy_<pid>__<tool>`` (claude-agent-acp naming).
ACP_PROXY_MCP_PREFIX: frozenset[str] = frozenset({"claude-code"})
# Shared-ACP-layer providers whose CLI mangles our tools as
# ``deep_worker_acp_proxy_<pid>__<tool>`` — no ``mcp__`` prefix — and
# dispatches through its own built-in (grok 1.0.5 ``use_tool``).
ACP_PROXY_MANGLED: frozenset[str] = frozenset({"grok"})

STATELESS_API: frozenset[str] = frozenset({
    "deepseek",
    "openai",
    "anthropic",
    "google",
    "gemini",
    "vertexai",
    "minimax",
})


def _config_provider_names() -> frozenset[str]:
    """OpenAI-compatible providers declared in ``~/.deep_worker/providers.json``.

    Loaded lazily so this module remains import-safe even when the config
    file is malformed; any error here is swallowed at ``logging.DEBUG``
    because a broken providers.json must not break prompt rendering.
    """
    try:
        from deep_worker_cli.runtime.provider_registry import load_provider_specs  # noqa: PLC0415

        return frozenset(spec.name for spec in load_provider_specs())
    except Exception as exc:  # noqa: BLE001 — config problems must not break prompt rendering
        _LOGGER.debug("proxy_tool_hints: could not load providers.json: %s", exc)
        return frozenset()


def proxy_tool_form(
    provider: str | None,
    bare_name: str,
    *,
    pid: int | None = None,
    server_name: str | None = None,
) -> str:
    """Return the string the model should use to CALL ``bare_name`` under
    ``provider``.

    MCP tool names are MANGLED differently per provider; this is the
    single mapping. ``server_name`` (the live proxy server name) wins for
    shared-ACP-layer providers when the caller has one — the PID-derived
    fallback assumes rendering happens in the agent process. Never
    raises; unknown providers fall back to the bare name (matches the
    regression-safe default rendered by
    ``provider_tool_invocation_note``).
    """
    if provider in ACP_PROXY_STATIC:
        return f"mcp__{_CLAUDE_SERVER}__{bare_name}"
    if provider in ACP_PROXY_WITH_PID:
        return f"{_codex_prefix(pid)}{bare_name}"
    if provider in ACP_PROXY_MCP_PREFIX:
        return f"mcp__{_acp_proxy_server(pid, server_name)}__{bare_name}"
    if provider in ACP_PROXY_MANGLED:
        # grok 1.0.5+: the CLI's tool list carries the mangled name
        # directly (no mcp__ prefix); the CLI wraps the dispatch in its
        # use_tool built-in itself.
        return f"{_acp_proxy_server(pid, server_name)}__{bare_name}"
    # stateless-API → bare (native LangGraph BaseTool);
    # unknown → bare (matches the regression-safe default).
    return bare_name


# ----- CRITICAL emphasis on send_message_to_agent ------------------------------

# The CRITICAL block is identical across all providers: the policy
# ("reply via send_message_to_agent, never free-text") is invariant. Only
# the invocation form inside the example changes per-provider.
_CRITICAL_TEMPLATE = (
    "**CRITICAL — inter-agent messaging (`send_message_to_agent`).**\n"
    "When you see an incoming user-role message whose content starts with\n"
    "`<sender-name> -> {your_name} :> …`, it was sent by another\n"
    "deep-worker agent, NOT by the human user. You MUST reply using\n"
    "`{send_msg_form}(<sender-name>, <your reply>)` — NOT a free-text turn.\n"
    "This is the ONLY way to respond to a peer. A free-text reply is\n"
    "delivered to the human user as if it were your turn, silently\n"
    "breaking the peer conversation. The fire-and-forget semantics mean\n"
    "a successful call IS your final response for this turn — do not\n"
    "add preamble or trailing narration. On ACP providers the peer\n"
    "message may surface as a user-role block prefixed with `System:>`\n"
    "(or similar); the required action is invariant regardless of the\n"
    "transport detail."
)


def _critical_send_message_block(
    provider: str | None,
    *,
    pid: int | None,
    server_name: str | None = None,
    selection: object | None = None,
) -> str:
    """Return the CRITICAL block with the provider-correct invocation form.

    Kept as a separate helper so the ACP-layer / per-provider
    ``_build_tool_contract`` functions can pull just this block into their
    per-turn contract without re-rendering the full note.

    Returns ``""`` when the selection removed ``send_message_to_agent``:
    the block's entire content is an order to call that tool, so under a
    selection that denies it the block would be both unfollowable and the
    single most emphatic tool advertisement in the prompt.
    """
    if not _permitted(selection, "send_message_to_agent"):
        return ""
    return _CRITICAL_TEMPLATE.format(
        your_name="<your-name>",
        send_msg_form=proxy_tool_form(
            provider, "send_message_to_agent", pid=pid, server_name=server_name
        ),
    )


# ----- Provider-class notes ---------------------------------------------------

def _stateless_api_note(
    provider: str | None, *, pid: int | None, selection: object | None = None
) -> str:
    """Generic note for stateless API providers (no MCP-proxy).

    Rendered for: deepseek / openai / anthropic / google / gemini /
    vertexai / minimax / any OpenAI-compatible provider from
    ``providers.json``. These all behave identically: the bare name IS
    the call name; no MCP prefix to remember.
    """
    commands = _command_tools_for(selection)
    send_form = proxy_tool_form(provider, "send_message_to_agent", pid=pid)
    command_examples = "\n".join(
        f"  - `{t}` → `{proxy_tool_form(provider, t, pid=pid)}`" for t in commands
    )
    lead = (
        f"Call each base tool by its short name directly (`{send_form}`\n"
        "and friends); the langgraph runtime routes the call to the\n"
        "deep-worker implementation."
        if _permitted(selection, "send_message_to_agent")
        else (
            "Call each base tool by its short name directly; the langgraph\n"
            "runtime routes the call to the deep-worker implementation."
        )
    )
    note = (
        "### Calling base deep-worker tools\n"
        "This provider has NO MCP-proxy — the bare name IS the call name.\n"
        f"{lead}"
    )
    if command_examples:
        note += f" Command tools:\n{command_examples}\n"
    else:
        note += "\n"
    # Line breaks are preserved verbatim from the pre-selection text so an
    # unrestricted agent's prompt stays byte-identical to what it was
    # before this feature existed.
    if _samples_intact(selection):
        note += (
            "The same rule applies to ALL base tools (read_file, execute,\n"
            "ls, …): call them by their bare name.\n"
        )
    else:
        note += (
            f"The same rule applies to ALL base tools"
            f"{_base_tool_examples(selection)}: call them by their bare "
            "name.\n"
        )
    critical = _critical_send_message_block(
        provider, pid=pid, selection=selection
    )
    return f"{note}\n{critical}" if critical else note


def _acp_proxy_static_note(
    provider: str | None, *, pid: int | None, selection: object | None = None
) -> str:
    """Note for ACP-SDK providers with a static MCP server name
    (claude-code-sdk). The bare name is unstable after a CLI restart
    (incident 2026-06-28), so the MCP form is the safe invocation."""
    examples = "\n".join(
        f"  - `{t}` → `{proxy_tool_form(provider, t, pid=pid)}`"
        for t in _command_tools_for(selection)
    )
    note = (
        "### Calling base deep-worker tools\n"
        "For INVOCATION on this provider use the MCP forms below; the "
        "short names that appear elsewhere in this prompt denote the "
        "tool's logical identity, NOT its call name. The base deep-worker "
        "tools are exposed through the MCP server `deepworker`. If you "
        "reference a tool by its short name `<X>` and get "
        "\"No such tool available\", call its MCP version "
        "`mcp__deepworker__<X>` instead."
    )
    if examples:
        note += f" Command tools:\n{examples}\n"
    else:
        note += "\n"
    note += (
        f"The same rule applies to ALL base tools{_base_tool_examples(selection)}: "
        "if the short name does not resolve, use "
        "`mcp__deepworker__<tool>`.\n"
    )
    critical = _critical_send_message_block(
        provider, pid=pid, selection=selection
    )
    return f"{note}\n{critical}" if critical else note


def _acp_proxy_with_pid_note(
    provider: str | None, *, pid: int | None, selection: object | None = None
) -> str:
    """Note for ACP providers whose MCP server name carries the PID
    (codex: ``deepworker_<pid>``). Bare names are reserved for the CLI's
    NATIVE tools, so the MCP form is the ONLY way to call our tools."""
    prefix = _codex_prefix(pid)
    examples = "\n".join(
        f"  - `{t}` → `{prefix}{t}`" for t in _command_tools_for(selection)
    )
    note = (
        "### Calling base deep-worker tools\n"
        "For INVOCATION on this provider use the MCP forms below; the "
        "short names that appear elsewhere in this prompt denote the "
        "tool's logical identity, NOT its call name. Short names are "
        "reserved for this CLI's NATIVE tools — call base deep-worker "
        f"tools ONLY through the MCP server, i.e. `{prefix}<X>` (the "
        "prefix carries the current process PID)."
    )
    if examples:
        note += f" Command tools:\n{examples}\n"
    else:
        note += "\n"
    note += (
        f"The same rule applies to ALL base tools{_base_tool_examples(selection)}: "
        f"call them as `{prefix}<tool>`.\n"
    )
    critical = _critical_send_message_block(
        provider, pid=pid, selection=selection
    )
    return f"{note}\n{critical}" if critical else note


def _acp_proxy_mcp_prefix_note(
    provider: str | None, *, pid: int | None, selection: object | None = None
) -> str:
    """Note for shared-ACP-layer providers surfacing our tools under the
    Claude Code naming scheme ``mcp__deep_worker_acp_proxy_<pid>__<tool>``
    (claude-code via claude-agent-acp)."""
    server = _acp_proxy_server(pid)
    examples = "\n".join(
        f"  - `{t}` → `mcp__{server}__{t}`"
        for t in _command_tools_for(selection)
    )
    note = (
        "### Calling base deep-worker tools\n"
        "For INVOCATION on this provider use the MCP forms below; the "
        "short names that appear elsewhere in this prompt denote the "
        "tool's logical identity, NOT its call name. The base deep-worker "
        f"tools live on the MCP server `{server}` (the name carries the "
        "current process PID)."
    )
    if examples:
        note += f" Command tools:\n{examples}\n"
    else:
        note += "\n"
    note += (
        f"The same rule applies to ALL base tools{_base_tool_examples(selection)}: "
        f"call them as `mcp__{server}__<tool>`.\n"
    )
    critical = _critical_send_message_block(
        provider, pid=pid, selection=selection
    )
    return f"{note}\n{critical}" if critical else note


def _acp_proxy_mangled_note(
    provider: str | None, *, pid: int | None, selection: object | None = None
) -> str:
    """Note for shared-ACP-layer providers whose CLI mangles our tools as
    ``<server>__<tool>`` in its own tool list — no ``mcp__`` prefix — and
    dispatches through its own built-in (grok 1.0.5 ``use_tool``; ≤1.0.4
    used ``CallMcpTool``). See
    a related internal incident analysis (not included here)."""
    server = _acp_proxy_server(pid)
    commands = _command_tools_for(selection)
    examples = "\n".join(f"  - `{t}` → `{server}__{t}`" for t in commands)
    # The opening sentence used to name two command tools as examples;
    # under a selection it names only surviving ones, and falls back to a
    # description when none survive.
    sample = ", ".join(f"`{t}`" for t in commands[:2])
    subject = (
        f"The base deep-worker tools ({sample}, … and every "
        "file/shell/web tool)"
        if sample
        else "The base deep-worker tools"
    )
    note = (
        "### Calling base deep-worker tools\n"
        f"{subject} are deep-worker MCP tools — they are NOT "
        "this CLI's native built-ins. In YOUR tool list they appear under "
        f"the mangled name `{server}__<tool>` (the server name carries "
        "the current process PID). Select the tool under exactly that "
        "name; your CLI routes the call to our MCP server itself."
    )
    if examples:
        note += f" Command tools:\n{examples}\n"
    else:
        note += "\n"
    # These names are the CLI's OWN built-ins, not ours — the sentence
    # warns the model NOT to mistake them for deep-worker tools. Our
    # selection cannot switch them off (that is the whole reason grok is
    # listed in ``_UNSUPPRESSABLE_NATIVES``), so naming them stays correct
    # and useful whatever the selection says.
    note += (
        "Short names (`read_file`, `grep`, `run_terminal_command`, …) are "
        "this CLI's NATIVE built-ins — our tools are NOT among them. Do "
        "NOT search their schema on disk (no filesystem hunt over "
        "`mcps/.../tools/*.json`) and do NOT treat them as unavailable.\n"
    )
    critical = _critical_send_message_block(
        provider, pid=pid, selection=selection
    )
    return f"{note}\n{critical}" if critical else note


# ----- Public surface --------------------------------------------------------

def provider_tool_invocation_note(
    provider: str | None,
    *,
    pid: int | None = None,
    selection: object | None = None,
) -> str:
    """Render the provider-specific system-prompt note explaining how to
    invoke base deep-worker tools.

    This function is the single source of truth and ALWAYS returns a
    non-empty note: any provider that ships with deep-worker falls into
    one of the five categories (ACP_PROXY_STATIC / ACP_PROXY_WITH_PID /
    ACP_PROXY_MCP_PREFIX / ACP_PROXY_MANGLED / stateless-API, including
    all config-driven OpenAI-compatible providers from
    ``providers.json``). Unknown names log a ``WARNING`` (so silent
    regressions when a new provider is added show up in the logs) and
    fall back to the stateless-API note, which is the regression-safe
    default.
    """
    if provider in ACP_PROXY_STATIC:
        return _acp_proxy_static_note(provider, pid=pid, selection=selection)
    if provider in ACP_PROXY_WITH_PID:
        return _acp_proxy_with_pid_note(provider, pid=pid, selection=selection)
    if provider in ACP_PROXY_MCP_PREFIX:
        return _acp_proxy_mcp_prefix_note(
            provider, pid=pid, selection=selection
        )
    if provider in ACP_PROXY_MANGLED:
        return _acp_proxy_mangled_note(provider, pid=pid, selection=selection)
    if provider in STATELESS_API or provider in _config_provider_names():
        return _stateless_api_note(provider, pid=pid, selection=selection)
    if provider is None:
        # No provider specified → render the stateless-API default; this
        # is the safe path for tests / introspection.
        return _stateless_api_note(None, pid=pid, selection=selection)
    _LOGGER.warning(
        "proxy_tool_hints: unknown provider %r — no per-provider note; "
        "falling back to the stateless-API default. Add %r to one of "
        "ACP_PROXY_STATIC / ACP_PROXY_WITH_PID / ACP_PROXY_MCP_PREFIX / "
        "ACP_PROXY_MANGLED / STATELESS_API in "
        "deep_worker_cli/prompts/proxy_tool_hints.py to silence this "
        "warning and get a provider-correct note.",
        provider,
        provider,
    )
    return _stateless_api_note(provider, pid=pid, selection=selection)


def critical_send_message_to_agent_block(
    provider: str | None,
    *,
    pid: int | None = None,
    server_name: str | None = None,
) -> str:
    """Return the CRITICAL ``send_message_to_agent`` emphasis block with
    the provider-correct invocation form.

    Exposed for ACP-layer / per-provider ``_build_tool_contract``
    functions to inject into their per-turn contracts. Returns the same
    block as ``provider_tool_invocation_note`` does — kept as a separate
    entry point so callers that already render their own per-provider
    note can pull just this part. ``server_name`` is the LIVE proxy MCP
    server name when the caller has one (ACP layer) — it wins over the
    PID-derived fallback.

    Callers on the ACP path do not need to pass a selection: they build
    the contract from the tools actually BOUND to the model, which the
    selection has already filtered, and only ask for this block when
    ``send_message_to_agent`` is among them.
    """
    return _critical_send_message_block(
        provider, pid=pid, server_name=server_name
    )