"""Provider-dependent invocation forms for base deep-worker tools.

See design: docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md
Incident:   a related internal incident analysis (not included here)

The base deep-worker tools (``send_message_to_agent``, ``restart_self`` …)
are exposed to the model through an in-process MCP-proxy. The NAME/FORM the
model must use to call them is provider-specific:

* **claude-code** — static MCP server ``deepworker`` →
  ``mcp__deepworker__<tool>``. The short (bare) name is unstable: after a
  CLI restart/resume the validator stops resolving it and answers
  ``No such tool available`` (see incident).
* **codex** — MCP server name carries the PID (``deepworker_<pid>``) →
  ``mcp__deepworker_<pid>__<tool>``. Bare names are reserved for Codex's
  native tools, so the MCP form is the ONLY way.
* **grok** — NO string prefix. The grok CLI routes a tool call through its
  own ``CallMcpTool`` built-in, supplying ``server=deep_worker_proxy_<pid>``
  and ``toolName=<tool>`` itself. Bare names belong to grok's NATIVE
  built-ins (Shell/Read/Write/…); our tools are reachable ONLY as MCP tools.
* **deepseek / openai / anthropic / google** — no MCP-proxy; base tools are
  ordinary LangGraph ``BaseTool``s, so the bare name IS the call name.

This module is the SINGLE pure source of truth for that mapping. It does
NOT touch tool metadata or the CLI name-resolution machinery — it only
produces strings for system-prompt / cognitive-hint text.
"""

from __future__ import annotations

import os

# The three "command" tools the model tends to call from memory (without a
# runtime-supplied schema), which is why they break first. Covered with an
# explicit per-provider form; all other base tools fall under one general
# rule in the rendered note.
COMMAND_TOOLS: tuple[str, ...] = (
    "send_message_to_agent",
    "restart_self",
    "restart_agent",
)

# Static MCP server name for the claude-code proxy
# (create_sdk_mcp_server(name="deepworker", ...)).
_CLAUDE_SERVER = "deepworker"


def _codex_prefix(pid: int | None) -> str:
    """``mcp__deepworker_<pid>__`` — codex proxy server name carries the PID
    (``codex_provider/proxy_mcp_server.py``: ``deepworker_{os.getpid()}``)."""
    p = pid if pid is not None else os.getpid()
    return f"mcp__deepworker_{p}__"


def proxy_tool_form(
    provider: str | None,
    bare_name: str,
    *,
    pid: int | None = None,
) -> str:
    """Return the string the model should use to CALL ``bare_name`` under
    ``provider``.

    For claude-code/codex this is the flat MCP name. For grok and for
    no-proxy providers it is the bare name (grok routes via CallMcpTool with
    no string prefix; native providers call the tool directly). Never
    raises; unknown providers fall back to the bare name.
    """
    if provider == "claude-code":
        return f"mcp__{_CLAUDE_SERVER}__{bare_name}"
    if provider == "codex":
        return f"{_codex_prefix(pid)}{bare_name}"
    # grok → bare (CLI resolves as MCP tool); deepseek/openai/… → native.
    return bare_name


def provider_tool_invocation_note(
    provider: str | None,
    *,
    pid: int | None = None,
) -> str:
    """Render the provider-specific system-prompt note about how to invoke
    base deep-worker tools. Returns ``""`` for providers without an
    MCP-proxy (deepseek/openai/anthropic/google) and for unknown providers.
    """
    if provider == "claude-code":
        examples = "\n".join(
            f"  - `{t}` → `{proxy_tool_form(provider, t)}`" for t in COMMAND_TOOLS
        )
        return (
            "### Calling base deep-worker tools\n"
            "For INVOCATION on this provider use the MCP forms below; the "
            "short names that appear elsewhere in this prompt denote the "
            "tool's logical identity, NOT its call name. The base deep-worker "
            "tools are exposed through the MCP server `deepworker`. If you "
            "reference a tool by its short name `<X>` and get "
            "\"No such tool available\", call its MCP version "
            "`mcp__deepworker__<X>` instead. Command tools:\n"
            f"{examples}\n"
            "The same rule applies to ALL base tools (read_file, execute, "
            "ls, …): if the short name does not resolve, use "
            "`mcp__deepworker__<tool>`."
        )
    if provider == "codex":
        prefix = _codex_prefix(pid)
        examples = "\n".join(
            f"  - `{t}` → `{prefix}{t}`" for t in COMMAND_TOOLS
        )
        return (
            "### Calling base deep-worker tools\n"
            "For INVOCATION on this provider use the MCP forms below; the "
            "short names that appear elsewhere in this prompt denote the "
            "tool's logical identity, NOT its call name. Short names are "
            "reserved for Codex's NATIVE tools — call base deep-worker tools "
            f"ONLY through the MCP server, i.e. `{prefix}<X>` (the prefix "
            "carries the current process PID). Command tools:\n"
            f"{examples}\n"
            "The same rule applies to ALL base tools (read_file, execute, "
            f"ls, …): call them as `{prefix}<tool>`."
        )
    if provider == "grok":
        return (
            "### Calling base deep-worker tools\n"
            "The base deep-worker tools "
            "(`send_message_to_agent`, `restart_self`, … and every "
            "file/shell/web tool) are deep-worker MCP tools — they are NOT "
            "grok's native built-ins (Shell/Read/Write/Edit/Glob/Grep/"
            "WebSearch/Task). Call them by name directly: the grok CLI routes "
            "the call to our MCP server `deep_worker_proxy_<pid>` "
            "automatically. Do NOT search their schema on disk (no "
            "Glob/Grep/Read over `mcps/.../tools/*.json`) and do NOT treat "
            "them as unavailable."
        )
    # deepseek/openai/anthropic/google/unknown/None → native tools, no note.
    return ""
