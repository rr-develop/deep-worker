"""Agent management and creation for the CLI."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# CRITICAL: Ensure local deepagents is imported instead of pip-installed version
# This is required for ProjectMemoryMiddleware which is not in official deepagents
_deepagents_path = Path(__file__).parent.parent.parent.parent / "deepagents"
if _deepagents_path.exists() and str(_deepagents_path) not in sys.path:
    sys.path.insert(0, str(_deepagents_path))

# Lazy import of deepagents to avoid import errors when not needed
if TYPE_CHECKING:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.backends.sandbox import SandboxBackendProtocol
    from deepagents.middleware import (
        MemoryMiddleware,
        SkillsMiddleware,
    )
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from deep_worker_cli.backends import CLIShellBackend, patch_filesystem_middleware
from langchain.agents.middleware import (
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deep_worker_cli.config import (
    COLORS,
    config,
    console,
    get_default_coding_instructions,
    get_glyphs,
    settings,
)
from deep_worker_cli.integrations.sandbox_factory import get_default_working_dir
from deep_worker_cli.local_context import LocalContextMiddleware
from deep_worker_cli.prompts.proxy_tool_hints import (
    provider_tool_invocation_note,
    proxy_tool_form,
)
from deep_worker_cli.middleware.dangling_tool_calls import (
    DanglingToolCallMiddleware,
)
from deep_worker_cli.subagents import list_subagents

DEFAULT_AGENT_NAME = "agent"
"""The default agent name used when no `-a` flag is provided."""


def list_agents() -> None:
    """List all available agents."""
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ "
            "when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "AGENTS.md"
            is_default = agent_name == DEFAULT_AGENT_NAME
            default_label = " [dim](default)[/dim]" if is_default else ""

            bullet = get_glyphs().bullet
            if agent_md.exists():
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}",
                    style=COLORS["primary"],
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}"
                    " [dim](incomplete)[/dim]",
                    style=COLORS["tool"],
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "AGENTS.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no AGENTS.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(
            f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"]
        )

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    agent_md.write_text(source_content)

    console.print(
        f"{get_glyphs().checkmark} Agent '{agent_name}' reset to {action_desc}",
        style=COLORS["primary"],
    )
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(
    assistant_id: str,
    sandbox_type: str | None = None,
    project_memory_dir: Path | None = None,
    provider: str | None = None,
) -> str:
    """Get the base system prompt for the agent.

    This includes:
    1. The immutable base instructions from default_agent_prompt.md
    2. Environment-specific context (working directory, model info, etc.)

    Args:
        assistant_id: The agent identifier for path references
        sandbox_type: Type of sandbox provider
            ("daytona", "langsmith", "modal", "runloop").

            If None, agent is operating in local mode.
        project_memory_dir: Working directory for the agent.

            CRITICAL: This MUST be the same directory passed to backend!
            If None, Path.cwd() is used (DANGEROUS after daemonization!).
        provider: Active model provider ("claude-code", "codex", "grok", …).

            Drives the provider-specific MCP-proxy tool-invocation note and
            the parametrized invocation examples in the agent-identity
            section. If None, falls back to ``settings.model_provider``.
            See docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md.

    Returns:
        The system prompt string (base instructions + environment context)
    """
    # Always load base instructions fresh from package
    base_instructions = get_default_coding_instructions()
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    # Resolve the active provider (explicit arg wins; else global settings).
    # Drives the provider-specific MCP-proxy invocation form for base tools
    # (see docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md).
    effective_provider = provider if provider is not None else settings.model_provider
    # Provider-correct call name for send_message_to_agent (the tool the
    # identity section instructs the model to use). For claude-code/codex
    # this is the mcp__deepworker[_<pid>]__ form; for grok / no-proxy
    # providers it stays the bare name.
    send_msg_form = proxy_tool_form(effective_provider, "send_message_to_agent")
    # The provider-specific note explaining how to invoke base tools (empty
    # for providers without an MCP-proxy).
    provider_tool_note = provider_tool_invocation_note(effective_provider)
    provider_tool_note_block = (
        provider_tool_note + "\n\n" if provider_tool_note else ""
    )

    # Build agent identity section — tells the LLM its OWN registry name.
    # This is critical for inter-agent messaging: the agent must know
    # which name appears as the "<sender>" half of an outgoing
    # "<sender> -> <receiver> :> <msg>" marker (always this agent's
    # own name) and which name it should look for on the receiver side
    # of incoming messages from peers.
    #
    # RC-A: the send_message_to_agent example below uses the PROVIDER-CORRECT
    # call form (``send_msg_form``) instead of a hard-coded bare name, so a
    # claude-code/codex agent never sees the unresolvable short name as the
    # invocation example.
    agent_identity_section = f"""### Your Agent Identity

Your agent name is `{assistant_id}`. This is the name by which other
agents and the process registry refer to you. In inter-agent messages
of the form `<sender> -> <receiver> :> <text>`:

- When YOU send a message via `{send_msg_form}(<peer>, <text>)`,
  the delivered content is automatically formatted as
  `{assistant_id} -> <peer> :> <text>` — your name appears as the
  sender, so you do NOT need to write it yourself.
- When you RECEIVE a user-role message whose content starts with
  `<someone> -> {assistant_id} :> <text>`, it was sent to you by
  another agent named `<someone>`; treat it as a peer request.
  You MUST reply via `{send_msg_form}` unless
  ``reply_back=False`` was specified.

{provider_tool_note_block}"""

    # Build model identity section
    model_identity_section = ""
    if settings.model_name:
        model_identity_section = f"""### Model Identity

You are running as model `{settings.model_name}`"""
        if settings.model_provider:
            model_identity_section += f" (provider: {settings.model_provider})"
        model_identity_section += ".\n"
        if settings.model_context_limit:
            model_identity_section += (
                f"Your context window is {settings.model_context_limit:,} tokens.\n"
            )
        model_identity_section += "\n"

    if sandbox_type:
        # Get provider-specific working directory

        working_dir = get_default_working_dir(sandbox_type)

        working_dir_section = f"""### Current Working Directory

You are operating in a **remote Linux sandbox** at `{working_dir}`.

All code execution and file operations happen in this sandbox environment.

**Important:**
- The CLI is running locally on the user's machine, but you execute code remotely
- Use `{working_dir}` as your working directory for all operations

"""
    else:
        # CRITICAL: MUST use project_memory_dir, NOT Path.cwd()!
        # After daemonization, Path.cwd() returns "/" instead of the original directory.
        # Backend uses project_memory_dir, so prompt MUST match!
        cwd = project_memory_dir if project_memory_dir else Path.cwd()
        working_dir_section = f"""### Current Working Directory

The filesystem backend is currently operating in: `{cwd}`

### File System and Paths

**IMPORTANT - Path Handling:**
- All file paths must be absolute paths (e.g., `{cwd}/file.txt`)
- Use the working directory to construct absolute paths
- Example: To create a file in your working directory, use `{cwd}/research_project/file.md`
- Never use relative paths - always construct full absolute paths

"""  # noqa: E501

    return (
        base_instructions
        + "\n\n---\n\n"
        + agent_identity_section
        + model_identity_section
        + working_dir_section
        + f"""### Skills Directory

Your skills are stored at: `{agent_dir_path}/skills/`
Skills may contain scripts or supporting files. When executing skill scripts with bash, use the real filesystem path:
Example: `bash python {agent_dir_path}/skills/web-research/script.py`

### Human-in-the-Loop Tool Approval

Some tool calls require user approval before execution. When a tool call is rejected by the user:
1. Accept their decision immediately - do NOT retry the same command
2. Explain that you understand they rejected the action
3. Suggest an alternative approach or ask for clarification
4. Never attempt the exact same rejected command again

Respect the user's decisions and work with them collaboratively.

### Web Search Tool Usage

When you use the web_search tool:
1. The tool will return search results with titles, URLs, and content excerpts
2. You MUST read and process these results, then respond naturally to the user
3. NEVER show raw JSON or tool results directly to the user
4. Synthesize the information from multiple sources into a coherent answer
5. Cite your sources by mentioning page titles or URLs when relevant
6. If the search doesn't find what you need, explain what you found and ask clarifying questions

The user only sees your text responses - not tool results. Always provide a complete, natural language answer after using web_search.

### Todo List Management

When using the write_todos tool:
1. Keep the todo list MINIMAL - aim for 3-6 items maximum
2. Only create todos for complex, multi-step tasks that truly need tracking
3. Break down work into clear, actionable items without over-fragmenting
4. For simple tasks (1-2 steps), just do them directly without creating todos
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work
   - Create the todos, let them render, then ask: "Does this plan look good?" or similar
   - Wait for the user's response before marking the first todo as in_progress
   - If they want changes, adjust the plan accordingly
6. Update todo status promptly as you complete each item

The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking."""  # noqa: E501
    )


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format write_file tool call for approval prompt.

    Returns:
        Formatted description string for the write_file tool call.
    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "Overwrite" if Path(file_path).exists() else "Create"
    line_count = len(content.splitlines())

    return f"File: {file_path}\nAction: {action} file\nLines: {line_count}"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format edit_file tool call for approval prompt.

    Returns:
        Formatted description string for the edit_file tool call.
    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    scope = "all occurrences" if replace_all else "single occurrence"
    return f"File: {file_path}\nAction: Replace text ({scope})"


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format web_search tool call for approval prompt.

    Returns:
        Formatted description string for the web_search tool call.
    """
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return (
        f"Query: {query}\nMax results: {max_results}\n\n"
        f"{get_glyphs().warning}  This will use Tavily API credits"
    )


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format fetch_url tool call for approval prompt.

    Returns:
        Formatted description string for the fetch_url tool call.
    """
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return (
        f"URL: {url}\nTimeout: {timeout}s\n\n"
        f"{get_glyphs().warning}  Will fetch and convert web content to markdown"
    )


def _format_task_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format task (subagent) tool call for approval prompt.

    The task tool signature is: task(description: str, subagent_type: str)
    The description contains all instructions that will be sent to the subagent.

    Returns:
        Formatted description string for the task tool call.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:
        description_preview = description[:500] + "..."

    glyphs = get_glyphs()
    separator = glyphs.box_horizontal * 40
    warning_msg = "Subagent will have access to file operations and shell commands"
    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"Task Instructions:\n"
        f"{separator}\n"
        f"{description_preview}\n"
        f"{separator}\n\n"
        f"{glyphs.warning}  {warning_msg}"
    )


def _format_execute_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format execute tool call for approval prompt.

    Returns:
        Formatted description string for the execute tool call.
    """
    args = tool_call["args"]
    command = args.get("command", "N/A")
    # NOTE: Backend knows the correct working directory (project_memory_dir).
    # Don't display Path.cwd() here as it returns "/" after daemonization.
    # The command will execute in the backend's root_dir regardless.
    return f"Execute Command: {command}"


def _format_restart_agent_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    args = tool_call["args"]
    target = args.get("agent_name", "unknown")
    glyphs = get_glyphs()
    return (
        f"Target Agent: {target}\n\n"
        f"{glyphs.warning}  This may stop and restart an idle/waiting deep-worker agent "
        f"using deep-worker --restart. Approve only when the user or procedure "
        f"explicitly requires this agent to be running."
    )


def _format_send_message_to_agent_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format send_message_to_agent tool call for approval prompt.

    The agent is trying to deliver a message to another running
    deep-worker agent. Show the target and a preview of the message so
    the user can decide whether to approve.

    Returns:
        Formatted description string for the send_message_to_agent tool call.
    """
    args = tool_call["args"]
    target = args.get("agent_name", "unknown")
    text = args.get("message", "")

    preview = text if len(text) <= 500 else text[:500] + "..."

    glyphs = get_glyphs()
    separator = glyphs.box_horizontal * 40
    warning_msg = (
        "The message will be prefixed with 'agent:<your-name>:> ' and "
        "delivered to the target agent as a user-role message."
    )
    return (
        f"Target Agent: {target}\n\n"
        f"Message:\n{separator}\n{preview}\n{separator}\n\n"
        f"{glyphs.warning}  {warning_msg}"
    )


def _add_interrupt_on(
    *,
    auto_approve_send_message: bool = False,
) -> dict[str, InterruptOnConfig]:
    """Configure human-in-the-loop interrupt settings for all gated tools.

    Every tool that can have side effects or access external resources
    (shell execution, file writes/edits, web search, URL fetch, task
    delegation) is gated behind an approval prompt unless auto-approve
    is enabled.

    Args:
        auto_approve_send_message: When True, the ``send_message_to_agent``
            tool is NOT included in the returned dict. LangGraph treats
            absence as "no interrupt needed", so the tool runs without
            prompting the user even when global ``auto_approve`` (``-sp``)
            is False. Other side-effect tools remain gated. Intended for
            agents that are expected to cooperate with other agents
            frequently but whose file/shell operations should still
            require confirmation.

    Returns:
        Dictionary mapping tool names to their interrupt configuration.
    """
    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,
    }

    send_message_to_agent_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_send_message_to_agent_description,
    }

    restart_agent_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_restart_agent_description,
    }

    restart_self_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": "Restart the current agent process to pick up fresh code.",
    }

    result: dict[str, InterruptOnConfig] = {
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
        "restart_agent": restart_agent_interrupt_config,
        "restart_self": restart_self_interrupt_config,
    }

    # Only gate send_message_to_agent when we are NOT explicitly told to
    # auto-approve inter-agent messages. Omitting the key means LangGraph
    # will run the tool without asking for approval.
    if not auto_approve_send_message:
        result["send_message_to_agent"] = send_message_to_agent_interrupt_config

    return result


def _maybe_add_delivery_resolve_middleware(
    agent_middleware: list[Any], model: "str | BaseChatModel",
) -> None:
    """Append the claude-code delivery-resolve middleware iff ``model`` is a
    ``ClaudeCodeChatModel``.

    The proxy-MCP early-resolve knowledge lives in the claude_code_provider
    layer (``InterAgentDeliveryResolveMiddleware``); other providers (codex,
    deepseek) and plain string models get nothing. Best-effort: if the
    provider package can't be imported, leave the stack unchanged rather than
    breaking agent construction.
    """
    try:
        from deep_worker.providers.claude_code_provider import ClaudeCodeChatModel
        from deep_worker.providers.claude_code_provider.delivery_resolve_middleware import (
            InterAgentDeliveryResolveMiddleware,
        )
    except Exception:  # noqa: BLE001 — provider optional / import path issue
        return
    if isinstance(model, ClaudeCodeChatModel):
        agent_middleware.append(InterAgentDeliveryResolveMiddleware(model))


def _dedupe_tools_by_name_priority(
    tools: Sequence[Any],
) -> list[Any]:
    """Keep the FIRST tool for each name; drop later duplicates.

    Fix A (DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md): the bound
    tool set of a Codex agent (and a Codex subordinate's TaskManager) contains
    both base/proxy deep-worker tools and the Codex native passthrough tools,
    and ``web_search`` exists in both. LangGraph's ``ToolNode`` resolves a name
    by "last registration wins", and native tools are appended last, so the
    native passthrough silently overrides the proxy tool.

    Resolution is made deterministic by a single, order-preserving "keep first
    by name" rule applied here, at the one funnel both the main agent and the
    TaskManager pass through. Base/proxy tools are assembled before
    ``NATIVE_TOOLS`` is appended, so "keep first" == base/proxy wins, native is
    a fallback only for names not already provided (user decision: on a
    collision the BASE tool wins). Tools without a ``.name`` are passed through
    untouched.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for t in tools:
        name = getattr(t, "name", None)
        if isinstance(name, str):
            if name in seen:
                # A higher-priority tool already owns this name — drop the dup.
                continue
            seen.add(name)
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# History-index catch-up helpers
# ---------------------------------------------------------------------------
#
# Why these live here (and not in semantic_search): the catch-up decision
# is part of the agent's startup flow and must NOT pay the cost of importing
# the embedding model on the cheap "no work to do" path. The semantic_search
# functions (`index_all_history_files`, `history_index_exists`) are fine,
# but a delta computation must stay zero-dependency.
#
# See ``docs/DESIGN_HISTORY_INDEX_CATCHUP_ON_START.md``.

_semantic_search_warned_once = False
"""Log-once flag for the automatic catch-up path (DESIGN_PROVIDERS_FROM_WHEEL §3.6)."""


def _warn_semantic_search_unavailable_once(resolution: Any, exc: Exception | None = None) -> None:
    """Log the semantic-search unavailability exactly once (auto path).

    Per DESIGN_PROVIDERS_FROM_WHEEL §3.6 the automatic startup catch-up must
    NOT raise when the semantic-search extra is missing — it logs once and
    startup continues (history memory simply stays unavailable). Explicit
    requests (``--history-memory`` / reindex) raise ``CapabilityError``
    instead; this warn-once helper covers only the automatic path.
    """
    global _semantic_search_warned_once
    if _semantic_search_warned_once:
        return
    _semantic_search_warned_once = True
    import logging  # noqa: PLC0415

    reason = resolution.unavailable
    if reason is None and exc is not None:
        reason = f"import failed: {exc}"
    remedy = resolution.remedy or "pip install deep-worker[semantic-search]"
    logging.getLogger(__name__).warning(
        "semantic-search unavailable (%s); history memory disabled. Remedy: %s",
        reason,
        remedy,
    )


def _count_pending_history_lines(
    summaries_dir: Path,
    jsonl_files: list[Path],
) -> int:
    """Return total number of JSONL records NOT YET indexed.

    Cheap pre-check: reads ``.history_index/line_hashes.json`` (tiny
    JSON, microseconds) and counts records in each provided JSONL file
    via :func:`semantic_search.history_indexer.count_jsonl_lines` — the
    SAME counter the indexer records with, so the two never desync (the
    ``U+2028`` infinite-reindex bug). Counting is by physical ``\\n``
    lines; no JSON parsing.

    Contract:

    * If ``line_hashes.json`` is missing or unreadable: every line of
      every provided file is treated as pending. Caller will run a
      full index.
    * Files referenced in ``line_hashes.json`` that no longer exist on
      disk are silently ignored — they cannot contribute pending
      lines.
    * If a JSONL file shrank below its recorded count, the current
      line count is reported as pending (NOT a negative number). The
      downstream ``index_history_file`` already handles the
      shrank-file case by re-indexing from scratch.
    """
    import json as _json  # noqa: PLC0415

    if not jsonl_files:
        return 0

    hashes_path = summaries_dir / ".history_index" / "line_hashes.json"
    try:
        line_counts: dict[str, int] = _json.loads(
            hashes_path.read_text(encoding="utf-8"),
        )
    except (FileNotFoundError, OSError, _json.JSONDecodeError):
        line_counts = {}

    # Single source of truth: count JSONL records the SAME way the indexer
    # does (physical ``\n`` lines, NOT ``str.splitlines()``). Counting them
    # differently here caused a permanent off-by-N for files whose record
    # content embedded ``U+2028`` etc., which re-indexed the whole history on
    # every start. See semantic_search.history_indexer.count_jsonl_lines.
    #
    # Auto catch-up contract (DESIGN_PROVIDERS_FROM_WHEEL §3.6): when the
    # semantic-search extra is unavailable, do NOT raise — report "nothing
    # pending" (log-once) so startup continues with history memory disabled.
    from deep_worker_cli.provider_paths import ensure_provider_importable

    resolution = ensure_provider_importable("deep_worker.providers.semantic_search")
    if resolution.unavailable:
        _warn_semantic_search_unavailable_once(resolution)
        return 0
    try:
        from deep_worker.providers.semantic_search.history_indexer import count_jsonl_lines  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - defensive (resolve said available)
        _warn_semantic_search_unavailable_once(resolution, exc)
        return 0

    pending = 0
    for jsonl in jsonl_files:
        current = count_jsonl_lines(jsonl)
        if current == 0 and not jsonl.exists():
            continue
        recorded = int(line_counts.get(str(jsonl), 0))
        delta = current - recorded
        if delta > 0:
            pending += delta
        elif current < recorded:
            # File shrank → treat the whole current content as pending,
            # so the caller re-indexes it. Never report negative.
            pending += current
    return pending


def _should_catchup_history_index(
    summaries_dir: Path,
) -> tuple[bool, int]:
    """Return ``(needs_catchup, pending_lines)`` for startup decision.

    Used by ``create_cli_agent`` to decide whether to run
    ``index_all_history_files`` on agent start. Returns ``(False, 0)``
    when there's nothing to do (most restarts after the first one), so
    the caller can skip both the embedding-model load and the progress
    UI entirely.

    Returns ``(True, N)`` when:

    * The summaries dir contains ``*.jsonl`` files, AND
    * Their combined line count exceeds what's recorded in
      ``.history_index/line_hashes.json`` (or that file is missing).

    Returns ``(False, 0)`` when:

    * The summaries dir doesn't exist, OR
    * There are no JSONL files, OR
    * The index is fully up to date.
    """
    if not summaries_dir.exists() or not summaries_dir.is_dir():
        return False, 0
    jsonl_files = sorted(summaries_dir.glob("*.jsonl"))
    if not jsonl_files:
        return False, 0
    pending = _count_pending_history_lines(summaries_dir, jsonl_files)
    return (pending > 0), pending


def _provider_of_model(model: "str | BaseChatModel") -> str | None:
    """Best-effort detection of the active provider from a model object.

    Used to drive the provider-specific MCP-proxy tool-invocation forms in
    the system prompt and InterAgentReplyMiddleware. Detection is by class
    name so we don't hard-import every optional provider package. Returns
    None when the provider can't be determined (callers then fall back to
    ``settings.model_provider`` / bare names).
    """
    cls = type(model).__name__
    mapping = {
        "ClaudeCodeChatModel": "claude-code",
        "CodexChatModel": "codex",
        "GrokChatModel": "grok",
    }
    return mapping.get(cls)


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    auto_approve: bool = False,
    auto_approve_send_message: bool = False,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
    project_memory_dir: Path | None = None,
    enable_history_memory: bool = False,
    external_mcp_tools: Sequence[BaseTool] | None = None,
    external_mcp_auto_approved_names: set[str] | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.

    This is the main entry point for creating a deepagents CLI agent, usable
    both internally and from external code (e.g., benchmarking frameworks).

    Args:
        model: LLM model to use (e.g., `'anthropic:claude-sonnet-4-5-20250929'`)
        assistant_id: Agent identifier for memory/state storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution
            (e.g., `ModalBackend`).

            If `None`, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider
            (`'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).
            Used for system prompt generation.
        system_prompt: Override the default system prompt.

            If `None`, generates one based on `sandbox_type` and `assistant_id`.
        auto_approve: If `True`, no tools trigger human-in-the-loop
            interrupts — all calls (shell execution, file writes/edits,
            web search, URL fetch) run automatically.

            If `False`, tools pause for user confirmation via the approval menu.
            See `_add_interrupt_on` for the full list of gated tools.
        auto_approve_send_message: If `True` AND `auto_approve` is
            `False`, the `send_message_to_agent` inter-agent messaging
            tool is auto-approved while every other side-effect tool
            still requires confirmation. Has no effect when
            `auto_approve=True` (everything is already auto-approved).
            Default: `False` (send_message_to_agent is gated).
        enable_memory: Enable `MemoryMiddleware` for persistent memory
        enable_skills: Enable `SkillsMiddleware` for custom agent skills
        enable_shell: Enable shell execution via `CLIShellBackend`
            (only in local mode). When enabled, the `execute` tool is available.
        checkpointer: Optional checkpointer for session persistence.

            If `None`, uses `InMemorySaver` (no persistence across
            CLI invocations).
        project_memory_dir: REQUIRED working directory for the agent (project root).
            Used for FilesystemBackend, skills, memory sources, semantic index
            lookup under .vector_index/, and AGENTS.md. Caller (AgentProcess)
            MUST pass explicit value (e.g. launch working_dir); None raises.
        enable_history_memory: If True, enables HistoryMemoryMiddleware and
            performs startup catch-up indexing so the history_memory tool
            can search pre-summarization conversation history.
        external_mcp_tools: Ready MCP adapter tools owned by the calling
            `AgentProcess` lifecycle.
        external_mcp_auto_approved_names: External MCP tool names that bypass
            HITL even when global auto-approve is disabled. Global
            `auto_approve=True` / `-sp` still bypasses all HITL.

    Returns:
        2-tuple of `(agent_graph, backend)`

            - `agent_graph`: Configured LangGraph Pregel instance ready
                for execution
            - `composite_backend`: `CompositeBackend` for file operations

    Raises:
        ValueError: if project_memory_dir is None.
    """
    # Import deepagents here to avoid import errors when not needed
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware import (
        MemoryMiddleware,
        SkillsMiddleware,
        SummarizationMiddleware,
    )
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

    # Import ProjectMemoryMiddleware from local deepagents if available
    try:
        # Try local version first (has ProjectMemoryMiddleware)
        from deepagents.middleware.project_memory import ProjectMemoryMiddleware
    except (ImportError, ModuleNotFoundError):
        # Fallback: define a no-op placeholder that is compatible with AgentMiddleware
        from langchain.agents.middleware.types import AgentMiddleware

        class ProjectMemoryMiddleware(AgentMiddleware):
            """Placeholder when ProjectMemoryMiddleware is unavailable.

            CRITICAL: This placeholder is ONLY for type compatibility.
            In production, ProjectMemoryMiddleware MUST be available from local deepagents.
            If you see this being used, CHECK YOUR PYTHON PATH - you are testing
            WRONG CODE (installed packages instead of repository code)!
            """
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.tools = []  # No tools available in placeholder

    tools = list(tools or [])

    # External MCP tools (see docs/DESIGN_EXTERNAL_MCP_TOOLS.md). These tools
    # are already started by the caller (``AgentProcess.start()``) via
    # ``ExternalMCPClientPool.create(...)`` — we simply append them to the
    # common ``tools`` list; from there they flow as regular BaseTool through
    # SafeToolCallMiddleware / BoundedToolCallMiddleware / interrupt_on.
    _external_mcp_names: set[str] = set()
    if external_mcp_tools:
        for t in external_mcp_tools:
            tools.append(t)
            _external_mcp_names.add(t.name)

    # CRITICAL: project_memory_dir is REQUIRED - no fallback!
    if project_memory_dir is None:
        msg = (
            "project_memory_dir is REQUIRED and cannot be None. "
            "Backend MUST know its working directory unambiguously."
        )
        raise ValueError(msg)

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            # Create empty file for user customizations
            # Base instructions are loaded fresh from get_system_prompt()
            agent_md.touch()

    # Skills directories (if enabled)
    skills_dir = None
    project_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        project_skills_dir = settings.get_project_skills_dir()

    # Load custom subagents from filesystem
    custom_subagents: list[SubAgent | CompiledSubAgent] = []
    user_agents_dir = settings.get_user_agents_dir(assistant_id)
    project_agents_dir = settings.get_project_agents_dir()

    for subagent_meta in list_subagents(
        user_agents_dir=user_agents_dir,
        project_agents_dir=project_agents_dir,
    ):
        subagent: SubAgent = {
            "name": subagent_meta["name"],
            "description": subagent_meta["description"],
            "system_prompt": subagent_meta["system_prompt"],
        }
        if subagent_meta["model"]:
            subagent["model"] = subagent_meta["model"]
        custom_subagents.append(subagent)

    # Detect the active provider from the model object (drives the
    # provider-specific MCP-proxy tool-invocation forms in the prompt and
    # in InterAgentReplyMiddleware). Falls back to settings / bare names
    # inside the helpers when None.
    active_provider = _provider_of_model(model)

    # Build middleware stack based on enabled features
    agent_middleware = []

    # FIRST before_model middleware (graph entry_node): close any tool-call
    # left dangling by a previous, interrupted session BEFORE the model or
    # ToolNode runs, so a stale tool is never silently re-executed on restart.
    # Must precede every other before_model hook so they all see a normalised,
    # tool-call-consistent state. See
    # docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md (Fix B).
    agent_middleware.append(DanglingToolCallMiddleware())

    # Treat a successful send_message_to_agent tool call as the agent's
    # final response for this turn — injects an echo AIMessage (so Full
    # Dialog and other history clients can see it) and jumps to the end
    # of the graph, skipping the redundant model call that would
    # otherwise produce a pointless filler assistant message.
    # See docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md for details.
    from deep_worker_cli.middleware.send_message_completion import (
        SendMessageCompletionMiddleware,
    )
    agent_middleware.append(SendMessageCompletionMiddleware())

    # Inject cognitive hints for incoming inter-agent messages: guides the
    # receiving agent to either reply via send_message_to_agent (reply_back=True)
    # or respond directly to the user (reply_back=False).
    # Must come AFTER DanglingToolCall & SendMessageCompletion, and BEFORE
    # SummarizationMiddleware so the hint is visible before summarization.
    from deep_worker_cli.middleware.inter_agent_reply import (
        InterAgentReplyMiddleware,
    )
    # Provider-correct reply-channel form for send_message_to_agent, so the
    # injected reply hint names a tool the CLI can actually resolve
    # (claude-code/codex MCP prefix; grok bare name + "MCP tool, don't search
    # disk" clarification). See
    # docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md.
    agent_middleware.append(
        InterAgentReplyMiddleware(
            name=assistant_id,
            reply_tool_form=proxy_tool_form(
                active_provider, "send_message_to_agent"
            ),
            grok_mcp_note=(active_provider == "grok"),
            grok_receive_completion=(active_provider == "grok"),
        )
    )

    # Unconditionally register history access tools (history_range,
    # history_at, history_slice) for ALL agents. These tools let the agent
    # read pre-summarization history directly from markdown section files
    # regardless of the --history-memory flag.
    from deep_worker_cli.middleware.history_access import (
        HistoryAccessMiddleware,
    )
    agent_middleware.append(
        HistoryAccessMiddleware(assistant_id=assistant_id)
    )

    # claude-code only: resolve the inter-agent send Future at DELIVERY time
    # (not on the next round-trip), so the Claude CLI gets its tool_result
    # immediately and the proxy-MCP timeout bounds delivery — not the
    # recipient's work. Provider-agnostic stack stays untouched.
    # See deep_worker_web/docs/DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 2).
    _maybe_add_delivery_resolve_middleware(agent_middleware, model)

    # Add memory middleware
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        # CRITICAL: settings.get_project_agent_md_path() depends on the
        # global settings.project_root which is derived from cwd at module
        # import time.  After daemonize chdirs to "/" that resolution yields
        # None, so the project AGENTS.md is silently skipped.  Fall back to
        # the explicit --dir / working_dir value which is always correct.
        project_agent_md = settings.get_project_agent_md_path()
        if project_agent_md:
            memory_sources.append(str(project_agent_md))
        project_md_from_wd = project_memory_dir / "AGENTS.md"
        if project_md_from_wd.exists():
            memory_sources.append(str(project_md_from_wd))

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(root_dir=project_memory_dir),
                sources=memory_sources,
            )
        )

    # Add skills middleware
    if enable_skills:
        # Built-in first (lowest precedence), then user, then project (highest)
        sources = [str(settings.get_built_in_skills_dir())]
        sources.append(str(skills_dir))
        if project_skills_dir:
            sources.append(str(project_skills_dir))

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(root_dir=project_memory_dir),
                sources=sources,
            )
        )

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            # Use CLIShellBackend for filesystem + shell execution.
            # Provides `execute` tool via FilesystemMiddleware with per-command
            # timeout support.
            # CRITICAL: MUST use project_memory_dir, NOT Path.cwd()!
            backend = CLIShellBackend(
                root_dir=project_memory_dir,
                inherit_env=True,
                env=shell_env,
            )
        else:
            # No shell access - use plain FilesystemBackend
            # CRITICAL: MUST use project_memory_dir, NOT Path.cwd()!
            backend = FilesystemBackend(root_dir=project_memory_dir)

        # Local context middleware (git info, directory tree, etc.)
        agent_middleware.append(LocalContextMiddleware())

        # Project memory middleware (semantic search if index exists)
        # CRITICAL: NO FALLBACK - project_memory_dir is guaranteed to be set!
        agent_middleware.append(ProjectMemoryMiddleware(working_dir=project_memory_dir))

        # History memory middleware (semantic search over summarised history)
        # NOTE: Created AFTER startup indexing below so that _has_index check
        # returns the correct result.
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalBackend, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

        # Project memory middleware (if working directory is specified)
        if project_memory_dir is not None:
            agent_middleware.append(ProjectMemoryMiddleware(working_dir=project_memory_dir))

        # History memory middleware (if enabled)
        # NOTE: Created AFTER startup indexing below.

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id,
            sandbox_type=sandbox_type,
            project_memory_dir=project_memory_dir,
            provider=active_provider,
        )

    # Optional LSP navigation (see lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md §4.4).
    # Mixed in AFTER the pool tool names are known, and ONLY if the symbolic
    # tools actually arrived: without LSP the prompt does not change by a
    # single byte (optional, like the semantic index).
    # We use the actual (already mangled) names — the agent calls exactly
    # those.
    #
    # NB: deliberately OUTSIDE the ``if system_prompt is None`` branch — the
    # fragment is also added to a prompt passed explicitly by the caller.
    # Having a custom base prompt is no reason to hide the documentation on
    # available capabilities from the agent (confirmed in code review).
    # Behavior lock: tests/unit_tests/lsp/test_prompt_injection_point.py
    if _external_mcp_names:
        from deep_worker_cli.lsp.prompt import lsp_prompt_fragment  # noqa: PLC0415

        _lsp_fragment = lsp_prompt_fragment(sorted(_external_mcp_names))
        if _lsp_fragment:
            system_prompt = f"{system_prompt}\n\n{_lsp_fragment}"

    # Configure interrupt_on based on auto_approve / auto_approve_send_message
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve:
        # No interrupts - all tools run automatically. Per-tool overrides
        # (like auto_approve_send_message) are implied and have no extra
        # effect here. EXTERNAL MCP TOOLS ARE NOT INTERRUPTED EITHER under -sp
        # (user rule 2026-06-23) — `interrupt_on={}` is exactly what ensures
        # this: absence of the key = no HITL.
        interrupt_on = {}
    else:
        # Full HITL for destructive operations, with optional per-tool
        # carve-out for send_message_to_agent.  External MCP tools land
        # here too (see DESIGN_EXTERNAL_MCP_TOOLS.md §3.6) — except
        # those whose server is marked ``auto_approve: true``.
        from deep_worker_cli.external_mcp.interrupt import (
            build_interrupt_on_with_external,
        )
        interrupt_on = cast(
            "dict[str, bool | InterruptOnConfig]",
            build_interrupt_on_with_external(
                auto_approve=False,
                auto_approve_send_message=auto_approve_send_message,
                external_tool_names=_external_mcp_names,
                external_auto_approved_names=external_mcp_auto_approved_names or set(),
            ),
        )

    # Set up composite backend with routing
    # For local FilesystemBackend, route large tool results to /tmp to avoid polluting
    # the working directory. For sandbox backends, no special routing is needed.
    if sandbox is None:
        # Local mode: Route large results to a unique temp directory
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_large_results_"),
            virtual_mode=True,
        )
        # NOTE: the previous "/conversation_history/" route to an
        # ephemeral tempdir was removed in favour of a persistent,
        # human-inspectable on-disk location under
        # ~/.deep_worker/<agent>/summaries/ — SummarizationMiddleware
        # now writes directly to that absolute path via the default
        # backend. See DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md.
        composite_backend = CompositeBackend(
            default=backend,
            routes={
                "/large_tool_results/": large_results_backend,
            },
        )
    else:
        # Sandbox mode: No special routing needed
        composite_backend = CompositeBackend(
            default=backend,
            routes={},
        )

    # Create the agent
    # Use provided checkpointer or fallback to InMemorySaver
    if sandbox is None and enable_shell:
        # Patch FilesystemMiddleware so the SDK constructs our subclass with
        # per-command timeout support on the execute tool. Only needed in local
        # shell mode -- remote sandbox backends do not accept the timeout kwarg.
        patch_filesystem_middleware()
    final_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

    # Per-agent on-disk summaries directory. The path quoted in the
    # summarization HumanMessage will be this absolute filesystem path
    # (see DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md). Created eagerly
    # so the backend write never races a missing-dir error.
    from importlib import import_module

    summaries_mod = cast(Any, import_module("deep_worker_cli.summaries"))
    summaries_dir = summaries_mod.ensure_summaries_dir(assistant_id)
    composite_backend = summaries_mod.PrivateSummariesBackend(
        composite_backend,
        summaries_dir,
    )

    # Fix A: deterministic tool-name resolution. Drop later duplicates so a
    # base/proxy tool always wins over a same-named Codex native passthrough
    # (e.g. ``web_search``). See
    # docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md (Fix A).
    tools = _dedupe_tools_by_name_priority(tools)

    # Wire up history memory indexer as summarization callback
    if enable_history_memory:
        # Explicit request (--history-memory flag) → hard-fail with an
        # actionable remedy when the semantic-search extra is missing
        # (DESIGN_PROVIDERS_FROM_WHEEL §3.6).
        from deep_worker_cli.provider_paths import ensure_provider_importable
        from deep_worker_cli.exceptions import CapabilityError

        _semantic_resolution = ensure_provider_importable(
            "deep_worker.providers.semantic_search"
        )
        if _semantic_resolution.unavailable:
            raise CapabilityError(
                capability="semantic-search",
                remedy="pip install deep-worker[semantic-search]",
            )
        from deep_worker.providers.semantic_search.history_indexer import index_all_history_files
        _on_summarized = lambda path_str: index_all_history_files(summaries_dir)  # noqa: E731

        # --- Start-up catch-up of existing history ---
        #
        # Run on EVERY agent start (not only when no index exists). The
        # `_on_summarized` callback only fires during a live process;
        # between two starts the JSONL can grow without anyone updating
        # the index. So the next start must catch up.
        #
        # ``_should_catchup_history_index`` does a cheap line-count diff
        # vs ``.history_index/line_hashes.json`` — microseconds — and
        # returns ``(False, 0)`` when there's nothing to do, in which
        # case we skip both the embedding-model load AND the progress UI.
        # See ``docs/DESIGN_HISTORY_INDEX_CATCHUP_ON_START.md``.
        from deep_worker.providers.semantic_search.history_indexer import history_index_exists

        summaries_path = Path(summaries_dir)
        needs_catchup, _pending_lines = _should_catchup_history_index(summaries_path)

        if needs_catchup:
            from deep_worker_cli.config import console

            jsonl_files = list(summaries_path.glob("*.jsonl"))
            try:
                file_count = len(jsonl_files)
                if file_count >= 3:
                    from rich.progress import (
                        BarColumn,
                        Progress,
                        SpinnerColumn,
                        TextColumn,
                    )

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            "Indexing conversation history...", total=file_count
                        )

                        def _on_progress(filename: str) -> None:
                            progress.update(
                                task,
                                description=f"Indexing {filename}...",
                                advance=1,
                            )

                        index_all_history_files(
                            summaries_dir=summaries_path,
                            on_progress=_on_progress,
                        )
                else:
                    console.print("[dim]Indexing conversation history...[/dim]")
                    index_all_history_files(summaries_dir=summaries_path)
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to index conversation history. "
                    "history_memory tool will not be available.",
                    exc_info=True,
                )

        # Enable history memory middleware only if index exists
        history_ready = history_index_exists(summaries_path)
        from deep_worker_cli.middleware.history_memory import HistoryMemoryMiddleware

        agent_middleware.append(
            HistoryMemoryMiddleware(
                assistant_id=assistant_id,
                enabled=history_ready,
            )
        )
    else:
        _on_summarized = None

    # Tool-dispatch safety net — MUST be the last entry in agent_middleware
    # so it is the innermost ``awrap_tool_call`` wrapper.  Per the
    # No-Agent-Error-State Policy (docs/POLICIES.md, §1) any exception
    # raised by a tool must be converted into a ToolMessage(status="error")
    # rather than propagating up to the agent invoke catch-all (which used
    # to flip the whole agent into AgentStatus.ERROR).
    from deep_worker_cli.middleware.safe_tool_call import SafeToolCallMiddleware
    agent_middleware.append(SafeToolCallMiddleware())

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=final_checkpointer,
        subagents=custom_subagents or None,
        summarization_history_prefix=str(summaries_dir),
        summarization_on_summarized=_on_summarized,
    ).with_config(config)
    return agent, composite_backend
