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

    Returns:
        The system prompt string (base instructions + environment context)
    """
    # Always load base instructions fresh from package
    base_instructions = get_default_coding_instructions()
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    # Build agent identity section — tells the LLM its OWN registry name.
    # This is critical for inter-agent messaging: the agent must know
    # which name appears as the "<sender>" half of an outgoing
    # "<sender> -> <receiver> :> <msg>" marker (always this agent's
    # own name) and which name it should look for on the receiver side
    # of incoming messages from peers.
    agent_identity_section = f"""### Your Agent Identity

Your agent name is `{assistant_id}`. This is the name by which other
agents and the process registry refer to you. In inter-agent messages
of the form `<sender> -> <receiver> :> <text>`:

- When YOU send a message via `send_message_to_agent(<peer>, <text>)`,
  the delivered content is automatically formatted as
  `{assistant_id} -> <peer> :> <text>` — your name appears as the
  sender, so you do NOT need to write it yourself.
- When you RECEIVE a user-role message whose content starts with
  `<someone> -> {assistant_id} :> <text>`, it was sent to you by
  another agent named `<someone>`; treat it as a peer request.

"""

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

    result: dict[str, InterruptOnConfig] = {
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
    }

    # Only gate send_message_to_agent when we are NOT explicitly told to
    # auto-approve inter-agent messages. Omitting the key means LangGraph
    # will run the tool without asking for approval.
    if not auto_approve_send_message:
        result["send_message_to_agent"] = send_message_to_agent_interrupt_config

    return result


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
        project_memory_dir: Directory to check for semantic search index.

            If `None`, defaults to `Path.cwd()`. When a semantic index
            exists at `{project_memory_dir}/.vector_index/`, the
            `project_memory` tool becomes available for code search.

    Returns:
        2-tuple of `(agent_graph, backend)`

            - `agent_graph`: Configured LangGraph Pregel instance ready
                for execution
            - `composite_backend`: `CompositeBackend` for file operations
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

    tools = tools or []

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

    # Build middleware stack based on enabled features
    agent_middleware = []

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

    # Add memory middleware
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        project_agent_md = settings.get_project_agent_md_path()
        if project_agent_md:
            memory_sources.append(str(project_agent_md))

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
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalBackend, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

        # Project memory middleware (if working directory is specified)
        if project_memory_dir is not None:
            agent_middleware.append(ProjectMemoryMiddleware(working_dir=project_memory_dir))

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id,
            sandbox_type=sandbox_type,
            project_memory_dir=project_memory_dir,
        )

    # Configure interrupt_on based on auto_approve / auto_approve_send_message
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve:
        # No interrupts - all tools run automatically. Per-tool overrides
        # (like auto_approve_send_message) are implied and have no extra
        # effect here.
        interrupt_on = {}
    else:
        # Full HITL for destructive operations, with optional per-tool
        # carve-out for send_message_to_agent.
        # `_add_interrupt_on` returns `dict[str, InterruptOnConfig]` — a
        # narrower type than the `dict[str, bool | InterruptOnConfig]`
        # variable annotation. `dict` is invariant in its value type, so a
        # direct assignment fails ty. Cast widens the value type
        # explicitly; the runtime dict itself is unchanged.
        interrupt_on = cast(
            "dict[str, bool | InterruptOnConfig]",
            _add_interrupt_on(auto_approve_send_message=auto_approve_send_message),
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
        # human-inspectable on-disk location — the per-agent summaries
        # directory under the tool's data directory.
        # SummarizationMiddleware now writes directly to that absolute
        # path via the default backend. See
        # DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md.
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
    summaries_dir = Path("~/.deep_worker").expanduser() / assistant_id / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

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
    ).with_config(config)
    return agent, composite_backend
