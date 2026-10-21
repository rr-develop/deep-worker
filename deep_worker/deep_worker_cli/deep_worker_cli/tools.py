"""Custom tools for the CLI agent."""

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)
from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError

from deep_worker_cli.config import WebSearchPolicy, settings
from deep_worker_cli.exceptions import AgentNotFoundError, CapabilityError, IPCConnectionError
from deep_worker_cli.process.registry import ProcessRegistry

logger = logging.getLogger(__name__)

# Initialize Tavily client if API key is available
tavily_client = (
    TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None
)


# ============================================================================
# Universal web-search delegation tool — lazy, tri-state (DESIGN §3.3)
# ============================================================================
# The tool delegates web searches to Claude Code's built-in WebSearch
# capability via the Claude Agent SDK fork. It works with ANY LLM provider
# (DeepSeek, OpenAI, Gemini, Claude, etc.) and needs NO extra API key.
#
# The provider package + SDK are resolved LAZILY, under the operator's
# tri-state policy (auto/enabled/disabled). tools.py must import cleanly
# even when the claude provider / SDK are absent — the old top-level
# ``raise ImportError`` + eager ``WebSearchDelegationTool(...)`` made the
# whole CLI unbootable without the SDK.
#
# SDK-error classification (STRICT order, rev 4 precision-fix):
#   1. ImportError with ``name == "claude_agent_sdk"`` (exactly the root)
#      -> sdk_missing;
#   2. the mandatory fork-private module ``claude_agent_sdk._mcp_context``
#      is absent -> sdk_incompatible (official SDK installed, not the fork);
#   3. ANYTHING else (SyntaxError, AttributeError, other submodule defects)
#      -> re-raise (internal defect, not a missing capability).
#
# REGRESSION WARNING:
# - ToolResultBlock is in UserMessage, NOT AssistantMessage!
# - See web_search_delegation.py:_extract_results() for critical implementation
# - Always test with: pytest tests/integration_ai/test_websearch_delegation_deepseek.py


@dataclass(frozen=True)
class WebSearchAvailability:
    """Result of resolving the universal web-search tool under a policy."""

    tool: BaseTool | None
    reason: str  # "available" | "sdk_missing" | "sdk_incompatible" | "provider_missing" | "disabled"
    remedy: str | None


_web_search_warned_once = False


def _sdk_fork_status() -> tuple[str, str | None]:
    """Return ``(status, remedy)`` for the Claude Agent SDK fork.

    Status is one of ``"ok"``, ``"sdk_missing"``, ``"sdk_incompatible"``.
    Re-raises any non-import error (internal defect).
    """
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except ImportError as exc:
        if getattr(exc, "name", None) == "claude_agent_sdk":
            return (
                "sdk_missing",
                "Claude Agent SDK not found. Install the SDK fork "
                "(e.g. CLAUDE_AGENT_SDK_SRC=<path>/claude_sdk/src) or run "
                "the agent without web search.",
            )
        raise  # internal defect, not a missing capability

    try:
        import claude_agent_sdk._mcp_context  # noqa: PLC0415, F401
    except ImportError:
        return (
            "sdk_incompatible",
            "Installed claude_agent_sdk is NOT the fork: the mandatory "
            "private module 'claude_agent_sdk._mcp_context' is missing. "
            "Install the deep-worker SDK fork (CLAUDE_AGENT_SDK_SRC).",
        )
    return "ok", None


_WEB_SEARCH_PROVIDER = "deep_worker.providers.claude_code_sdk_provider.web_search"


def _web_search_provider_status() -> tuple[str, str | None]:
    """Return ``(status, remedy)`` for the package that hosts the tool.

    The delegation tool lives in the claude-code-sdk provider, which is not
    part of every distribution. Its absence is a missing capability
    (``"provider_missing"``), not a defect; a broken submodule of a present
    provider is re-raised.
    """
    try:
        import deep_worker.providers.claude_code_sdk_provider.web_search  # noqa: PLC0415, F401
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name == _WEB_SEARCH_PROVIDER or _WEB_SEARCH_PROVIDER.startswith(name + "."):
            return (
                "provider_missing",
                "The web_search delegation tool requires the claude-code-sdk "
                "provider, which is not part of this distribution.",
            )
        raise
    return "ok", None


def resolve_web_search_tool(
    policy: WebSearchPolicy | str = WebSearchPolicy.AUTO,
) -> WebSearchAvailability:
    """Resolve the universal web-search tool under a tri-state policy.

    - ``AUTO``: return the tool when the SDK fork is available; otherwise
      log exactly once and return ``tool=None`` (the agent still starts).
    - ``ENABLED``: return the tool, or raise ``CapabilityError`` with the
      remedy when the SDK is unavailable.
    - ``DISABLED``: NEVER touch the claude provider or the SDK; return
      ``tool=None`` without any import attempt.

    The claude provider module is imported lazily, only when the SDK fork
    is confirmed available and the policy permits resolution.
    """
    global _web_search_warned_once

    if isinstance(policy, str):
        policy = WebSearchPolicy(policy)

    if policy is WebSearchPolicy.DISABLED:
        return WebSearchAvailability(None, "disabled", None)

    status, remedy = _sdk_fork_status()
    if status == "ok":
        status, remedy = _web_search_provider_status()
    if status != "ok":
        if policy is WebSearchPolicy.ENABLED:
            raise CapabilityError(capability="web_search", remedy=remedy)
        # AUTO: log once, skip the tool, agent keeps starting.
        if not _web_search_warned_once:
            logger.warning(
                "web_search tool unavailable (%s): %s — skipping. "
                "Use --enable-web-search to fail loudly instead, or "
                "--disable-web-search to silence this.",
                status,
                remedy,
            )
            _web_search_warned_once = True
        return WebSearchAvailability(None, status, remedy)

    from deep_worker.providers.claude_code_sdk_provider.web_search import (  # noqa: PLC0415
        WebSearchDelegationTool,
    )

    return WebSearchAvailability(
        WebSearchDelegationTool(verbose=False, timeout=120.0),
        "available",
        None,
    )


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs: dict[str, Any] = {}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(method.upper(), url, timeout=timeout, **kwargs)

        try:
            content = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepWorker/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}


# ===========================================================================
# IMAGE ANALYSIS TOOL
# ===========================================================================

# Supported image formats and their MIME types
_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Anthropic API limit
_MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def read_image(file_path: str) -> list[dict[str, Any]] | str:
    """Read an image file and return it for visual analysis by the model.

    The image is base64-encoded and returned as multimodal content blocks
    that Claude can analyze visually.

    Supports PNG, JPEG, GIF, WebP formats. Maximum file size: 20 MB.

    Args:
        file_path: Absolute path to the image file.

    Returns:
        List of content blocks [text_metadata, image_data] on success,
        or error string on failure.
    """
    import base64

    path = Path(file_path)

    # Validate file exists
    if not path.exists():
        return f"Error: File not found: {file_path}"

    # Validate format
    ext = path.suffix.lower()
    if ext not in _IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(_IMAGE_MIME_TYPES.keys()))
        return f"Error: Unsupported image format: {ext}. Supported: {supported}"

    # Validate size
    file_size = path.stat().st_size
    if file_size > _MAX_IMAGE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return f"Error: Image too large: {size_mb:.1f} MB. Maximum: 20 MB"

    # Read and encode
    try:
        image_data = path.read_bytes()
        b64_data = base64.b64encode(image_data).decode("ascii")
        mime_type = _IMAGE_MIME_TYPES[ext]

        # Try to get dimensions
        dimensions = ""
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as img:
                dimensions = f" ({img.width}x{img.height})"
        except Exception:
            pass

        return [
            {"type": "text", "text": f"Image: {path.name}{dimensions}, {mime_type}, {file_size} bytes"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
        ]
    except Exception as e:
        return f"Error: Failed to read image: {e}"


# ===========================================================================
# AGENT RESTART TOOL
# ===========================================================================


class _RestartAgentArgs(BaseModel):
    agent_name: str = Field(
        ...,
        description=(
            "Registry/session name of the agent to ensure is running by "
            "using the same lifecycle as 'deep-worker --restart <agent_name>'."
        ),
    )


def _map_restart_error(stderr: str) -> str:
    text = stderr.lower()
    if "no previous launch configuration found" in text:
        return "no_launch_config"
    if "actively running" in text or "status: running" in text:
        return "actively_running"
    return "restart_failed"


def make_restart_agent_tool(sender_name: str) -> BaseTool:
    @tool(
        "restart_agent",
        args_schema=_RestartAgentArgs,
    )
    async def restart_agent(agent_name: str) -> dict[str, Any]:
        """Restart or start a named deep-worker agent.

        This tool runs the equivalent of `deep-worker --restart <agent_name>`.

        Usage policy:
        - Do NOT use this tool by default.
        - Use it only when the human user explicitly asks to restart an agent,
          explicitly asks to ensure an agent is running, or provides a procedure
          that explicitly requires a named agent to be in a running state.
        - Do not use it for speculative cleanup, optimization, or autonomous
          process management.
        """
        if not isinstance(agent_name, str) or not agent_name.strip():
            return {
                "success": False,
                "agent_name": "",
                "already_running": False,
                "error": "invalid_argument",
                "message": "agent_name must be a non-empty string",
            }

        target = agent_name.strip()
        if target == sender_name:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": "self_restart_forbidden",
                "message": "Agents cannot restart themselves.",
            }

        registry = ProcessRegistry()
        existing = registry.lookup(target)
        # A running agent is NOT a reason to bail out. This tool documents
        # itself as the equivalent of ``deep-worker --restart <name>``, and
        # that path deliberately stops a live agent (``force=True``) before
        # relaunching it — restarting a busy agent is what the destructive
        # tool is *for*, and the only way to interrupt a stuck turn.
        #
        # Returning early here degraded the tool into "start it if it is
        # dead": a live agent was never restarted, so a caller asking for
        # fresh code kept talking to the old process. It also returned
        # ``success=True`` alongside ``error="already_running"``, so the
        # refusal read as success (owner report, 2026-09-26: an agent could
        # not be moved onto the status fix).
        previous_pid = existing.pid if existing is not None else None

        command = shutil.which("deep-worker")
        if command is None:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": "command_unavailable",
                "message": "deep-worker command not found on PATH",
            }

        try:
            completed = subprocess.run(
                [command, "--restart", target],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": "restart_timeout",
                "message": "deep-worker --restart timed out",
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
            }

        if completed.returncode != 0:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": _map_restart_error(completed.stderr),
                "message": completed.stderr.strip() or "deep-worker --restart failed",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        # Linux inotify + asyncio to watch the registry file for changes.
        info = await registry.wait_for_registration(target, timeout=30)
        # Seeing the *same* PID we set out to replace is not a restart: the
        # wait can resolve against the still-present entry of the old
        # process. Reporting that as success would promise fresh code while
        # the old process is still serving.
        if info is not None and info.pid == previous_pid:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": "restart_timeout",
                "message": (
                    f"Agent '{target}' still runs the previous process "
                    f"(PID {previous_pid}); restart did not replace it."
                ),
                "previous_pid": previous_pid,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        if info is not None:
            return {
                "success": True,
                "agent_name": target,
                "already_running": False,
                "message": f"Agent '{target}' restarted and registered (PID {info.pid}).",
                "restarted_pid": info.pid,
                "previous_pid": previous_pid,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        info = registry.lookup(target)
        if info is not None:
            return {
                "success": False,
                "agent_name": target,
                "already_running": False,
                "error": "restart_timeout",
                "message": f"Agent '{target}' did not register after restart.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        return {
            "success": False,
            "agent_name": target,
            "already_running": False,
            "error": "restart_timeout",
            "message": f"Agent '{target}' did not register after restart.",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    return restart_agent


# ===========================================================================
# SELF-RESTART TOOL (Issue #89)
# ===========================================================================


def write_restart_notice(data_dir: Path, *, cause: str, message: str) -> Path:
    """Write the ``.restart_notice.json`` the next process consumes on startup.

    THE single source of truth for the notice format ``{ts, cause, message}``.
    A process that finds this file on start loads it into
    ``_pending_restart_notice`` and runs ``_auto_continue_after_restart`` —
    i.e. it CONTINUES the pending task instead of going idle. Used by BOTH the
    ``restart_self`` tool (cause="self") and provider failover
    (cause="failover") so failover-restarted agents resume their work
    (the 2026-06-17 incident, a live agent). Returns the notice path.
    """
    from datetime import datetime, timezone
    import json as _json

    data_dir.mkdir(parents=True, exist_ok=True)
    notice_path = data_dir / ".restart_notice.json"
    notice = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cause": cause,
        "message": message,
    }
    notice_path.write_text(_json.dumps(notice))
    from deep_worker_cli.storage_paths import ensure_private_file
    ensure_private_file(notice_path)
    return notice_path


def _is_executable(path: Path) -> bool:
    """True if ``path`` is a file with an execute bit set.

    Used by ``restart_self`` to pre-flight the driver. Written against
    ``Path`` rather than ``os.access`` on purpose: ``tools.py`` must not
    import ``os`` at all, since the only thing it was ever used for in this
    module was the hard-exit that fabricated the false denial, and a test
    pins the import's absence.
    """
    try:
        return path.is_file() and bool(path.stat().st_mode & 0o111)
    except OSError:
        return False


def make_restart_self_tool(sender_name: str) -> BaseTool:
    """Create a tool that allows an agent to restart its own process.

    Unlike ``restart_agent`` (which explicitly forbids self-restart), this
    tool restarts the agent SYNCHRONOUSLY at the moment of the call.

    Concretely, the tool:

      1. Writes a ``.restart_notice.json`` file into the agent's data
         directory containing ``{ts, cause: "self"}``. The newly-started
         process consumes this notice on startup and delivers it as a
         ``SystemMessage`` to the next ``ainvoke`` so the model learns
         it was restarted.
      2. Spawns ``deep-worker --restart <name>`` as a detached subprocess
         IMMEDIATELY (no flag-file, no "after the turn completes"). The
         ``deep-worker --restart`` driver stops the running agent (this
         very process) via SIGTERM, then launches a fresh process with
         the same saved launch_config.

    The tool returns the success/failure dict like any other regular
    tool. Calling it does NOT put the agent into idle — control flow
    in the agent is the same as for any other tool call.
    """

    @tool("restart_self")
    async def restart_self(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> dict[str, Any]:
        """Restart your own agent process synchronously.

        This is a regular tool — it does NOT defer anything. The agent
        is restarted at call time. The new process loads from the same
        on-disk checkpointer (conversation history, pending task) and
        receives a SYSTEM notification of the restart in its next turn.

        Usage policy:
        - Only use when explicitly asked by the user to restart/refresh.
        - Only use when you detect a critical code change that requires
          a restart to take effect.
        - Do NOT use for speculative cleanup, housekeeping, or optimization.
        """
        from deep_worker_cli.storage_paths import agent_root
        from deep_worker_cli.process.self_restart_barrier import (
            arm_self_restart,
            self_restart_delivery_available,
        )
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        # The barrier fires on the checkpointer's commit, which is the same
        # path for every provider — so there is no provider gate here. What
        # IS required is a durable commit to fire on: under --memory-only or
        # in tests the checkpointer is an InMemorySaver with no commit, and a
        # restart could never be confirmed. Fail BEFORE notice/token/arm so
        # that case cannot report success and then silently disarm.
        if not self_restart_delivery_available():
            return {
                "success": False,
                "error": "no_delivery_signal",
                "message": (
                    "restart_self is unavailable: this process has no "
                    "committing checkpointer, so the tool result could "
                    "never be confirmed as part of the model's context."
                ),
            }

        # 1. Write the restart-notice the new process will consume (shared
        #    writer — same format used by provider failover).
        try:
            data_dir = agent_root(sender_name)
            notice_path = write_restart_notice(
                data_dir,
                cause="self",
                message=(
                    f"Agent '{sender_name}' was restarted by its own "
                    f"restart_self tool call."
                ),
            )
        except OSError as exc:
            return {
                "success": False,
                "error": "notice_write_error",
                "message": f"Cannot write restart notice: {exc}",
            }

        # 2. Locate the deep-worker driver.
        cmd = shutil.which("deep-worker")
        if cmd is None:
            return {
                "success": False,
                "error": "command_unavailable",
                "message": (
                    "deep-worker is not on PATH; cannot perform restart. "
                    "The restart-notice file was written but no restart "
                    "subprocess was spawned."
                ),
            }

        # 3. Mint the one-time capability that authorizes stopping this
        #    process mid-turn. Written into THIS agent's private data dir
        #    (0600), so only code already running as the agent can produce
        #    it. ``stop_agent`` consumes it exactly once. Without it a bare
        #    ``--self-requested`` from an external shell is refused, so the
        #    waiver is not a public bypass of the active-turn guard.
        try:
            token = mint_self_restart_token(data_dir)
        except OSError as exc:
            return {
                "success": False,
                "error": "notice_write_error",
                "message": f"Cannot write self-restart token: {exc}",
            }

        # 4. Arm the restart, but DO NOT spawn the driver yet.
        #
        # ⚠️ DELIVERY RACE — why the spawn is deferred.
        #
        # The driver is a separate process. Spawning it here starts a race
        # against our own tool result: the driver can pass ``stop_agent``
        # and land SIGTERM before this result has been recorded in the
        # model's context. ``return`` establishes no happens-before with
        # delivery. If the process dies first, the transcript holds a
        # ``tool_use`` whose result never came and records the SAME
        # fabricated denial this fix exists to remove — just through a
        # narrower window.
        #
        # So the spawn is handed to the delivery barrier, which fires it
        # only after the result has been recorded in the context. See
        # ``arm_self_restart`` and DESIGN_RESTART_SELF_FALSE_DENIAL.md §4.
        # Deferring the spawn also defers any spawn ERROR past the point
        # where this tool can report it — by the time the barrier fires,
        # the result the model sees is already written. Pre-flight the
        # driver here so the ordinary failure (missing/non-executable
        # binary) is still reported synchronously and truthfully, instead
        # of returning "success" and dying quietly in a log line.
        # NB: deliberately NOT ``os.access`` — ``tools.py`` must not import
        # ``os`` at all, because the only thing it was ever used for here
        # was the hard-exit that caused this bug, and a test pins its
        # absence. ``Path`` gives the same answer.
        if not _is_executable(Path(cmd)):
            return {
                "success": False,
                "error": "spawn_failed",
                "message": (
                    f"deep-worker at {cmd} is not executable; "
                    f"cannot perform restart."
                ),
            }

        # Armed BOUND: ``tool_call_id`` is injected by LangChain, so this
        # call knows its own identity without a transport layer supplying
        # one. The barrier fires only when a checkpoint commit carries this
        # exact id, so a concurrent tool's commit cannot stop the process
        # while this result is still unrecorded.
        try:
            arm_self_restart(
                [cmd, "--restart", sender_name,
                 "--self-requested", "--self-requested-token", token],
                tool_call_id,
            )
        except OSError as exc:
            return {
                "success": False,
                "error": "spawn_failed",
                "message": f"Could not arm deep-worker --restart: {exc}",
            }

        # 5. Return normally. The driver is fired by the delivery barrier
        #    once this result has been delivered, and only then stops us.
        #
        # ⚠️ HISTORY — why there is no ``os._exit(0)`` here any more.
        #
        # This used to hard-exit so the old process could not finalise the
        # turn. That severed the in-process MCP server (``create_sdk_mcp_
        # server`` lives in THIS process) before the tool result reached the
        # CLI. The CLI saw a ``tool_use`` whose result never came, recorded
        # ``toolDenialKind: "user-rejected"`` and fed the model a fabricated
        # "The user doesn't want to proceed with this tool use", plus a
        # "[Request interrupted by user for tool use]" line. Both are
        # replayed into the next process on resume. 5 of 5 historical
        # restarts were mis-recorded this way; the user had denied nothing.
        # See DESIGN_RESTART_SELF_FALSE_DENIAL.md.
        #
        # ⚠️ INVARIANT PRESERVED (incident 2026-06-12 — "after this you are
        # in idle, which is WRONG"): the new process must CONTINUE the pending
        # task, not wake up idle. That invariant does NOT depend on killing
        # this process: it is carried by ``.restart_notice.json`` written in
        # step 1. ``AgentProcess.start`` consumes the notice and schedules
        # ``_auto_continue_after_restart`` unconditionally on its presence
        # (agent_process.py §7.1) — it never inspects whether the previous
        # turn was committed. Pinned by
        # ``test_restart_self_turn_continues_not_idle``.
        #
        # The ARMED driver stops this process via SIGTERM (with
        # ``--self-requested`` + its one-time token waiving the "actively
        # running" guard, which would otherwise refuse a restart requested
        # mid-turn). It is spawned only after this result has been written
        # to the CLI — see ``self_restart_barrier`` — so the result cannot
        # lose a race against our own SIGTERM and the transcript records
        # what actually happened.
        return {
            "success": True,
            "message": (
                f"Agent '{sender_name}' is being restarted now. "
                f"The new process will receive a system notice about "
                f"this restart."
            ),
            "restart_notice": str(notice_path),
        }

    return restart_self


# ===========================================================================
# INTER-AGENT MESSAGING TOOL
# ===========================================================================
#
# Tool: send_message_to_agent(agent_name, message)
#
# Lets one running deep-worker agent deliver a user-role message to another
# running deep-worker agent. The sending agent's name is injected by the
# factory (make_send_message_to_agent_tool) from AgentProcess.name, so the
# LLM cannot forge it.
#
# The delivered message always has the shape:
#     <sender> -> <receiver> :> <original text>
# Both the sending agent's name and the receiving agent's name are
# embedded so the message is self-describing in every history view
# (sender's Full Dialog, receiver's Full Dialog, dwcli --history, etc.).
# The sender name is bound by the factory from AgentProcess.name and
# cannot be forged by the LLM.
#
# Full design: docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md
# ===========================================================================

# Importing AgentClient at module import time would create a circular import
# path in some test scenarios. It's imported lazily at call time below, and
# it's also re-exported as a module-level attribute (AgentClient) so that
# tests can monkeypatch "deep_worker_cli.tools.AgentClient" with a fake.
# Note: we MUST NOT set AgentClient=None at module top level — that would
# cause `from deep_worker_cli.tools import AgentClient` to succeed but
# return None. Instead, we populate the attribute the first time it is
# needed, and unit tests can pre-populate it via monkeypatch.
try:  # pragma: no cover - import fallback exercised only in test setups
    from deep_worker_cli.client.agent_client import AgentClient  # noqa: F401
except Exception:  # pragma: no cover
    # Keep the attribute available for monkeypatch even if the eager
    # import fails (e.g. during unrelated unit tests that don't need it).
    # Intentionally rebind the imported name to None so tests / callers
    # can monkeypatch `deep_worker_cli.tools.AgentClient` without worrying
    # about import failures during collection. The shape violation is
    # deliberate — callers check for None before use.
    AgentClient = None  # ty: ignore[invalid-assignment]


class _SendMessageToAgentArgs(BaseModel):
    """Schema for the send_message_to_agent tool.

    CRITICAL: this schema MUST NOT contain a 'sender' field. The sender
    name is bound by the factory from the running agent's own identity
    and is not controllable by the LLM.
    """

    agent_name: str = Field(
        ...,
        description=(
            "Registry name of the target (receiving) agent. The target "
            "agent MUST be currently running (registered in the deep-worker "
            "process registry)."
        ),
    )
    message: str = Field(
        ...,
        description=(
            "The message text to deliver. It will be delivered to the "
            "target agent as a user-role message whose content is "
            "automatically prefixed with '<your-name> -> <target-name> :> ' "
            "(both sender and receiver names are included so every "
            "reader of the conversation can see who said what to whom)."
        ),
    )
    reply_back: bool = Field(
        True,
        description=(
            "Controls whether the receiving agent is expected to reply back "
            "to you (the sender) via send_message_to_agent, or to respond "
            "directly to the user. "
            "When True (the default), the recipient will be instructed to "
            "reply using send_message_to_agent back to you. Use this for "
            "requests, commands, or questions that need a response from the "
            "target agent. "
            "When False, the recipient will be instructed to respond directly "
            "to the user instead of replying to you. Use this when you want "
            "the target agent to produce a result for the user rather than "
            "communicating back to you."
        ),
    )


def make_send_message_to_agent_tool(sender_name: str) -> BaseTool:
    """Create a LangChain tool that lets the current agent send a message
    to another running deep-worker agent.

    The sender name is baked into the tool via this closure and cannot be
    overridden from tool arguments. This prevents the LLM from forging
    the sender field.

    Args:
        sender_name: The registry name of the agent that will use this
            tool. Typically `AgentProcess.name`.

    Returns:
        A LangChain BaseTool named "send_message_to_agent" with the
        signature (agent_name: str, message: str) -> dict[str, Any].
    """

    @tool(
        "send_message_to_agent",
        args_schema=_SendMessageToAgentArgs,
    )
    async def send_message_to_agent(
        agent_name: str,
        message: str,
        reply_back: bool = True,
    ) -> dict[str, Any]:
        """Send a message to another running deep-worker agent.

        Both the sending agent's name and the target agent's name are
        automatically embedded in the delivered content in the form:

            <sender_name> -> <target_name> :> <message>

        The target agent will see this as an incoming user-role message.
        It is delivered fire-and-forget: this call returns once the target
        has accepted the message, NOT once the target finishes replying.

        The ``reply_back`` parameter controls how the receiving agent
        handles the response. True (default): the receiving agent MUST
        reply via send_message_to_agent. False: the receiving agent
        responds directly to the user.

        Usage policy for OUTGOING messages (initiating contact):
        Do NOT use this tool to initiate conversations on your own,
        broadcast to multiple agents, or send unsolicited status updates.
        Only use it when:
        1. The human user explicitly asks you to pass a message or
           command to another agent, OR
        2. You were previously instructed by another agent or by the
           user to do so.

        Usage policy for INCOMING messages (replying to a peer):
        When you receive a message from another agent (content starts
        with ``<sender> -> <your-name> :> ...``), you MUST reply using
        ``send_message_to_agent`` unless ``reply_back=False`` was
        specified. This is NOT optional.

        Args:
            agent_name: Registry name of the target agent. Must refer to a
                currently running agent.
            message: The text to deliver. Must be non-empty.
            reply_back: Whether the receiving agent should reply via
                send_message_to_agent (True, default) or respond directly
                to the user (False).

        Returns:
            A dict describing the outcome. On success:

                {"success": True,
                 "target_agent": "<agent_name>",
                 "delivered_content": "<sender> -> <agent_name> :> <message>",
                 "reply_back": True}

            On failure (NEVER raises — always returns a dict):

                {"success": False,
                 "error": "<code>",
                 "message": "<human-readable explanation>"}

            Possible error codes:
                invalid_argument, self_send_forbidden, agent_not_found,
                connection_failed.
        """
        # --- validation ------------------------------------------------
        if not isinstance(agent_name, str) or not agent_name.strip():
            return {
                "success": False,
                "error": "invalid_argument",
                "message": "agent_name must be a non-empty string",
            }
        if not isinstance(message, str) or not message.strip():
            return {
                "success": False,
                "error": "invalid_argument",
                "message": "message must be a non-empty string",
            }

        target = agent_name.strip()

        # --- policy: reject self-send ---------------------------------
        if target == sender_name:
            return {
                "success": False,
                "error": "self_send_forbidden",
                "message": (
                    f"Agents cannot send messages to themselves "
                    f"(agent_name='{target}' == sender='{sender_name}')."
                ),
            }

        # --- prefixing ------------------------------------------------
        # Exact shape: "<sender> -> <receiver> :> <message>"
        # Both sender and receiver are included so every reader of the
        # history (sender side, receiver side, Full Dialog, dwcli
        # --history) can see at a glance who sent what to whom. The
        # sender name is bound by the factory and cannot be forged by
        # the LLM; the receiver is the "agent_name" argument.
        prefixed = f"{sender_name} -> {target} :> {message}"

        # --- delivery -------------------------------------------------
        # Resolve AgentClient at call time so tests can monkeypatch the
        # module-level attribute with a fake.
        client_cls = globals().get("AgentClient")
        if client_cls is None:
            # Late import fallback (unit tests always pre-populate it).
            try:
                from deep_worker_cli.client.agent_client import AgentClient as _AC
                client_cls = _AC
            except Exception as e:  # pragma: no cover
                return {
                    "success": False,
                    "error": "connection_failed",
                    "message": f"AgentClient unavailable: {e!r}",
                }

        client = client_cls(target)
        try:
            await client.connect()
        except AgentNotFoundError as e:
            return {
                "success": False,
                "error": "agent_not_found",
                "message": (
                    f"Agent '{target}' is not running or not registered: {e}"
                ),
            }
        except IPCConnectionError as e:
            return {
                "success": False,
                "error": "connection_failed",
                "message": f"Could not connect to agent '{target}': {e}",
            }
        except Exception as e:  # pragma: no cover - defensive
            return {
                "success": False,
                "error": "connection_failed",
                "message": f"Unexpected error connecting to '{target}': {e!r}",
            }

        try:
            try:
                # TRUE fire-and-forget: enqueue the prefixed message on the
                # target agent and return immediately. The server schedules
                # the LLM turn in the background via asyncio.create_task so
                # our own turn (which produced this tool call) can end
                # promptly — SendMessageCompletionMiddleware will inject
                # the echo AIMessage and jump to END on the sender side.
                resp = await client._send_command({
                    "command": "enqueue_message",
                    "message": prefixed,
                    "reply_back": reply_back,
                })
                if not resp.get("success"):
                    # A duplicate refusal is a POLICY outcome, not a transport
                    # fault. Passing it through as connection_failed would
                    # invite exactly the wrong reaction — a retry — when the
                    # message has in fact already been received.
                    if resp.get("error") == "duplicate_message":
                        return {
                            "success": False,
                            "error": "duplicate_message",
                            "message": str(
                                resp.get("message")
                                or f"'{target}' already has this message in its "
                                "context; automatic resends are forbidden."
                            ),
                        }
                    return {
                        "success": False,
                        "error": "connection_failed",
                        "message": (
                            f"Delivery to '{target}' rejected: "
                            f"{resp.get('error', 'unknown error')}"
                        ),
                    }
            except AgentNotFoundError as e:
                return {
                    "success": False,
                    "error": "agent_not_found",
                    "message": (
                        f"Agent '{target}' disappeared during delivery: {e}"
                    ),
                }
            except IPCConnectionError as e:
                return {
                    "success": False,
                    "error": "connection_failed",
                    "message": f"Delivery to '{target}' failed: {e}",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": "connection_failed",
                    "message": f"Unexpected delivery error to '{target}': {e}",
                }

            return {
                "success": True,
                "target_agent": target,
                "delivered_content": prefixed,
                "reply_back": reply_back,
            }
        finally:
            # Always disconnect. Swallow errors during teardown so we
            # don't mask a successful delivery with a cleanup failure.
            try:
                await client.disconnect()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "send_message_to_agent: disconnect from '%s' failed: %r",
                    target, e,
                )

    return send_message_to_agent
