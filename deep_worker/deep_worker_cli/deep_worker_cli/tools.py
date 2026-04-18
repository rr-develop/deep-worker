"""Custom tools for the CLI agent."""

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import requests
from langchain_core.tools import BaseTool, tool
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

from deep_worker_cli.config import settings
from deep_worker_cli.exceptions import AgentNotFoundError, IPCConnectionError

logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL: CLAUDE_CODE_PROVIDER MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# Path calculation: __file__ is at deep_worker/deep_worker_cli/deep_worker_cli/tools.py
# .parent x4 = the repository root
_repo_root = Path(__file__).parent.parent.parent.parent
_claude_code_provider_path = _repo_root / "claude_code_provider"
if not _claude_code_provider_path.exists():
    raise ImportError(
        f"CRITICAL: claude_code_provider not found at {_claude_code_provider_path}\n"
        f"This is required for WebSearchDelegationTool.\n"
        f"Expected structure: {_repo_root}/claude_code_provider/"
    )
# Add REPOSITORY ROOT to sys.path, not the claude_code_provider dir itself
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from claude_code_provider.web_search import WebSearchDelegationTool  # noqa: E402

# Initialize Tavily client if API key is available
tavily_client = (
    TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None
)

# ============================================================================
# CRITICAL: WebSearch Delegation Tool - DO NOT MODIFY WITHOUT TESTING!
# ============================================================================
# This tool delegates web searches to Claude Code's built-in WebSearch capability.
# Works with ANY LLM provider (DeepSeek, OpenAI, Gemini, Claude, etc.)
# NO additional API keys required - uses Claude Code SDK.
#
# REGRESSION WARNING:
# - ToolResultBlock is in UserMessage, NOT AssistantMessage!
# - See web_search_delegation.py:_extract_results() for critical implementation
# - Always test with: pytest tests/integration_ai/test_websearch_delegation_deepseek.py
# ============================================================================
web_search_delegation_tool = WebSearchDelegationTool(verbose=False, timeout=120.0)


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
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
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
#     agent:<sender>:> <original text>
# mirroring the existing TaskManager:> convention used for corrective
# prompts (task_manager.py:345).
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
    AgentClient = None  # type: ignore[assignment,misc]


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
            "automatically prefixed with 'agent:<your-name>:> '."
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
    ) -> dict[str, Any]:
        """Send a message to another running deep-worker agent.

        The name of the CURRENT (sending) agent is automatically prepended to
        the message in the form:

            agent:<sender_name>:> <message>

        The target agent will see this as an incoming user-role message.
        It is delivered fire-and-forget: this call returns once the target
        has accepted the message, NOT once the target finishes replying.

        Usage policy:

        - Do NOT use this tool by default. Only use it when:
          1. The human user explicitly asks you to pass a message or
             command to another agent (e.g. "tell beta to ...", "send
             this to agent alpha"), OR
          2. You previously received an `agent:<name>:> ...` message from
             another agent and want to reply to that sender.

        - Do not initiate agent-to-agent conversations on your own, do not
          broadcast, and do not send unsolicited status updates.

        Args:
            agent_name: Registry name of the target agent. Must refer to a
                currently running agent.
            message: The text to deliver. Must be non-empty.

        Returns:
            A dict describing the outcome. On success:

                {"success": True,
                 "target_agent": "<agent_name>",
                 "delivered_content": "agent:<sender>:> <message>"}

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
        # Exact shape: "agent:<sender>:> <message>"
        # Mirrors the existing TaskManager:> convention.
        prefixed = f"agent:{sender_name}:> {message}"

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
                # stream=True matches how dwcli and TaskManager feed
                # messages so the target broadcasts per-chunk to its
                # subscribers normally. We do NOT interpret the return
                # value as "the reply" — see fire-and-forget note in the
                # design doc.
                await client.invoke_agent(prefixed, stream=True)
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
