"""Custom tools for the CLI agent."""

import sys
from pathlib import Path
from typing import Any, Literal

import requests
from markdownify import markdownify
from tavily import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)
from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError

from deep_worker_cli.config import settings

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
