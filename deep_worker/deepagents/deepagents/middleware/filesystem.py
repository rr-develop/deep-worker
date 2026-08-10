"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, NotRequired, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # Re-export type here for backwards compatibility
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)
from deepagents.middleware._utils import append_to_system_message

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100

# Template for truncation message in read_file
# {file_path} will be filled in at runtime
READ_FILE_TRUNCATION_MSG = (
    "\n\n[Output was truncated due to size limits. "
    "The file content is very large. "
    "Consider reformatting the file to make it easier to navigate. "
    "For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. "
    "For other formats, you can use appropriate formatting tools to split long lines.]"
)

# Approximate number of characters per token for truncation calculations.
# Using 4 chars per token as a conservative approximation (actual ratio varies by content)
# This errs on the high side to avoid premature eviction of content that might fit
NUM_CHARS_PER_TOKEN = 4


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and rejects
    Windows absolute paths (e.g., C:/..., F:/...) to maintain consistency
    and prevent path format ambiguity.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., C:/...), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    # This maintains consistency in virtual filesystem paths
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in a directory.

This is useful for exploring the filesystem and finding the right file to read or edit.
You should almost ALWAYS use this tool before using the read_file or edit_file tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- ALWAYS prefer editing existing files over creating new ones.
- Only use emojis if the user explicitly requests it."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The write_file tool will create the a new file.
- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
Returns a list of absolute file paths that match the pattern.

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a text pattern across files.

Searches for literal text (not regex) and returns matching files or content based on output_mode.
Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`
- Search for code with special chars: `grep(pattern="def __init__(self):")`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a shell command in an isolated sandbox environment.

Usage:
Executes a given command in the sandbox environment with proper handling and security measures.
Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory
2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command
   - Capture the output of the command
Usage notes:
  - Commands run in an isolated sandbox environment
  - Returns combined stdout/stderr output with exit code
  - If the output is very large, it may be truncated
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


# ═══════════════════════════════════════════════════════════════════════════
# 🛑 PROTECTED INVARIANT — DO NOT BREAK (the 2026-05-28 execute-unavailable incident)
# ═══════════════════════════════════════════════════════════════════════════
# `_supports_execution` decides whether the `execute` shell tool is EXPOSED to
# the model: (a)wrap_model_call FILTERS `execute` out of request.tools when this
# returns False. If the backend is wrapped in a transparent proxy
# (e.g. deep_worker's PrivateSummariesBackend → CompositeBackend → sandbox),
# a bare `isinstance(backend, CompositeBackend)` check returns False and the
# model silently loses `execute` (the 2026-05-28 regression, commit
# 9b8b570). RULES for anyone editing this:
#   1. Always go through `_unwrap_proxy_backend` before the isinstance checks.
#   2. NEVER narrow it back to a bare CompositeBackend check.
#   3. A new backend wrapper MUST expose its inner backend via `_backend`
#      (the convention this unwrap follows) or update _unwrap_proxy_backend.
#   4. Mirror any change in deep_worker_cli `_get_sandbox_backend` (same invariant).
# Symptom of breakage: agent says "I have no execute tool".
# Covered by tests/unit_tests/test_supports_execution_unwrap.py.
# ═══════════════════════════════════════════════════════════════════════════
def _unwrap_proxy_backend(backend: object) -> object:
    """Peel transparent proxy wrappers to reach the wrapped backend.

    Some backends are proxies that delegate to an inner backend held in
    ``_backend`` (e.g. deep_worker's ``PrivateSummariesBackend``). Such a proxy
    subclasses ``BackendProtocol`` — NOT ``CompositeBackend``/sandbox — so the
    ``isinstance`` checks below cannot see through it. We follow the
    ``_backend`` chain so ``proxy(composite(sandbox))`` is still recognised as
    execution-capable; otherwise the ``execute`` tool is wrongly filtered out
    of the model's toolset entirely. Duck-typed (no import of the concrete
    proxy type — keeps deepagents independent). Iterative + visited-set so a
    cyclic wrapper can never spin. See a related internal incident analysis
    (not included here).
    """
    seen: set[int] = set()
    current = backend
    while (
        not isinstance(current, (CompositeBackend, SandboxBackendProtocol))
        and hasattr(current, "_backend")
        and id(current) not in seen
    ):
        seen.add(id(current))
        inner = getattr(current, "_backend", None)
        if inner is None or inner is current:
            break
        current = inner
    return current


def _supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.

    Peels transparent proxy wrappers (``_backend`` chain) first, then:
    for CompositeBackend, checks if the default backend supports execution;
    for other backends, checks if they implement SandboxBackendProtocol.

    Args:
        backend: The backend to check (possibly proxy- and/or composite-wrapped).

    Returns:
        True if the backend supports execution, False otherwise.
    """
    resolved = _unwrap_proxy_backend(backend)

    # For CompositeBackend, check the default backend (also proxy-unwrapped).
    if isinstance(resolved, CompositeBackend):
        return isinstance(_unwrap_proxy_backend(resolved.default), SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(resolved, SandboxBackendProtocol)


# Tools that should be excluded from the large result eviction logic.
#
# This tuple contains tools that should NOT have their results evicted to the filesystem
# when they exceed token limits. Tools are excluded for different reasons:
#
# 1. Tools with built-in truncation (ls, glob, grep):
#    These tools truncate their own output when it becomes too large. When these tools
#    produce truncated output due to many matches, it typically indicates the query
#    needs refinement rather than full result preservation. In such cases, the truncated
#    matches are potentially more like noise and the LLM should be prompted to narrow
#    its search criteria instead.
#
# 2. Tools with problematic truncation behavior (read_file):
#    read_file is tricky to handle as the failure mode here is single long lines
#    (e.g., imagine a jsonl file with very long payloads on each line). If we try to
#    truncate the result of read_file, the agent may then attempt to re-read the
#    truncated file using read_file again, which won't help.
#
# 3. Tools that never exceed limits (edit_file, write_file):
#    These tools return minimal confirmation messages and are never expected to produce
#    output large enough to exceed token limits, so checking them would be unnecessary.
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)


# --- Image-block preservation during large-tool-result eviction ---------------
#
# Tool results may carry image blocks (e.g. from `read_image`). Those blocks are
# semantically irreplaceable: unlike text, the model cannot recover them by
# reading an offloaded file. They are therefore preserved in place rather than
# stringified, subject to the conservative context-cost ceilings below.
#
# `deepagents` is provider-agnostic and deliberately performs no MIME, magic-byte
# or dimension validation -- provider adapters remain authoritative for whether a
# preserved block is acceptable to their API. These caps bound context cost only.

# Per-block cap on the base64/ASCII payload, measured as an exact character count
# before any decode (no decoding, no image library dependency here).
_MAX_PRESERVED_IMAGE_ENCODED_BYTES = 10_000_000

# Per-result cap on how many image blocks may be preserved.
_MAX_PRESERVED_IMAGE_BLOCKS = 8

# Aggregate encoded budget across a whole tool result. Without this, the
# per-block cap alone would permit _MAX_PRESERVED_IMAGE_BLOCKS * 10 MB.
_MAX_PRESERVED_IMAGE_TOTAL_BYTES = 10_000_000

# Block `type` values recognised as images across the shapes we interoperate
# with: LangChain (`image_url`), Anthropic transcript and MCP (both `image`).
_IMAGE_BLOCK_TYPES = ("image_url", "image")

_IMAGE_OMITTED_OVERSIZE = "[Image omitted: encoded payload {actual} exceeds {limit} per-image limit]"
_IMAGE_OMITTED_COUNT = "[Image omitted: more than {limit} image blocks in one tool result]"
_IMAGE_OMITTED_BUDGET = "[Image omitted: aggregate image budget {limit} exhausted]"
_IMAGE_OMITTED_MALFORMED = "[Image omitted: malformed or empty image block]"

# Base64 alphabet used for the ASCII precheck (standard and URL-safe variants).
_BASE64_ALPHABET = (
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-_"
)


def _format_mb(num_bytes: int) -> str:
    """Render a byte count as a short human-readable MB string."""
    return f"{num_bytes / 1_000_000:.1f} MB"


def _base64_payload_length(payload: object) -> int | None:
    """Return the exact ASCII length of a base64 payload, or None if unusable.

    The measurement is an exact character count taken **before** any decode, per
    the design: no decoding, no image library, no MIME sniffing. But a count is
    only meaningful if the payload really is base64 text, so this enforces the
    base64 alphabet and ASCII-ness first. A payload that is not base64 (for
    example the plain-text body of `data:image/png,hello`) or that carries
    non-ASCII characters is rejected, so it can be downgraded rather than
    silently preserved and measured as if it were an image.
    """
    if not isinstance(payload, str) or not payload:
        return None

    try:
        encoded = payload.encode("ascii")
    except UnicodeEncodeError:
        return None

    # Base64 is 4-character groups, optionally padded with one or two '='.
    stripped = encoded.rstrip(b"=")
    padding = len(encoded) - len(stripped)
    if padding > 2 or len(encoded) % 4 != 0:
        return None
    if not stripped or stripped.translate(None, _BASE64_ALPHABET) != b"":
        return None

    return len(encoded)


def _image_payload_length(block: dict) -> int | None:
    """Return the encoded payload length of an image block, or None.

    None means "not a preservable inline image": a malformed block, a block
    whose payload is not valid base64, or a block referencing a remote URL.
    Remote URLs are never fetched -- doing so would add egress and a
    time-of-replay contract -- so they stay evictable.
    """
    block_type = block.get("type")

    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            url = image_url
        elif isinstance(image_url, dict):
            raw_url = image_url.get("url")
            url = raw_url if isinstance(raw_url, str) else None
        else:
            url = None
        if not url or not url.startswith("data:"):
            return None
        header, separator, payload = url.partition(",")
        # Only a base64 data URI carries an inline payload we can measure.
        # `data:image/png,hello` is a valid data URI but is not base64.
        if not separator or not header.rstrip().endswith(";base64"):
            return None
        return _base64_payload_length(payload)

    if block_type == "image":
        # MCP shape: {"type": "image", "data": ..., "mimeType": ...}
        length = _base64_payload_length(block.get("data"))
        if length is not None:
            return length
        # Anthropic transcript shape: {"type": "image", "source": {"data": ...}}
        source = block.get("source")
        if isinstance(source, dict):
            return _base64_payload_length(source.get("data"))
        return None

    return None


def _is_remote_url_image(block: dict) -> bool:
    """True if the block references an image by URL rather than inlining it.

    Such a block is left exactly as it is: `deepagents` is provider-agnostic and
    must not decide on a provider's behalf that a URL is unusable.
    """
    image_url = block.get("image_url")
    url = image_url if isinstance(image_url, str) else None
    if isinstance(image_url, dict):
        raw_url = image_url.get("url")
        url = raw_url if isinstance(raw_url, str) else None
    if url is None:
        # The Anthropic/MCP `image` shape may also carry a bare `url`.
        raw_url = block.get("url")
        url = raw_url if isinstance(raw_url, str) else None
    return bool(url) and not url.startswith("data:")


def _classify_content_blocks(
    blocks: list,
) -> tuple[list[tuple[int, object]], list[tuple[int, object]], bool]:
    """Partition tool-result blocks into preserved images and evictable content.

    Blocks are considered in their original order. An image block is preserved
    only if it is a well-formed inline payload that passes the per-block cap, the
    block-count cap and the remaining aggregate budget. The first block that
    would exceed the budget -- and every image block after it -- is downgraded to
    a short diagnostic text block carrying no payload bytes, which then follows
    the normal evictable path. Downgrading is deterministic and order-preserving.

    Args:
        blocks: The original list content of a ToolMessage.

    Returns:
        A tuple of (preserved, evictable, transformed):
        - preserved: (index, block) pairs for image blocks kept verbatim.
        - evictable: (index, block) pairs for everything else, where a rejected
          image appears as its replacement diagnostic text block.
        - transformed: True if any block was downgraded, i.e. the classification
          was not a no-op. Callers must materialize the result when this is True,
          even if no text offload is needed, otherwise a rejected payload would
          be resurrected.
    """
    preserved: list[tuple[int, object]] = []
    evictable: list[tuple[int, object]] = []
    transformed = False
    used_budget = 0
    budget_exhausted = False

    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") not in _IMAGE_BLOCK_TYPES:
            evictable.append((index, block))
            continue

        payload_len = _image_payload_length(block)

        if payload_len is None:
            if _is_remote_url_image(block):
                # A remote URL carries no inline payload for us to preserve, but
                # deepagents is provider-agnostic and must not decide the block
                # is unusable: a downstream provider may well support URLs. Leave
                # the original block untouched in the evictable partition and
                # perform no fetch.
                evictable.append((index, block))
                continue
            # Malformed inline block: downgraded to a payload-free diagnostic so
            # it cannot masquerade as a usable image downstream.
            evictable.append((index, {"type": "text", "text": _IMAGE_OMITTED_MALFORMED}))
            transformed = True
            continue

        if payload_len > _MAX_PRESERVED_IMAGE_ENCODED_BYTES:
            notice = _IMAGE_OMITTED_OVERSIZE.format(
                actual=_format_mb(payload_len),
                limit=_format_mb(_MAX_PRESERVED_IMAGE_ENCODED_BYTES),
            )
            evictable.append((index, {"type": "text", "text": notice}))
            transformed = True
            continue

        if len(preserved) >= _MAX_PRESERVED_IMAGE_BLOCKS:
            notice = _IMAGE_OMITTED_COUNT.format(limit=_MAX_PRESERVED_IMAGE_BLOCKS)
            evictable.append((index, {"type": "text", "text": notice}))
            transformed = True
            continue

        if budget_exhausted or used_budget + payload_len > _MAX_PRESERVED_IMAGE_TOTAL_BYTES:
            # Latched, per the signed contract: the first block that would
            # exceed the aggregate budget AND every image block after it are
            # downgraded. Without the latch a later, smaller image would slip
            # through, making the preserved prefix depend on block sizes rather
            # than on position -- non-deterministic and untestable.
            budget_exhausted = True
            notice = _IMAGE_OMITTED_BUDGET.format(limit=_format_mb(_MAX_PRESERVED_IMAGE_TOTAL_BYTES))
            evictable.append((index, {"type": "text", "text": notice}))
            transformed = True
            continue

        used_budget += payload_len
        preserved.append((index, block))

    return preserved, evictable, transformed


def _evictable_content_str(content: object, evictable: list[tuple[int, object]]) -> str:
    """Compute the string form of the evictable portion of a tool result.

    Pinned representation, so that offloaded files do not silently change shape:
    - `str` content is written verbatim;
    - exactly one evictable block of type `text` yields its `text` value, which
      preserves the existing readability special case;
    - otherwise the stringified list of evictable blocks, i.e. the same form as
      before, just computed over the evictable subset.
    """
    if isinstance(content, str):
        return content

    blocks = [block for _, block in evictable]

    if len(blocks) == 1 and isinstance(blocks[0], dict):
        # `blocks` holds opaque content blocks, so the narrowed element is only
        # known to be *some* mapping; cast to spell out the key/value types.
        only_block = cast(dict[str, Any], blocks[0])
        if only_block.get("type") == "text" and "text" in only_block:
            return str(only_block["text"])

    return str(blocks)


def _replace_content(message: ToolMessage, content: str | list) -> ToolMessage:
    """Return a copy of `message` carrying new content and nothing else changed.

    Rebuilding a ToolMessage from a handful of named fields silently drops
    everything else the original carried -- `id`, `status`, `artifact`,
    `response_metadata` and any additional keys. Copying the model instance and
    overriding only `content` preserves all of them.
    """
    updated = message.model_copy(deep=False)
    updated.content = content
    return updated


def _classified_blocks(
    preserved: list[tuple[int, object]],
    evictable: list[tuple[int, object]],
) -> list:
    """All blocks in original order, with classification decisions applied."""
    return [block for _, block in sorted(preserved + evictable, key=lambda item: item[0])]


def _rebuild_with_preserved_images(
    preserved: list[tuple[int, object]],
    evictable: list[tuple[int, object]],
    replacement_text: str,
) -> list:
    """Rebuild message content after a text offload, keeping images in order.

    Walking the original block positions: the first evictable position becomes
    the single offload notice, later evictable positions are dropped (their
    content now lives in the offloaded file), and preserved image blocks are
    emitted at their original relative positions.

    The contract is relative image *order*, not label association: inline labels
    are lost, and the notice names the file where they can be read back.
    """
    notice_index = evictable[0][0] if evictable else -1

    items: list[tuple[int, object]] = list(preserved)
    if notice_index >= 0:
        items.append((notice_index, {"type": "text", "text": replacement_text}))

    items.sort(key=lambda item: item[0])
    rebuilt = [block for _, block in items]

    if notice_index < 0:
        rebuilt.insert(0, {"type": "text", "text": replacement_text})

    return rebuilt


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> str:
    """Create a preview of content showing head and tail with truncation marker.

    Args:
        content_str: The full content string to preview.
        head_lines: Number of lines to show from the start.
        tail_lines: Number of lines to show from the end.

    Returns:
        Formatted preview string with line numbers.
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    # Show head and tail with truncation marker
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: `ls`, `read_file`, `write_file`,
    `edit_file`, `glob`, and `grep`.

    Files can be stored using any backend that implements the `BackendProtocol`.

    If the backend implements `SandboxBackendProtocol`, an `execute` tool is also added
    for running shell commands.

    This middleware also automatically evicts large tool results to the file system when
    they exceed a token threshold, preventing context window saturation.

    Args:
        backend: Backend for file storage and optional execution.

            If not provided, defaults to `StateBackend` (ephemeral storage in agent state).

            For persistent storage or hybrid setups, use `CompositeBackend` with custom routes.

            For execution support, use a backend that implements `SandboxBackendProtocol`.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        tool_token_limit_before_evict: Token limit before evicting a tool result to the
            filesystem.

            When exceeded, writes the result using the configured backend and replaces it
            with a truncated preview and file reference.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # Ephemeral storage only (default, no execution)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With hybrid storage (ephemeral + persistent /memories/)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # With sandbox backend (supports execution)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Backend for file storage and optional execution, or a factory callable.
                Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
        """
        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _get_backend(self, runtime: Any) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context (can be ToolRuntime or Runtime).

        Returns:
            Resolved backend instance.
        """
        # `self.backend` is `BackendProtocol | BackendFactory`. ty can't
        # narrow on `callable(...)`, so use an explicit isinstance check —
        # factories are plain Callables, not BackendProtocol instances.
        if isinstance(self.backend, BackendProtocol):
            return self.backend
        return self.backend(runtime)

    def _create_ls_tool(self) -> BaseTool:
        """Create the ls (list files) tool."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Synchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            infos = resolved_backend.ls_info(validated_path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Asynchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            infos = await resolved_backend.als_info(validated_path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
        )

    def _create_read_file_tool(self) -> BaseTool:
        """Create the read_file tool."""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> str:
            """Synchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            result = resolved_backend.read(validated_path, offset=offset, limit=limit)

            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # Check if result exceeds token threshold and truncate if necessary
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                # Calculate truncation message length to ensure final result stays under threshold
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> str:
            """Asynchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)

            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # Check if result exceeds token threshold and truncate if necessary
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                # Calculate truncation message length to ensure final result stays under threshold
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Synchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """Asynchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
        )

    def _create_edit_file_tool(self) -> BaseTool:
        """Create the edit_file tool."""
        tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

        def sync_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """Synchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """Asynchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
        )

    def _create_glob_tool(self) -> BaseTool:
        """Create the glob tool."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Synchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            infos = resolved_backend.glob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Asynchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            infos = await resolved_backend.aglob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
        )

    def _create_grep_tool(self) -> BaseTool:
        """Create the grep tool."""
        tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

        def sync_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """Synchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """Asynchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
        )

    def _create_execute_tool(self) -> BaseTool:
        """Create the execute tool for sandbox command execution."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Synchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                sandbox_backend = cast(SandboxBackendProtocol, resolved_backend)
                result = sandbox_backend.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Asynchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                sandbox_backend = cast(SandboxBackendProtocol, resolved_backend)
                result = await sandbox_backend.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, files_update):
            - processed_message: New ToolMessage with truncated content and file reference
            - files_update: Dict of file updates to apply to state, or None if eviction failed

        Note:
            Content blocks are partitioned into preserved images and evictable
            content. Only the evictable portion is measured against the eviction
            threshold and, if too large, written to
            /large_tool_results/{tool_call_id} and replaced by a truncated
            preview plus file reference.

            Image blocks are preserved in place, because unlike text the model
            cannot recover them by reading the offloaded file. They are subject
            to the per-block, count and aggregate caps defined above; a rejected
            image is downgraded to a payload-free diagnostic text block, never
            silently dropped.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        content = message.content
        if isinstance(content, list):
            preserved, evictable, transformed = _classify_content_blocks(content)
        else:
            preserved, evictable, transformed = [], [], False

        # Only the evictable portion counts toward the eviction threshold, so an
        # image never drags its own metadata line over the limit.
        content_str = _evictable_content_str(content, evictable)
        needs_offload = len(content_str) > NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict

        if not needs_offload:
            # A cap decision is never conditional on the offload decision: if
            # classification rejected a payload we must materialize that
            # rejection, otherwise the rejected bytes would be resurrected.
            if not transformed:
                return message, None
            return _replace_content(message, _classified_blocks(preserved, evictable)), None

        # Write the evictable content to the filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            # The offload failed, so evictable content must stay in the message.
            # But returning the original would resurrect payloads the caps
            # already rejected, so materialize the classification instead.
            if not transformed:
                return message, None
            return _replace_content(message, _classified_blocks(preserved, evictable)), None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        if preserved:
            processed_content: str | list = _rebuild_with_preserved_images(preserved, evictable, replacement_text)
        else:
            # No images to carry: return a plain string, exactly as before.
            processed_content = replacement_text

        return _replace_content(message, processed_content), result.files_update

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.
        See _process_large_message for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        content = message.content
        if isinstance(content, list):
            preserved, evictable, transformed = _classify_content_blocks(content)
        else:
            preserved, evictable, transformed = [], [], False

        content_str = _evictable_content_str(content, evictable)
        needs_offload = len(content_str) > NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict

        if not needs_offload:
            if not transformed:
                return message, None
            return _replace_content(message, _classified_blocks(preserved, evictable)), None

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            # See the sync twin: never resurrect a payload the caps rejected.
            if not transformed:
                return message, None
            return _replace_content(message, _classified_blocks(preserved, evictable)), None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        if preserved:
            processed_content: str | list = _rebuild_with_preserved_images(preserved, evictable, replacement_text)
        else:
            processed_content = replacement_text

        return _replace_content(message, processed_content), result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a Command with evicted
            content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.
        See _intercept_large_tool_result for full documentation.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return await handler(request)

        tool_result = await handler(request)
        return await self._aintercept_large_tool_result(tool_result, request.runtime)
