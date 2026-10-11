# Design: `read_image` Tool for Image Analysis

**Status: ✅ IMPLEMENTED**

## Problem

Agents using the `claude-code-sdk` provider cannot analyze images stored on disk.
Claude models natively support vision (image analysis), but there is no tool
in deep_worker that reads an image file and passes it to the model as visual
content.

## Architecture Constraints

1. Tool calls **must** go through LangGraph (ToolNode) — no direct API calls
2. **No blocking** — the tool must be a regular sync function (LangGraph
   runs tools in a thread pool via `asyncio.to_thread` automatically)
3. **No polling or separate threads** — event-driven only
4. **No changes to ClaudeCodeChatModel architecture**

## Solution

### How it works

The tool is a plain Python function (like `http_request`, `fetch_url`) that:
1. Reads image file from disk
2. Base64-encodes it
3. Returns the image as an MCP `image` content block

The key insight is that the MCP protocol supports image content blocks:
```json
{"type": "image", "data": "<base64_data>", "mimeType": "image/png"}
```

When `ClaudeCodeChatModel` resolves a tool result via `proxy_mcp.resolve_call()`,
it converts the `ToolMessage.content` into `MCPResult.content` blocks. Currently
(line 512 of `chat_model.py`) it wraps everything as text:

```python
mcp_result: MCPResult = {
    "content": [{"type": "text", "text": str(msg.content)}],
}
```

This must be updated to detect and preserve image content blocks.

### Data Flow

```
Agent requests: read_image(file_path="/tmp/x.png")
    │
    ▼
LangGraph ToolNode executes read_image()
    │ (runs in thread pool — no blocking of event loop)
    │
    ▼
Tool returns ToolMessage with multimodal content:
    content = [
        {"type": "text", "text": "Image: /tmp/x.png (320x640 PNG)"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]
    │
    ▼
ClaudeCodeChatModel.ainvoke() receives ToolMessage
    │
    ▼
resolve_call() converts to MCPResult with image block:
    MCPResult = {
        "content": [
            {"type": "text", "text": "Image: /tmp/x.png (320x640 PNG)"},
            {"type": "image", "data": "iVBOR...", "mimeType": "image/png"}
        ]
    }
    │
    ▼
Claude Code SDK sends tool result with image to Claude API
    │
    ▼
Claude model sees the image and can analyze it
```

### Changes Required

#### 1. New tool: `read_image` in `tools.py`

```python
def read_image(file_path: str) -> list[dict[str, Any]]:
    """Read an image file and return it for visual analysis.

    Supports PNG, JPEG, GIF, WebP formats.
    Maximum file size: 20MB (Anthropic API limit).

    Args:
        file_path: Absolute path to the image file.

    Returns:
        List of content blocks (text description + image data).
    """
```

The function:
- Validates file exists and is a supported image format
- Checks file size (≤20MB)
- Reads and base64-encodes the image
- Detects MIME type from file extension
- Returns a list with text metadata + image_url block (LangChain format)

This is a **sync function** — LangGraph's ToolNode runs it in a thread pool
automatically via `asyncio.to_thread`, so no blocking of the event loop.

#### 2. Update `chat_model.py` resolve_call (line 512)

Current code wraps all ToolMessage content as text:
```python
mcp_result: MCPResult = {
    "content": [{"type": "text", "text": str(msg.content)}],
}
```

Updated code must detect multimodal content (list of blocks) and convert
LangChain `image_url` blocks to MCP `image` blocks:

```python
if isinstance(msg.content, list):
    # Multimodal content — preserve structure
    mcp_content = []
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            # Convert LangChain image_url to MCP image block
            url = block["image_url"]["url"]
            # Extract base64 data from data URL
            # "data:image/png;base64,iVBOR..." → ("image/png", "iVBOR...")
            if url.startswith("data:"):
                parts = url.split(",", 1)
                mime = parts[0].split(";")[0].split(":")[1]
                data = parts[1]
                mcp_content.append({"type": "image", "data": data, "mimeType": mime})
        elif isinstance(block, dict) and block.get("type") == "text":
            mcp_content.append(block)
        else:
            mcp_content.append({"type": "text", "text": str(block)})
    mcp_result: MCPResult = {"content": mcp_content}
else:
    mcp_result: MCPResult = {
        "content": [{"type": "text", "text": str(msg.content)}],
    }
```

#### 3. Register tool in `agent_process.py`

Add `read_image` to the `web_tools` list (line ~282):

```python
from deep_worker_cli.tools import http_request, fetch_url, web_search_delegation_tool, read_image
web_tools = [http_request, fetch_url, web_search_delegation_tool, read_image]
```

### No Other Changes Required

- **LangGraph**: No changes — ToolNode already handles sync tools
- **deepagents/graph.py**: No changes — tools are passed through
- **ProcessRegistry, SessionManager, agent_command.py**: No changes
- **Frontend**: No changes — image analysis results are text responses

### Supported Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| PNG | image/png | Most common for screenshots |
| JPEG | image/jpeg | Photos |
| GIF | image/gif | Animated not supported by Claude |
| WebP | image/webp | Modern format |

### Limits

| Constraint | Value | Source |
|-----------|-------|--------|
| Max file size | 20 MB | Anthropic API limit |
| Max image dimension | 8192 px | Anthropic API limit |
| Max images per request | 20 | Anthropic API limit |

### Error Handling

| Error | Response |
|-------|----------|
| File not found | `{"error": "File not found: /path/to/file"}` |
| Unsupported format | `{"error": "Unsupported image format: .bmp. Supported: png, jpg, jpeg, gif, webp"}` |
| File too large | `{"error": "Image too large: 25.3 MB. Maximum: 20 MB"}` |
| Read error | `{"error": "Failed to read image: [details]"}` |

### Test Plan

1. **Unit tests for `read_image` tool:**
   - Read valid PNG → returns list with text + image_url blocks
   - Read valid JPEG → correct MIME type
   - File not found → error dict
   - Unsupported format → error dict
   - File too large → error dict

2. **Unit tests for `resolve_call` multimodal handling:**
   - ToolMessage with string content → unchanged behavior (text block)
   - ToolMessage with list content including image_url → MCP image block
   - ToolMessage with mixed text + image blocks → all preserved

3. **Guard test:**
   - Verify read_image is registered in web_tools list
