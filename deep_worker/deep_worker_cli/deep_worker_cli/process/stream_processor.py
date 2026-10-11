"""Stream processor for extracting messages from LangGraph stream.

CRITICAL: ALWAYS TEST LOCAL REPOSITORY CODE
============================================
This module processes streaming messages from LangGraph agent.astream()
and extracts all message types (text chunks, tool calls, tool results).

Tests are in tests/unit_tests/process/test_stream_processing.py
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ============================================================================
# Text Chunk Batching Configuration
# ============================================================================
# To reduce RPyC call overhead, we batch text chunks before sending.
# This prevents character-by-character transmission which creates massive latency.
TEXT_BATCH_DELAY_MS = 50  # Accumulate chunks for 50ms before sending
TEXT_BATCH_MAX_CHARS = 100  # Or send immediately if buffer reaches 100 chars

# Global buffer for batching text chunks per callback
_text_buffers: dict[int, dict[str, Any]] = {}


async def _flush_text_buffer(callback_id: int) -> None:
    """Flush accumulated text chunks for a specific callback.

    This sends batched text through notify_callback, reducing RPyC overhead.
    """
    if callback_id not in _text_buffers:
        return

    buffer = _text_buffers[callback_id]
    if not buffer["text"]:
        return

    # Cancel pending timer if exists
    if buffer["timer"] is not None:
        buffer["timer"].cancel()
        buffer["timer"] = None

    # Send accumulated text
    notify_callback = buffer["callback"]
    accumulated_text = buffer["text"]

    # ✨ CRITICAL: Clear buffer IMMEDIATELY to prevent race conditions!
    # If we clear AFTER await, flush_all_text_buffers() might flush the same text twice
    # This matches deepagents pattern where text is consumed immediately
    buffer["text"] = ""

    await notify_callback({
        "type": "assistant_chunk",
        "role": "assistant",
        "content": accumulated_text,
    })


async def _schedule_flush(callback_id: int, delay_ms: int) -> None:
    """Schedule a buffer flush after delay_ms milliseconds."""
    await asyncio.sleep(delay_ms / 1000.0)
    await _flush_text_buffer(callback_id)


async def _add_text_to_buffer(
    callback_id: int,
    text: str,
    notify_callback: Callable[[dict[str, Any]], Any],  # Now accepts async callbacks
) -> None:
    """Add text to batch buffer, flushing if size/time limits reached.

    This batches character-by-character chunks into larger blocks before
    sending through RPyC, dramatically reducing latency.
    """
    # Initialize buffer if needed
    if callback_id not in _text_buffers:
        _text_buffers[callback_id] = {
            "text": "",
            "callback": notify_callback,
            "timer": None,
        }

    buffer = _text_buffers[callback_id]
    buffer["text"] += text

    # Flush immediately if buffer is large
    if len(buffer["text"]) >= TEXT_BATCH_MAX_CHARS:
        await _flush_text_buffer(callback_id)
        return

    # Schedule flush if timer not already running
    if buffer["timer"] is None:
        timer_task = asyncio.create_task(_schedule_flush(callback_id, TEXT_BATCH_DELAY_MS))
        buffer["timer"] = timer_task


async def flush_all_text_buffers() -> None:
    """Flush all pending text buffers.

    Call this when stream ends to ensure all buffered text is sent.
    """
    callback_ids = list(_text_buffers.keys())
    for callback_id in callback_ids:
        await _flush_text_buffer(callback_id)

    # Clear all buffers
    _text_buffers.clear()


async def process_message_chunk(
    message: AIMessage | ToolMessage | HumanMessage,
    metadata: dict[str, Any],
    notify_callback: Callable[[dict[str, Any]], Awaitable[None]],
    tool_call_buffer: dict[str, Any] | None = None,
) -> None:
    """Process a single message chunk from LangGraph stream.

    Args:
        message: Message from LangGraph stream (AIMessage, ToolMessage, or HumanMessage)
        metadata: Metadata dict that may contain filtering info (e.g., lc_source)
        notify_callback: Async callback to send notifications to client
        tool_call_buffer: Optional buffer for accumulating tool_call_chunk pieces

    Returns:
        None - notifications are sent via callback

    Behavior:
        - HumanMessage: Ignored (already notified by user input)
        - AIMessage with content_blocks:
            - text blocks: Batched and sent as assistant_chunk (with debouncing)
            - tool_call blocks: Send as tool_call
            - tool_call_chunk blocks: Accumulate in buffer, send when complete
        - ToolMessage: Send as tool_result
        - Filters out messages with metadata["lc_source"] == "summarization"

    Text Batching:
        To reduce RPyC overhead, text chunks are batched for up to 50ms or 100 chars
        before sending. This prevents character-by-character RPyC calls.
    """
    # Use callback ID for buffer keying (based on object id)
    callback_id = id(notify_callback)

    # Filter out summarization messages
    if metadata.get("lc_source") == "summarization":
        return

    # Ignore HumanMessage (already notified by user input)
    if isinstance(message, HumanMessage):
        return

    # Process ToolMessage
    if isinstance(message, ToolMessage):
        # Flush any pending text before sending tool_result
        await _flush_text_buffer(callback_id)

        await notify_callback({
            "type": "tool_result",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
            "status": message.status,
            "name": message.name,
        })

        # Detect file operations and extract diffs
        # File operation tools that produce diffs: edit_file, write_file
        tool_name = metadata.get("tool_name") or message.name
        content = message.content if isinstance(message.content, str) else str(message.content)

        if tool_name in {"edit_file", "write_file"} and content:
            # Check if content contains diff markers (--- and +++)
            if "---" in content and "+++" in content:
                # Extract file path from diff if possible
                file_path = "unknown"
                lines = content.split("\n")
                for line in lines[:5]:  # Check first few lines
                    if line.startswith("+++"):
                        # Extract path from "+++ b/path/to/file" or "+++ path/to/file"
                        parts = line.split()
                        if len(parts) >= 2:
                            file_path = parts[1].removeprefix("b/")
                        break

                # Send diff notification
                await notify_callback({
                    "type": "file_diff",
                    "tool_name": tool_name,
                    "tool_call_id": message.tool_call_id,
                    "diff_content": content,
                    "file_path": file_path,
                })

        return

    # Process AIMessage with content_blocks (duck typing for testability)
    if hasattr(message, "content_blocks"):
        # Extract token usage from AIMessage metadata if available
        # This provides real-time token tracking during streaming
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage = message.usage_metadata
            total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            if total_tokens > 0:
                await notify_callback({
                    "type": "token_update",
                    "total_tokens": total_tokens,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                })

        for block in message.content_blocks:
            block_type = block.get("type")

            # Text chunks - BATCH THEM to reduce RPyC overhead
            if block_type == "text":
                await _add_text_to_buffer(
                    callback_id=callback_id,
                    text=block["text"],
                    notify_callback=notify_callback,
                )

            # Complete tool calls
            elif block_type == "tool_call":
                # Flush any pending text before sending tool_call
                await _flush_text_buffer(callback_id)

                await notify_callback({
                    "type": "tool_call",
                    "name": block["name"],
                    "args": block["args"],
                    "id": block["id"],
                })

            # Tool call chunks (streaming) - accumulate and send when complete
            elif block_type == "tool_call_chunk":
                if tool_call_buffer is None:
                    # No buffer provided, skip chunk accumulation
                    continue

                call_id = block.get("id")
                index = block.get("index", 0)

                # CRITICAL FIX: Find existing buffer entry by index if id is None
                # After first chunk, subsequent chunks have id=None, name=None
                # We need to find the buffer entry created by the first chunk
                buffer_key = None

                if call_id is not None:
                    # First chunk with actual ID - use it as key
                    buffer_key = f"{call_id}:{index}"
                else:
                    # Subsequent chunk with id=None - find existing entry by index
                    # Look for any buffer key ending with ":{index}"
                    for existing_key in tool_call_buffer.keys():
                        if existing_key.endswith(f":{index}"):
                            buffer_key = existing_key
                            # Extract the actual call_id from the buffer
                            call_id = tool_call_buffer[existing_key]["id"]
                            break

                # If still no buffer_key found, this is an error - skip chunk
                if buffer_key is None:
                    continue

                # Create buffer entry if doesn't exist (first chunk)
                if buffer_key not in tool_call_buffer:
                    tool_call_buffer[buffer_key] = {
                        "id": call_id,
                        "name": block.get("name"),
                        "args": "",
                    }

                # Update name if present (only from first chunk, don't overwrite with None)
                if block.get("name"):
                    tool_call_buffer[buffer_key]["name"] = block["name"]

                # Accumulate args
                if block.get("args"):
                    tool_call_buffer[buffer_key]["args"] += block["args"]

                # Try to parse accumulated args as JSON
                # If successful, send complete tool call and remove from buffer
                # Skip parsing if args is empty string
                accumulated_args = tool_call_buffer[buffer_key]["args"]
                if not accumulated_args:
                    # No args yet, keep waiting
                    continue

                try:
                    args_json = json.loads(accumulated_args)
                    # Successfully parsed - send complete tool call

                    # Get final id and name from buffer (not from current chunk!)
                    final_id = tool_call_buffer[buffer_key]["id"]
                    final_name = tool_call_buffer[buffer_key]["name"]

                    await notify_callback({
                        "type": "tool_call",
                        "name": final_name,
                        "args": args_json,
                        "id": final_id,
                    })

                    # Remove from buffer
                    del tool_call_buffer[buffer_key]
                except json.JSONDecodeError:
                    # Not yet complete - keep accumulating
                    pass
