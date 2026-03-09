"""Stream processor for extracting messages from LangGraph stream.

CRITICAL: ALWAYS TEST LOCAL REPOSITORY CODE
============================================
This module processes streaming messages from LangGraph agent.astream()
and extracts all message types (text chunks, tool calls, tool results).

Tests are in tests/unit_tests/process/test_stream_processing.py
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


async def process_message_chunk(
    message: AIMessage | ToolMessage | HumanMessage,
    metadata: dict[str, Any],
    notify_callback: Callable[[dict[str, Any]], None],
    tool_call_buffer: Optional[dict[str, Any]] = None,
) -> None:
    """Process a single message chunk from LangGraph stream.

    Args:
        message: Message from LangGraph stream (AIMessage, ToolMessage, or HumanMessage)
        metadata: Metadata dict that may contain filtering info (e.g., lc_source)
        notify_callback: Callback to send notifications to client
        tool_call_buffer: Optional buffer for accumulating tool_call_chunk pieces

    Returns:
        None - notifications are sent via callback

    Behavior:
        - HumanMessage: Ignored (already notified by user input)
        - AIMessage with content_blocks:
            - text blocks: Send as assistant_chunk
            - tool_call blocks: Send as tool_call
            - tool_call_chunk blocks: Accumulate in buffer, send when complete
        - ToolMessage: Send as tool_result
        - Filters out messages with metadata["lc_source"] == "summarization"
    """
    # Filter out summarization messages
    if metadata.get("lc_source") == "summarization":
        return

    # Ignore HumanMessage (already notified by user input)
    if isinstance(message, HumanMessage):
        return

    # Process ToolMessage
    if isinstance(message, ToolMessage):
        notify_callback({
            "type": "tool_result",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
            "status": message.status,
            "name": message.name,
        })
        return

    # Process AIMessage with content_blocks (duck typing for testability)
    if hasattr(message, "content_blocks"):
        for block in message.content_blocks:
            block_type = block.get("type")

            # Text chunks
            if block_type == "text":
                notify_callback({
                    "type": "assistant_chunk",
                    "role": "assistant",
                    "content": block["text"],
                })

            # Complete tool calls
            elif block_type == "tool_call":
                notify_callback({
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

                # Create buffer entry if doesn't exist
                buffer_key = f"{call_id}:{index}"
                if buffer_key not in tool_call_buffer:
                    tool_call_buffer[buffer_key] = {
                        "id": call_id,
                        "name": block.get("name"),
                        "args": "",
                    }

                # Accumulate name if present
                if block.get("name"):
                    tool_call_buffer[buffer_key]["name"] = block["name"]

                # Accumulate args
                if block.get("args"):
                    tool_call_buffer[buffer_key]["args"] += block["args"]

                # Try to parse accumulated args as JSON
                # If successful, send complete tool call and remove from buffer
                try:
                    args_json = json.loads(tool_call_buffer[buffer_key]["args"])
                    # Successfully parsed - send complete tool call
                    notify_callback({
                        "type": "tool_call",
                        "name": tool_call_buffer[buffer_key]["name"],
                        "args": args_json,
                        "id": call_id,
                    })
                    # Remove from buffer
                    del tool_call_buffer[buffer_key]
                except json.JSONDecodeError:
                    # Not yet complete - keep accumulating
                    pass
