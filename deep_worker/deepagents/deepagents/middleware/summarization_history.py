"""Serialize/deserialize summarized messages as plain JSONL.

Each line is one JSON object with the message payload stored as a plain
``messages_to_dict`` dictionary serialized with ``json.dumps`` (no msgpack/base64).

Backward compatibility: ``deserialize_summarized_messages_jsonl`` also handles
the deprecated msgpack+base64 v4 format (where ``message.encoding == "langgraph_jsonplus"``).
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.messages import AnyMessage, messages_from_dict, messages_to_dict
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4
KIND = "deepagents.summarization_history.message"
_DEPRECATED_ENCODING = "langgraph_jsonplus"
_DEPRECATED_SERDE = JsonPlusSerializer()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class _SummarizationJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime, Decimal, and other non-JSON types
    by converting them to strings via ``default=str``."""

    def default(self, o: Any) -> str:
        return str(o)


def _encode_message(message: AnyMessage) -> dict[str, Any]:
    return messages_to_dict([message])[0]


def _decode_message(entry_message: dict[str, Any]) -> dict[str, Any]:
    """Decode a message payload, handling both current plain-JSON and deprecated msgpack formats."""
    if entry_message.get("encoding") == _DEPRECATED_ENCODING:
        # Deprecated v4 format: msgpack+base64 wrapper
        serde_type = entry_message.get("type")
        data = entry_message.get("data")
        if isinstance(serde_type, str) and isinstance(data, str):
            payload = base64.b64decode(data.encode("ascii"))
            decoded = _DEPRECATED_SERDE.loads_typed((serde_type, payload))
            if isinstance(decoded, dict):
                return decoded
        logger.warning("Failed to decode deprecated msgpack message, falling through")
    # Current format: the message is already a plain messages_to_dict dict
    return entry_message


def serialize_summarized_messages_jsonl(
    messages: list[AnyMessage],
    *,
    summarized_at: str | None = None,
) -> str:
    """Serialize a list of messages into JSONL format (one JSON object per line).

    Args:
        messages: The messages to serialize.
        summarized_at: Optional ISO-8601 timestamp for the summarization event.
            If ``None``, the current UTC time is used.

    Returns:
        A string where each line is a JSON object representing one message.
    """
    event_timestamp = summarized_at or _utc_now()
    lines: list[str] = []
    for index, message in enumerate(messages):
        entry = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "summarized_at": event_timestamp,
            "message_index": index,
            "message_type": getattr(message, "type", None),
            "message_id": getattr(message, "id", None),
            "message": _encode_message(message),
        }
        lines.append(json.dumps(entry, cls=_SummarizationJSONEncoder, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def deserialize_summarized_messages_jsonl(content: str) -> list[AnyMessage]:
    """Deserialize JSONL content back into LangChain messages.

    Args:
        content: The JSONL string to deserialize.

    Returns:
        A list of LangChain ``AnyMessage`` objects in file order.

    Raises:
        ValueError: If ``schema_version`` or ``kind`` are unexpected.
    """
    message_dicts: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("schema_version") != SCHEMA_VERSION:
            msg = f"Unsupported summarization history schema at line {line_number}: {entry.get('schema_version')!r}"
            raise ValueError(msg)
        if entry.get("kind") != KIND:
            msg = f"Unsupported summarization history kind at line {line_number}: {entry.get('kind')!r}"
            raise ValueError(msg)
        raw = entry.get("message")
        if not isinstance(raw, dict):
            msg = f"Invalid summarization history entry at line {line_number}: missing message payload"
            raise ValueError(msg)
        message_dicts.append(_decode_message(raw))
    return cast("list[AnyMessage]", messages_from_dict(message_dicts))
