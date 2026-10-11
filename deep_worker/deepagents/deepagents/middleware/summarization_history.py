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

SCHEMA_VERSION = 5
_COMPAT_SCHEMA_VERSIONS = frozenset({4, 5})
KIND = "deepagents.summarization_history.message"
SEGMENT_KIND = "deepagents.summarization_history.segment"
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


def _usable_origin(origin: Any) -> dict[str, Any] | None:
    if not isinstance(origin, dict):
        return None
    provider = str(origin.get("provider") or "").strip().lower()
    if not provider or provider == "unknown":
        return None
    out = dict(origin)
    out["provider"] = provider
    return out


def _effective_origin_from_messages(messages: list[AnyMessage]) -> dict[str, Any] | None:
    """Nearest origin stamp on or before the first message of this batch."""
    for message in messages:
        meta = getattr(message, "response_metadata", None) or {}
        found = _usable_origin(meta.get("deep_worker_origin"))
        if found is not None:
            return found
        extra = getattr(message, "additional_kwargs", None) or {}
        found = _usable_origin(extra.get("deep_worker_origin"))
        if found is not None:
            return found
    return None


def serialize_summarized_messages_jsonl(
    messages: list[AnyMessage],
    *,
    summarized_at: str | None = None,
    thread_id: str | None = None,
    origin: dict[str, Any] | None = None,
    reason: str | None = None,
    provider_session_id: str | None = None,
) -> str:
    """Serialize a list of messages into JSONL format (one JSON object per line).

    Schema v5 prepends a segment header (``kind=…segment``) carrying the
    effective provider/model origin so it survives offload. Message lines
    follow. ``deserialize_summarized_messages_jsonl`` still returns only
    messages and accepts both v4 (no header) and v5.

    Args:
        messages: The messages to serialize.
        summarized_at: Optional ISO-8601 timestamp for the summarization event.
            If ``None``, the current UTC time is used.
        thread_id: Optional LangGraph thread id for the segment header.
        origin: Effective origin of the first message; resolved from
            ``messages`` when omitted.
        reason: ``hot`` / ``cold`` / ``requested``.
        provider_session_id: CLI/API session id of the summarizer call.

    Returns:
        A string where each line is a JSON object.
    """
    event_timestamp = summarized_at or _utc_now()
    resolved_origin = origin if origin is not None else _effective_origin_from_messages(messages)
    header = {
        "schema_version": SCHEMA_VERSION,
        "kind": SEGMENT_KIND,
        "summarized_at": event_timestamp,
        "thread_id": thread_id,
        "origin": resolved_origin,
        "reason": reason,
        "provider_session_id": provider_session_id,
    }
    lines: list[str] = [
        json.dumps(header, cls=_SummarizationJSONEncoder, ensure_ascii=False, separators=(",", ":"))
    ]
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
    return "\n".join(lines) + "\n"


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
        version = entry.get("schema_version")
        if version not in _COMPAT_SCHEMA_VERSIONS:
            msg = f"Unsupported summarization history schema at line {line_number}: {version!r}"
            raise ValueError(msg)
        kind = entry.get("kind")
        if kind == SEGMENT_KIND:
            continue
        if kind != KIND:
            msg = f"Unsupported summarization history kind at line {line_number}: {kind!r}"
            raise ValueError(msg)
        raw = entry.get("message")
        if not isinstance(raw, dict):
            msg = f"Invalid summarization history entry at line {line_number}: missing message payload"
            raise ValueError(msg)
        message_dicts.append(_decode_message(raw))
    return cast("list[AnyMessage]", messages_from_dict(message_dicts))
