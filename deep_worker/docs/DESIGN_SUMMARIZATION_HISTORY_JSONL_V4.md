# Design: Summarization History JSONL v4

Status: Draft
Date: 2026-05-29

## Problem

`SummarizationMiddleware` currently offloads messages removed from the active thread into a markdown file. The file is append-only, but the body is a human-readable transcript. That is useful for reading, but it is not a lossless storage format: role, message ids, `additional_kwargs`, response metadata, tool call metadata, timestamps, and provider-specific fields cannot be reconstructed exactly from the markdown text.

The history storage needs a new format version that preserves every summarized message as structured data and can be appended on repeated summarization events.

## Goals

1. Store summarized history in `*.jsonl`, not markdown.
2. Keep one JSON object per original message per line.
3. Preserve enough data to reconstruct the exact LangChain message thread, including role/type, content, ids, timestamps in `additional_kwargs`, tool calls, tool result ids, response metadata, and other serialized fields.
4. Append new summarized messages to the same file on repeated summarizations.
5. Keep the implementation synchronous/async only where the middleware already is; no polling, background threads, or extra workers.
6. Keep the summary message path usable by agents and humans.

## Non-goals

1. Do not synthesize Claude CLI-only fields like `uuid`, `parentUuid`, `sessionId`, `cwd`, or queue-operation entries. Those are needed for Claude CLI resume files, not for restoring LangGraph message objects.
2. Do not change checkpoint storage.
3. Do not migrate existing `.md` history files in this feature.
4. Do not make summarized history directly resumable by Claude CLI. Conversion can be built later from this lossless LangChain JSONL format.

## Format

History path changes from:

```text
{history_path_prefix}/{thread_id}.md
```

to:

```text
{history_path_prefix}/{thread_id}.jsonl
```

Each line is one JSON object:

```json
{
  "schema_version": 4,
  "kind": "deepagents.summarization_history.message",
  "summarized_at": "2026-05-29T10:15:30.123456+00:00",
  "message_index": 17,
  "message_type": "ai",
  "message_id": "run-...",
  "message": {
    "type": "ai",
    "data": {
      "content": "I have completed the task.",
      "additional_kwargs": {
        "created_at": "2026-05-29T10:00:00.000001+00:00"
      },
      "response_metadata": {},
      "id": "run-...",
      "tool_calls": []
    }
  }
}
```

The `message` payload is the canonical LangChain dictionary produced by `langchain_core.messages.messages_to_dict([message])[0]`, serialized directly with `json.dumps(..., default=str, ensure_ascii=False)`. Non-JSON-native types (`datetime`, `Decimal`) are converted to ISO strings via `default=str`. This keeps every line as valid, human-readable JSON without msgpack or base64 wrappers. A reader uses `json.loads` and `messages_from_dict` to restore messages.

Top-level fields are deliberately small:

- `schema_version`: integer format version, `4`.
- `kind`: stable discriminator for future mixed JSONL readers.
- `summarized_at`: UTC timestamp for the summarization event that evicted this message.
- `message_index`: zero-based index of the message within the filtered offload batch. This is metadata only; the JSONL line order is still authoritative for thread order.
- `message_type` / `message_id`: JSON-readable breadcrumbs derived from the canonical message for inspection and debugging. Restore does not depend on them.
- `message`: canonical LangChain `messages_to_dict` dictionary serialized as plain JSON (not msgpack/base64).

Because each line corresponds to one original message, there are no standalone section/header/meta lines.

## Restore Semantics

A reader reconstructs the historical thread by reading non-empty lines in file order, validating `schema_version == 4`, decoding each `entry["message"]` via `json.loads()` (the message is already a plain JSON dict), collecting the LangChain message dictionaries, and calling `messages_from_dict` on that list.

Repeated summarizations append more lines. File order remains the chronological order of eviction. Previous synthetic summarization messages are still filtered out before offload, as today, so the JSONL file contains original conversation messages rather than recursive summary pointers.

## Implementation Plan

1. Add a small helper module near the middleware, e.g. `deepagents/middleware/summarization_history.py`.
2. Implement `serialize_summarized_messages_jsonl(messages, summarized_at=None) -> str` using `messages_to_dict` and `json.dumps(..., default=str, ensure_ascii=False)` so the JSONL record remains human-readable plain JSON without msgpack or base64 wrappers.
3. Implement `deserialize_summarized_messages_jsonl(content: str) -> list[AnyMessage]` for tests and future callers.
4. Change `SummarizationMiddleware._get_history_path()` to return `.jsonl`.
5. Change sync and async `_offload_to_backend` to append serialized JSONL lines instead of markdown sections. Keep the current read-existing-then-edit/write backend contract to avoid requiring a new append API.
6. Update docstrings and tests that currently expect `.md`/markdown.
7. Add unit tests for lossless serialization of Human/AI/Tool messages with timestamps, tool calls, ids, names, response metadata, and `additional_kwargs`.
8. Add/adjust integration test so a real summarization writes `.jsonl`, the file has one JSON object per stored message, and `deserialize_summarized_messages_jsonl` restores messages with timestamps and tool-call metadata.

## Compatibility

Existing `.md` files are left untouched. New summarization events write to `.jsonl`. Summary messages generated after this change reference the `.jsonl` path. Older summary messages that reference `.md` files remain valid if those files already exist.

The storage format follows the same append-only JSONL pattern as the Claude Code provider's session files, but stores the canonical LangChain message dict directly as plain JSON (no msgpack/base64 encoding) for maximum human readability.

## Risks

1. `messages_to_dict` compatibility depends on LangChain's serialization contract. This is preferable to a custom partial serializer because it is the framework's supported message representation.
2. Plain `json.dumps` with `default=str` converts `datetime` to ISO string and `Decimal` to string representation, losing the original Python type. This is an acceptable tradeoff for human readability — the content, IDs, tool calls, and all string metadata are preserved exactly.
3. Backend writes still read the whole existing file before editing, as the current markdown implementation does. This feature does not introduce new blocking behavior, but a future backend append API would be more efficient.
4. Humans lose the easy markdown transcript. Agents can still read JSONL directly, and a later CLI/viewer can render JSONL as markdown without losing the original structured data.
