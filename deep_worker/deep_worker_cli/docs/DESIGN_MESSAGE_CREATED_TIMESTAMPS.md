# Design: Per-Message Created Timestamps

## Status

Approved for implementation.

## User Requirement

Every agent history message must carry its creation timestamp in the checkpoint database:

- user messages
- assistant messages
- assistant tool-call messages
- tool-result messages
- synthetic middleware messages
- TaskManager messages

Timestamps must be UTC and monotonically increasing as messages are appended. When summarization offloads old history to markdown, every individual message in that markdown must also show its timestamp.

Existing agents must keep working. Old checkpoints without message timestamps should remain readable.

## Current Behavior

The LangGraph checkpoint stores message objects in the SQLite checkpoint payload. `deep_worker_cli.process.message_dict_cache.serialize_message()` serializes those messages for history clients, but it does not expose any per-message timestamp.

`deepagents.middleware.summarization.SummarizationMiddleware` currently writes offloaded markdown via `get_buffer_string(filtered_messages)`, so the markdown has a section timestamp (`## Summarized at ...`) but no per-message timestamps.

`deep_worker_cli.sessions.get_thread_history()` currently adds the checkpoint timestamp to messages while reading offline history. That is not precise enough: all messages in one checkpoint can get the same timestamp, and the timestamp is not stored on each message.

## Storage Contract

Use a message-level timestamp in `additional_kwargs`:

```python
message.additional_kwargs["created_at"] = "2026-05-28T12:34:56.123456+00:00"
```

Reasons:

- It is serialized with the LangChain message object into the checkpoint payload.
- It survives restart and offline history reads.
- It does not require schema changes to the LangGraph checkpoint tables.
- It matches the user's suggested direction.

For `AIMessage.tool_calls`, store per-call timestamps in `AIMessage.additional_kwargs["tool_call_created_at"]` as a `call_id -> timestamp` map. LangChain's `AIMessage(tool_calls=...)` constructor rejects unknown keys inside tool-call dicts, so the map is the DB storage contract. The serializer projects this map back out as `tool_calls[*]["created_at"]` for clients. The value is the parent `AIMessage` timestamp for every tool call in that message.

## Timestamp Semantics

Add a small per-process monotonic UTC clock:

- Source: `datetime.now(UTC)`.
- Format: ISO 8601 with `+00:00` timezone and microsecond precision.
- Monotonic rule: if the wall clock returns a value less than or equal to the last emitted timestamp, emit `last + 1 microsecond`.
- Scope: per agent process. No polling, no background task, no thread.

This gives stable ordering inside one agent/TaskManager writer, which is the write model this project already uses. Main agent and TaskManager histories use separate writers/clocks, so their individual histories are monotonic, but no total ordering across both processes is guaranteed.

## Stamping Points

Add `deep_worker_cli.message_timestamps` with helpers:

- `CREATED_AT_KEY = "created_at"`
- `MonotonicUtcClock.next() -> str`
- `ensure_message_created_at(message, clock) -> AnyMessage`
- `ensure_messages_created_at(messages, clock) -> list[AnyMessage]`
- `format_message_with_timestamp(message) -> str`

Helpers preserve a valid existing `created_at`. Missing timestamps get assigned. If an existing timestamp would break monotonic order inside the processed list, assign a new monotonic timestamp.

Integrate into the agent runtime:

1. User inputs passed to `agent.ainvoke()` / `agent.astream()` become explicit `HumanMessage` objects with `additional_kwargs={"created_at": ...}` instead of plain `{"role": "user", "content": ...}` dicts.
2. Add `MessageTimestampMiddleware` to the CLI middleware stack immediately after `DanglingToolCallMiddleware` and before `SendMessageCompletionMiddleware`. It stamps any missing timestamps in `before_model` and wraps model/tool calls:
   - `wrap_model_call` / `awrap_model_call`: stamp the returned `AIMessage` before it is appended/checkpointed.
   - `wrap_tool_call` / `awrap_tool_call`: stamp returned `ToolMessage` or every message inside a returned `Command(update={"messages": ...})`.
3. Stamp synthetic messages created by deep-worker middleware at creation time where they bypass model/tool wrappers:
   - dangling tool-call closers
   - send-message completion echo
   - summarization synthetic HumanMessage

TaskManager uses the same approach: its prompt passed to `manager_agent.ainvoke()` is an explicit timestamped `HumanMessage`, and its manager agent uses the same middleware stack if created through `create_cli_agent`.

## History Serialization

Update `MessageDictCache.serialize_message()` to include:

```python
"created_at": msg.additional_kwargs.get("created_at")
```

when present.

For tool calls inside an `AIMessage`, include each call's `created_at` if present.

Keep backward compatibility: old messages without `created_at` serialize exactly as before except without the new field.

## Summarization Markdown

Replace the raw `get_buffer_string(filtered_messages)` body with timestamp-aware formatting:

```markdown
## Summarized at 2026-05-28T12:40:00.000000+00:00

[2026-05-28T12:34:56.123456+00:00] Human: ...
[2026-05-28T12:34:57.123456+00:00] AI: ...
[2026-05-28T12:34:57.123456+00:00] Tool call grep (call_1): {"pattern":"x"}
[2026-05-28T12:34:58.123456+00:00] Tool result grep (call_1): ...
```

If summarization encounters an old message without `created_at`, it assigns one immediately before formatting by walking the offloaded list through the same monotonic clock. Those fallback values are display timestamps for markdown; new checkpoints get real creation timestamps before persistence. This avoids timestamp-less markdown while keeping old history readable.

## Tests

TDD unit tests first:

1. Timestamp clock emits UTC ISO timestamps and strictly increases even when the wall clock does not.
2. Message stamping preserves valid existing timestamps and stamps missing Human/AI/Tool messages.
3. AI tool calls get an `additional_kwargs["tool_call_created_at"]` map when the parent AI message is stamped.
4. `MessageDictCache` exposes `created_at` for messages and projects tool-call timestamps into serialized tool calls.
5. Summarization markdown includes per-message timestamps for Human, AI, tool-call, and ToolMessage entries.

Integration test without mocks:

- Use a temporary HOME and the real SQLite checkpointer path.
- Build a real LangGraph/deepagents agent with a deterministic local fake chat model and a real tool invocation, not mocked storage.
- Invoke it through the compiled agent with a real checkpointer.
- Reopen the SQLite DB via the real checkpointer and verify the persisted messages contain monotonic `additional_kwargs.created_at` values.
- Trigger summarization and verify the real markdown file contains per-message timestamp lines.

This avoids live provider calls while still testing the actual checkpoint DB and backend file writes.

## Non-Goals

- No migration of old checkpoints. Old messages remain readable and get no stored timestamp until they are re-written by a future summarization/import/migration path.
- No polling.
- No new threads.
- No SQLite schema change.
- No attempt to establish total ordering across different agent processes. Each agent/TaskManager writer is monotonic within its own thread history.
