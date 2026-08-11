# Design: Claude CLI Session Reset on LangGraph Summarization

**Status: ✅ IMPLEMENTED**

## Problem

When using `claude-code` provider, two separate conversation histories exist:

| Layer | History | Managed by | Trimming |
|-------|---------|------------|----------|
| **LangGraph** | `state["messages"]` in checkpointer | `SummarizationMiddleware` | ✅ Trimmed + summarized |
| **Claude CLI** | Internal session per `session_id` | Claude Code CLI process | ❌ Only grows |

`SummarizationMiddleware` periodically replaces old messages with a summary
(e.g. 500 messages → 1 summary + 20 preserved). But the Claude CLI session
retains ALL messages ever sent via `client.query()`. After enough tool calls
(observed: 1255 in practice), the Claude CLI session exceeds the model's
context window (1M tokens for opus) and Claude API returns
`"Prompt is too long"` as a plain text response (not an exception).

### Cascade failure

When summarization triggers, `_create_summary()` calls `model.invoke()` which
goes through the same overflowing Claude CLI session → Claude returns
`"Prompt is too long"` as the "summary text" → this gets injected as the
conversation summary → agent receives `"Here is a summary: Prompt is too long"`
→ subsequent queries also fail with the same error.

### Existing detection (but no fix)

`chat_model.py:525` already detects the summarization event:
```python
if sent_count > len(messages):
    # History was reset, start from beginning
    # TODO: Ideally should also reset Claude session
    sent_count = 0
```

This resets `sent_count` but does NOT reset the Claude CLI session — so the
agent re-sends the summary into the same overflowing session.

## Solution: Session Epoch Rotation

### Key insight

`ClaudeSDKClient.query(prompt, session_id)` accepts an arbitrary `session_id`
string. Different `session_id` values create **independent conversation forks**
within the same Claude CLI process. The client, MCP proxy, tools, and
`system_prompt` are all **shared** — only the conversation history is per-session.

This means we can start a fresh conversation by simply changing the
`session_id`, without recreating the client or proxy.

### Architecture

```
ClientManager
├── clients: { "my_agent": (ClaudeSDKClient, ProxyMCPServer) }  ← UNCHANGED
├── _sent_count: { "my_agent": 42 }  ← reset to 0 on rotation
└── _session_epoch: { "my_agent": 3 }  ← NEW: incremented on rotation

Effective session_id sent to Claude CLI:
  epoch 0:  "my_agent"           (backward compatible)
  epoch 1:  "my_agent__epoch_1"
  epoch 2:  "my_agent__epoch_2"
  ...
```

### Changes

#### 1. `client_manager.py` — Track session epoch

```python
class ClientManager:
    def __init__(self, ...):
        ...
        self._session_epoch: dict[str, int] = {}

    def get_effective_session_id(self, thread_id: str) -> str:
        """Get the current session_id for Claude CLI queries.
        
        Returns thread_id for epoch 0 (backward compat),
        or "{thread_id}__epoch_{N}" for N > 0.
        """
        epoch = self._session_epoch.get(thread_id, 0)
        if epoch == 0:
            return thread_id
        return f"{thread_id}__epoch_{epoch}"

    def rotate_session(self, thread_id: str) -> str:
        """Rotate to a new Claude CLI session for this thread.
        
        Called when LangGraph summarization resets the message history.
        Increments epoch and resets sent_count.
        
        Returns the new effective session_id.
        """
        current = self._session_epoch.get(thread_id, 0)
        self._session_epoch[thread_id] = current + 1
        self._sent_count[thread_id] = 0
        new_id = self.get_effective_session_id(thread_id)
        return new_id
```

No new clients, no new proxies, no reconnection needed.

#### 2. `chat_model.py` — Rotate on summarization detection

Replace the current TODO block:

```python
# BEFORE (current):
if sent_count > len(messages):
    sent_count = 0

# AFTER (fix):
if sent_count > len(messages):
    # LangGraph summarization reset the message history.
    # Claude CLI session still has the full old history — we MUST rotate
    # to a new session_id to start fresh. Otherwise Claude CLI will
    # eventually hit "Prompt is too long" on the accumulated history.
    new_session_id = self._client_manager.rotate_session(thread_id)
    logger.info(
        f"Session rotated: {thread_id} → {new_session_id} "
        f"(old sent_count={sent_count}, new messages={len(messages)})"
    )
    sent_count = 0
```

And update the `query()` call to use effective session_id:

```python
# BEFORE:
await client.query(content, session_id=thread_id)

# AFTER:
effective_session_id = self._client_manager.get_effective_session_id(thread_id)
await client.query(content, session_id=effective_session_id)
```

#### 3. `cleanup_client()` — Also clean epoch

```python
async def cleanup_client(self, thread_id: str) -> None:
    ...
    self._session_epoch.pop(thread_id, None)
```

### What stays the same

- `ClaudeSDKClient` instance — reused (same process, same connection)
- `ProxyMCPServer` — reused (same tools, same Future-based proxy)
- `system_prompt` — shared across all sessions in the same client
- `SummarizationMiddleware` — unchanged (it already works correctly)
- `_build_mcp_content()` — unchanged
- All other middleware — unchanged

### Sequence diagram: Normal flow vs. After rotation

**Normal (no rotation):**
```
LangGraph messages: [Sys, H1, A1, H2, A2, ...]  (500 messages)
sent_count: 500
Claude CLI session "my_agent": 500 turns accumulated
→ Eventually: "Prompt is too long"
```

**After rotation:**
```
SummarizationMiddleware triggers:
  LangGraph messages: [Sys, Summary, Preserved1, Preserved2, ...]  (22 messages)
  
ClaudeCodeChatModel.ainvoke():
  sent_count(500) > len(messages)(22) → ROTATION!
  session_epoch: 0 → 1
  sent_count: 0
  effective_session_id: "my_agent__epoch_1"
  
  Send Summary + Preserved messages to Claude CLI session "my_agent__epoch_1"
  → Claude CLI starts fresh conversation with just the summary
  → Full context available again
```

### Edge cases

1. **Multiple rotations**: Epoch increments each time. "my_agent__epoch_1" → 
   "my_agent__epoch_2" → etc. Old sessions are abandoned (Claude CLI GC).

2. **Summary generation itself overflows**: If summarization middleware calls
   `model.invoke()` for summary generation, this goes through the SAME session
   that's already overflowing. **This is NOT fixed by rotation** because rotation
   happens in `ainvoke()` AFTER `before_model()` already ran.
   
   However, `SummarizationMiddleware.trim_tokens_to_summarize` is set to `None`
   in our config (graph.py:163), which means it doesn't trim messages for
   summary generation. The summary prompt is relatively small (~600 tokens)
   but the messages being summarized could be large.
   
   **Mitigation**: The summarization middleware uses the **same model** as the
   main agent. For claude-code, this `model.invoke()` goes through
   `ClaudeCodeChatModel.ainvoke()` which uses the same client but with
   `session_id=thread_id` (not the rotated one). Since rotation hasn't happened
   yet at this point, the summary generation query goes to the old session.
   
   If the old session is already at the context limit, the summary itself will
   be `"Prompt is too long"`. This is a **pre-existing issue** — the rotation
   fix prevents the agent from getting STUCK in this state by giving it a fresh
   session after the (possibly failed) summarization.

3. **TaskManager session**: TaskManager uses `thread_id="my_agent:tm"` — 
   completely separate session, independent rotation.

4. **Backward compatibility**: Epoch 0 produces the original thread_id as
   session_id (no suffix). Existing sessions work unchanged.

### Testing plan

1. **Unit test**: `test_rotate_session_increments_epoch` — verify epoch tracking
2. **Unit test**: `test_effective_session_id_format` — verify naming
3. **Unit test**: `test_rotation_resets_sent_count` — verify sent_count reset
4. **Unit test**: `test_ainvoke_detects_summarization_and_rotates` — mock
   scenario where len(messages) < sent_count → rotation triggered
5. **Unit test**: `test_query_uses_effective_session_id` — verify
   `client.query()` receives the rotated session_id
6. **Guard test**: `test_cleanup_removes_epoch` — verify cleanup
