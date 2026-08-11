# Design: thread isolation for subagents in the claude-code provider

## Problem

`ClaudeCodeChatModel` is stateful: it stores `sent_count` per `thread_id` in the global
`ClientManager` and mirrors the message history into a Claude SDK session (a `claude` subprocess).

When a subagent (the task tool from SubAgentMiddleware) calls `subagent.invoke(state)` without
a config, `ClaudeCodeChatModel.ainvoke()` receives `thread_id='default'` — the same one as the
main agent. The subagent overwrites the main agent's `sent_count` → on the next call the main
agent re-sends old messages → the Claude SDK ends up in an inconsistent state → hang.

For stateless models (ChatOpenAI, ChatAnthropic) there is no problem — each `ainvoke()` is an
independent HTTP request, there is no `sent_count` and there are no sessions.

## Principles

1. **No fallbacks.** `thread_id` is mandatory in every `ainvoke()` call.
   Missing = exception. Never use `'default'`.
2. Subagents receive a unique `thread_id` in the format `<agent_name>:sub_<suffix>`.
3. Each `thread_id` → a separate Claude SDK session with its own `sent_count`.
4. Subagent sessions are ephemeral; clean them up after completion.

## thread_id format

| Context           | thread_id format              | Example                  |
|--------------------|-------------------------------|--------------------------|
| Main agent        | `<agent_name>`                | `test_op`                |
| TaskManager       | `<agent_name>:tm`             | `test_op:tm`             |
| Subagent (task)   | `<agent_name>:sub_<uuid8>`    | `test_op:sub_a1b2c3d4`  |

The `uuid4().hex[:8]` suffix guarantees uniqueness for each subagent invocation.

## Audit: who calls ainvoke/astream

A full audit of all 10 LangGraph graph invocation sites in the project:

### Already pass thread_id (8 of 10) — no changes required

| # | File | Line | Context | thread_id |
|---|------|--------|----------|-----------|
| 1 | `agent_process.py` | ~1040 | agent.astream (TaskManager mode) | `self.name` |
| 2 | `agent_process.py` | ~1197 | agent.ainvoke (TaskManager iteration) | `self.name` |
| 3 | `agent_process.py` | ~1293 | agent.ainvoke (non-TaskManager) | `self.name` |
| 4 | `agent_process.py` | ~1438 | agent.ainvoke (HITL resume) | `self.name` |
| 5 | `textual_adapter.py` | ~323 | agent.astream (TUI) | from a parameter |
| 6 | `non_interactive.py` | ~392 | agent.astream (CLI) | `generate_thread_id()` |
| 7 | `task_manager.py` | ~368 | manager_agent.ainvoke | `f"{self.agent_name}:tm"` |
| 8 | `manager_thread.py` | ~202 | manager_agent.ainvoke | `f"{self.agent_name}:tm"` |

### Do NOT pass thread_id (2 of 10) — REQUIRE FIXING

| # | File | Line | Context | Problem |
|---|------|--------|----------|----------|
| 9 | `subagents.py` | 442 | `subagent.invoke(state)` | No config |
| 10 | `subagents.py` | 460 | `subagent.ainvoke(state)` | No config |

### Tests — will require updating

The tests in `claude_code_provider/tests/` — most of them call `model.ainvoke()` without a config.
All of them must pass `config={"configurable": {"thread_id": "test_..."}}`.

Affected files:
- `test_chat_model.py` — ~10 ainvoke calls without config
- `test_event_based_tool_detection.py` — ~6 calls
- `test_integration.py` — ~6 calls
- `test_chat_model_sent_count.py` — ~5 calls
- `test_multiple_sequential_queries.py` — ~2 calls
- `test_mcp_tool_direct_execution.py` — tests with real middleware tools

The `test_ainvoke_default_thread_id` test must be **replaced** with
`test_ainvoke_raises_without_thread_id` — verifying that a ValueError is raised.

## Detailed change plan

### Change 1: `ClaudeCodeChatModel.ainvoke()` — mandatory thread_id

**File:** `claude_code_provider/chat_model.py`, method `ainvoke()`, lines ~489-492

**Current code:**
```python
# 2. Extract thread_id from config
thread_id = "default"
if config and "configurable" in config:
    thread_id = config["configurable"].get("thread_id", "default")
```

**New code:**
```python
from langchain_core.runnables.config import ensure_config

# Pick up the config from the LangGraph context variable (if set)
config = ensure_config(config)

# ⚠️ MANDATORY: thread_id MUST be explicitly provided in every call.
# ClaudeCodeChatModel is STATEFUL — each thread_id maps to a separate
# Claude SDK session with its own sent_count and message history.
#
# DO NOT add a fallback to "default" or any other hardcoded value!
# A shared thread_id between main agent and subagent causes:
#   1. sent_count corruption (subagent overwrites main agent's count)
#   2. Message re-sending (main agent resends already-seen messages)
#   3. Claude SDK deadlock (inconsistent session state → hang)
#
# Every caller MUST explicitly provide its own thread_id.
# If thread_id is missing, the caller has a configuration bug — fail loudly.
if "configurable" not in config or "thread_id" not in config["configurable"]:
    raise ValueError(
        "ClaudeCodeChatModel requires explicit thread_id in config. "
        "Pass config={'configurable': {'thread_id': '...'}}. "
        "Each caller (main agent, subagent, TaskManager) must have its own thread_id. "
        "Implicit defaults are forbidden to prevent session state corruption."
    )
thread_id = config["configurable"]["thread_id"]
```

**Rationale:** `ensure_config(config)` reads the `var_child_runnable_config` context variable.
When LangGraph calls `model.ainvoke(messages)` from inside a graph, it sets that
context variable with the parent graph's config, including `thread_id`. That is why for the main
agent the `thread_id` is picked up from the context automatically. For a subagent, the config must
be passed explicitly when calling `subagent.invoke(state, config=...)`.

### Change 2: `subagents.py` — passing thread_id to subagents

**File:** the agent middleware module `subagents.py` (external dependency, not included here)

**Current code (line 442):**
```python
result = subagent.invoke(subagent_state)
```

**New code:**
```python
import uuid

# Generate a unique thread_id for the subagent.
# Format: <parent_agent>:sub_<uuid> — this provides:
# 1. Uniqueness — each subagent invocation gets its own Claude SDK session
# 2. Traceability — the prefix shows which agent spawned it
parent_thread_id = runtime.config.get("configurable", {}).get("thread_id", "agent")
agent_name = parent_thread_id.split(":")[0]
sub_thread_id = f"{agent_name}:sub_{uuid.uuid4().hex[:8]}"

result = subagent.invoke(
    subagent_state,
    config={"configurable": {"thread_id": sub_thread_id}},
)
```

**Similarly for `atask()` (line 460):**
```python
result = await subagent.ainvoke(
    subagent_state,
    config={"configurable": {"thread_id": sub_thread_id}},
)
```

**Note:** `runtime.config` contains the parent graph's config because
`ToolRuntime` is injected by LangGraph's ToolNode with the config of the current execution.
For the MCP path (claude-code provider), `runtime.config` is built in
`_make_runtime_executor` (see change 3).

### Change 3: `_make_runtime_executor` — pass thread_id into runtime.config

**File:** `claude_code_provider/chat_model.py`, method `_make_runtime_executor()`

**Current code:**
```python
args["runtime"] = ToolRuntime(
    state={},
    context=None,
    config={},  # ← empty config
    ...
)
```

**Problem:** When the `task` tool is called via MCP (rather than via the LangGraph ToolNode),
`runtime.config` is empty. The subagent does not know the parent's `thread_id` and cannot build
its own `sub_thread_id`.

**Solution:** Pass the current agent's `thread_id` into `runtime.config`. For this,
`direct_handler` in ProxyMCPServer needs to know the current `thread_id`.

**Implementation option:**

Add `agent_name` to `ProxyMCPServer`:

```python
# proxy_mcp_server.py
class ProxyMCPServer:
    def __init__(self, ..., agent_name: str | None = None):
        self._agent_name = agent_name
```

Pass it from `ClientManager.get_or_create_client()`:

```python
# client_manager.py
proxy_mcp = ProxyMCPServer(
    tools_metadata,
    tool_executors=tool_executors,
    working_dir=working_dir,
    agent_name=thread_id,  # the main agent's thread_id = agent_name
)
```

Pass `agent_name` into `_make_runtime_executor`:

```python
@staticmethod
def _make_runtime_executor(tool_instance: BaseTool, agent_name: str) -> Any:
    async def executor(input_dict: dict[str, Any]) -> Any:
        args = dict(input_dict)
        args["runtime"] = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": agent_name}},
            ...
        )
        return await tool_instance.ainvoke(args)
    return executor
```

But `_make_runtime_executor` is called from `bind_tools()`, where `agent_name` is not yet
known (bind_tools is called before `get_or_create_client`). Therefore `agent_name` must be
passed later.

**Better option:** make `agent_name` a field of `ProxyMCPServer` and build
`runtime.config` in `direct_handler` rather than in the executor. That is, the executor receives
`agent_name` as a parameter at call time:

```python
# In bind_tools:
tool_executors[t.name] = ("runtime", t)  # marker that a runtime is needed

# In ProxyMCPServer.direct_handler:
if executor_type == "runtime":
    # Create a ToolRuntime with agent_name from self._agent_name
    ...
```

This complicates the interface. **Simpler:** make `_make_runtime_executor` NOT a staticmethod
but a closure that captures `agent_name` later. Or pass `agent_name`
through `ProxyMCPServer` into `direct_handler`:

```python
# proxy_mcp_server.py — direct_handler for runtime tools:
async def direct_handler(args):
    call_args = dict(args)
    if self._working_dir:
        call_args.setdefault("working_dir", self._working_dir)
    # For the runtime executor: pass agent_name via a special key
    call_args["__agent_name__"] = self._agent_name
    result = await executor(call_args)
    return ...

# chat_model.py — _make_runtime_executor:
async def executor(input_dict):
    agent_name = input_dict.pop("__agent_name__", "unknown")
    args = dict(input_dict)
    args["runtime"] = ToolRuntime(
        config={"configurable": {"thread_id": agent_name}},
        ...
    )
    return await tool_instance.ainvoke(args)
```

**Final option (the cleanest):** pass `agent_name` via a closure:

```python
# In bind_tools — do NOT create the executor right away (agent_name is unknown)
# Instead, mark the tool as needs_runtime

# In ProxyMCPServer._create_tool_handler — create the executor with agent_name:

if tool_name in self._runtime_tools:
    tool_instance = self._runtime_tools[tool_name]
    async def direct_handler(args):
        call_args = dict(args)
        if self._working_dir:
            call_args.setdefault("working_dir", self._working_dir)
        call_args["runtime"] = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": self._agent_name}},
            stream_writer=lambda x: None,
            tool_call_id=f"mcp_{tool_name}_{id(args)}",
            store=None,
        )
        return await tool_instance.ainvoke(call_args)
    return direct_handler
```

This requires passing `runtime_tools` and `tool_call_id_tools` into ProxyMCPServer instead of
wrapped executors. A redesign of the `tool_executors` interface.

**Recommended compromise:** keep the current architecture with executors, but add
`agent_name` to `ProxyMCPServer` and pass it into `direct_handler` via a closure.
The executor for runtime tools will be lazy — created in `ProxyMCPServer._create_tool_handler`
with access to `self._agent_name`:

```python
# chat_model.py — bind_tools():
# For runtime tools: pass the BaseTool instance, not an executor
if needs_runtime:
    tool_executors[t.name] = t  # BaseTool instance, not a callable
elif needs_tool_call_id:
    tool_executors[t.name] = self._make_tool_call_executor(t)
else:
    tool_executors[t.name] = t.ainvoke  # callable

# proxy_mcp_server.py — _create_tool_handler():
executor = self._tool_executors[tool_name]
if isinstance(executor, BaseTool):
    # Runtime tool — create the executor with an inline ToolRuntime
    tool_instance = executor
    async def direct_handler(args):
        call_args = dict(args)
        call_args["runtime"] = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": self._agent_name or "unknown"}},
            stream_writer=lambda x: None,
            tool_call_id=f"mcp_{tool_name}_{id(args)}",
            store=None,
        )
        if self._working_dir:
            call_args.setdefault("working_dir", self._working_dir)
        result = await tool_instance.ainvoke(call_args)
        text = str(result) if not isinstance(result, str) else result
        return {"content": [{"type": "text", "text": text}]}
    return direct_handler
else:
    # Regular executor (callable) — call it as is
    ...
```

### Change 4: TaskManager — use the same provider as the main agent

**File:** the CLI's `task_manager/task_manager.py` module (external dependency, not included here)

**Problem:** TaskManager calls `create_model()` without arguments (line 197).
Without `provider` and `model_name` → auto-detection based on API keys → always DeepSeek (first
in priority order). The main agent may run on claude-code while TaskManager runs on DeepSeek.
This is illogical and prevents TaskManager from using the claude-code provider's capabilities.

**Current code (line 197):**
```python
model = create_model()
```

**New code:**
```python
model = create_model(
    provider=self.config.provider,
    model_name=self.config.model,
)
```

`self.config` is an `AgentConfig` that contains the main agent's `provider` and `model`.
TaskManager already receives it at initialization (line 57: `config: 'AgentConfig'`).

**thread_id:** TaskManager already passes its own thread_id `f"{self.agent_name}:tm"` in
the `manager_agent.ainvoke()` calls (lines 368 and 202 in manager_thread.py). With a mandatory
thread_id (change 1) this will work correctly — TaskManager will get its own
isolated Claude SDK session.

**Also check `manager_thread.py`** — the same problem applies if TaskManager creates the
manager agent through a different path.

**working_dir for TaskManager:** If TaskManager uses the claude-code provider, it
needs to set `_working_dir` on the model (the same way as for the main agent in
`agent_process.py:269`):
```python
if hasattr(model, '_working_dir'):
    model._working_dir = str(self.config.working_dir)
```

### Change 5: Cleanup of ephemeral sessions

**File:** `claude_code_provider/client_manager.py`

Each subagent invocation creates a new Claude SDK session (a `claude` subprocess). Without cleanup
the processes will accumulate.

**Options:**

A) Cleanup immediately after the subagent finishes — in the task tool's executor:
```python
result = await tool_instance.ainvoke(args)
# Cleanup ephemeral subagent session
await client_manager.cleanup_client(sub_thread_id)
return result
```

B) TTL-based cleanup — ClientManager automatically purges sessions older than N minutes.

C) Cleanup when the main agent finishes — purge all sessions, including subagent ones,
in `agent_process.py` at shutdown.

**Recommendation:** option A (immediate cleanup) — the simplest and most reliable.
The subagent is ephemeral; once it finishes, its session is no longer needed.

**Implementation:** the executor for runtime tools in ProxyMCPServer must have access to
the `ClientManager` for cleanup. This can be passed through `ProxyMCPServer`:

```python
class ProxyMCPServer:
    def __init__(self, ..., client_manager=None):
        self._client_manager = client_manager
```

### Change 6: Updating the tests

All tests that call `model.ainvoke()` must pass a config with a `thread_id`.

**Pattern:**
```python
# Before:
result = await model.ainvoke([HumanMessage(content="Hello")])

# After:
result = await model.ainvoke(
    [HumanMessage(content="Hello")],
    config={"configurable": {"thread_id": "test_xxx"}},
)
```

**New test replacing `test_ainvoke_default_thread_id`:**
```python
async def test_ainvoke_raises_without_thread_id(self):
    """ainvoke MUST raise ValueError when thread_id is not provided."""
    model = ClaudeCodeChatModel(base_options=None)
    with pytest.raises(ValueError, match="requires explicit thread_id"):
        await model.ainvoke([HumanMessage(content="Test")])
```

## Implementation order

1. **Tests (TDD):** write `test_ainvoke_raises_without_thread_id`, confirm that it fails
2. **`chat_model.py`:** `ensure_config` + mandatory `thread_id` + ValueError
3. **Tests:** update all ainvoke calls so that they pass a thread_id
4. **Verify** that 100% of the tests pass
5. **`task_manager.py`:** pass provider/model from the config into `create_model()`,
   set `_working_dir` on the model
6. **`subagents.py`:** pass a unique thread_id to subagents
7. **`proxy_mcp_server.py`:** add `agent_name`, update the runtime executor
8. **`client_manager.py`:** pass `agent_name` into ProxyMCPServer, cleanup
9. **E2E test:** verify that all 12 tools work, that a subagent does not break the main agent,
   and that TaskManager runs on the claude-code provider

## What does NOT change

- `WebSearchDelegationTool` — creates its own isolated Claude SDK session with a unique
  `session_id`; it does not depend on `thread_id` in ClaudeCodeChatModel
- The `direct_handler` executor invocation logic in ProxyMCPServer — unchanged
  (only the way executors are created for runtime tools changes)
- `ClientManager.get_or_create_client` — already supports multi-thread
- The main agent's calls in `agent_process.py`, `textual_adapter.py`, `non_interactive.py` —
  already pass a thread_id, no changes required

## Risks

1. **`claude` processes:** each subagent spawns a new subprocess. With frequent
   task tool calls there will be many processes. Cleanup (change 4) is critically important.
2. **Cost:** each new Claude SDK session = a new context window = new billing.
   This is expected — a subagent is a separate task with a separate context.
3. **Latency:** creating a new `ClaudeSDKClient` + a `claude` subprocess = ~1-2 s of overhead
   on the first subagent invocation.
