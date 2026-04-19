# Design: `SendMessageCompletionMiddleware` — Treat `send_message_to_agent` as a Turn-Final Response

**Status:** ✅ Approved — ready for implementation
**Date:** 2026-04-18
**Related:**
- `docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md` (v1 of the inter-agent messaging tool)
- `deep_worker_cli/tools.py` (existing tool implementation)
- `deep_worker_cli/local_context.py` (reference `AgentMiddleware` subclass in the project)
- `deepagents/deepagents/middleware/patch_tool_calls.py` (reference: middleware that mutates state before model call)
- `langchain.agents.middleware.tool_call_limit` (reference: `@hook_config(can_jump_to=["end"])` pattern used by LangChain-stock middleware)
- The web UI's agent service (`get_agent_dialog_formatted` — the client that loses inter-agent messages today)

---

## 1. Problem

When an agent calls `send_message_to_agent("beta", "hello")`:

1. Model produces `AIMessage(content="", tool_calls=[send_message_to_agent(...)])` — content is empty, only a tool call.
2. `ToolNode` executes the tool → `ToolMessage(name="send_message_to_agent", content=<json with success / delivered_content>)`.
3. Model is invoked again with the ToolMessage and usually produces a short filler `AIMessage` like *"The message has been sent to Eva. Waiting for a reply."* or, often, an empty one.

In the web UI's Full Dialog view, `get_agent_dialog_formatted` builds pairs `(HumanMessage, AIMessage-with-non-empty-content)` and **skips** every `ToolMessage` and every `AIMessage` whose `content` is empty. The result: the **actual** text the agent sent to another agent (the prefixed `agent:<sender>:> <message>`) is invisible in Full Dialog — only `<tool-call>` + `<filler>` remain, and the filler often adds no information.

The user explicitly wants: **sending a message to another agent is the agent's final response for this turn**. Its content is just addressed to another agent instead of the human user. After it, the agent waits for an incoming message (from the peer or from the human).

## 2. Goal

Make the tool call to `send_message_to_agent` effectively terminate the turn with a **text** `AIMessage` whose `content` is exactly the delivered prefixed message (`agent:<sender>:> <message>`). This:

- Guarantees the outgoing message shows up in **any** history-based client (Full Dialog, `dwcli --history`, TUI), not just one.
- Is symmetric with how the **recipient** sees the same text in its own `HumanMessage`, so the two halves of the dialog render identically in both agents' histories.
- Removes a dangling LLM call that would only produce filler, saving tokens and latency.

## 3. Why a middleware (and not a `Command` from the tool)

A LangChain tool can return `Command(update={"messages": [...]})` to augment state. It **cannot** jump to the graph's end node — only the standard agent control flow (modelcall ↔ toolcall) runs after a tool returns. So if we stuff an `AIMessage` into the tool's return, the agent will still re-enter the model node afterwards and generate a second assistant message on top of ours.

`langchain.agents.middleware` provides a first-class, documented mechanism:

```python
@hook_config(can_jump_to=["end"])
def before_model(self, state, runtime):
    return {"messages": [AIMessage(...)], "jump_to": "end"}
```

Used by stock middleware (`ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`, `PIIMiddleware`, `OpenAIModerationMiddleware`, `openai_moderation`). Not a hack.

Choice: **middleware with a `before_model` hook** that fires on the next model-call iteration after `send_message_to_agent` succeeded, injects the echo `AIMessage`, and sets `jump_to = "end"` — so the model is **not** called and the turn ends cleanly.

## 4. Control Flow

```
  User prompt
    │
    ▼
  [model]  →  AIMessage(tool_calls=[send_message_to_agent(...)], content="")
    │
    ▼
  [tools]  →  ToolMessage(name="send_message_to_agent",
                          content='{"success":true,"delivered_content":"agent:alpha:> hi"}')
    │
    ▼
  [before_model]  ← SendMessageCompletionMiddleware kicks in HERE
    │
    │  Check: is the last message a successful send_message_to_agent result?
    │
    ├─ NO  → return None; normal flow (model will be invoked)
    │
    └─ YES → return {"messages": [AIMessage(content="agent:alpha:> hi")],
                     "jump_to": "end"}
            │
            ▼
           [end]   ← turn finishes, no further model call
```

The echo `AIMessage.content` is literally `delivered_content` from the tool result, i.e. the exact `agent:<sender>:> <message>` text.

## 5. Component Design

### 5.1 Location

```
deep_worker_cli/
└── middleware/
    ├── __init__.py                  # NEW (empty package marker)
    └── send_message_completion.py   # NEW
```

A new `deep_worker_cli.middleware` subpackage. The project already keeps middleware-like code in a single file (`local_context.py`) — we create a proper subpackage to hold this and any future agent middlewares.

### 5.2 Class

```python
# deep_worker_cli/middleware/send_message_completion.py

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# The name of the tool whose successful results terminate the turn.
# Kept as a module-level constant to avoid typos in tests.
TOOL_NAME = "send_message_to_agent"


class SendMessageCompletionMiddleware(AgentMiddleware):
    """Turn-ending middleware for ``send_message_to_agent`` calls.

    When the last message in state is a successful ``ToolMessage`` from
    ``send_message_to_agent``, inject an ``AIMessage`` whose ``content`` is
    the prefixed text that was delivered to the target agent (``agent:<sender>:> <text>``)
    and jump to the graph's end node, skipping the upcoming model call.

    Rationale: an outgoing inter-agent message *is* the agent's final
    response for this turn — it just happens to be addressed to another
    agent instead of the human user. Making that fact explicit in the
    conversation state means every history-based client (Full Dialog,
    dwcli --history, TUI) shows the outgoing text as-is, symmetric to
    how the recipient sees it as an incoming ``HumanMessage``.
    """

    tools: list = []  # no new tools registered

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]

        # Must be a ToolMessage named after our tool.
        if not isinstance(last, ToolMessage):
            return None
        if getattr(last, "name", None) != TOOL_NAME:
            return None

        # Parse the tool result. Our tool returns a dict; LangGraph
        # serializes it to str(dict) when constructing ToolMessage, or
        # it may be a plain dict on some code paths. Handle both.
        raw = last.content
        data: dict[str, Any] | None = None
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # Some LangChain versions serialize dict with str() instead
                # of json.dumps() — fall back to ast.literal_eval.
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        data = parsed
                except (ValueError, SyntaxError):
                    data = None

        if data is None:
            logger.debug("SendMessageCompletionMiddleware: could not parse tool result")
            return None

        if not data.get("success"):
            # Delivery failed (agent_not_found, connection_failed, etc.).
            # Let the model handle the error normally — do NOT swallow it.
            return None

        delivered = data.get("delivered_content", "")
        if not isinstance(delivered, str) or not delivered.strip():
            return None

        # Replace the upcoming model call with an echo AIMessage and jump to end.
        logger.info(
            "SendMessageCompletionMiddleware: ending turn with echo content=%r",
            delivered[:120],
        )
        echo = AIMessage(content=delivered)
        return {"messages": [echo], "jump_to": "end"}
```

Notes on the implementation:

- **Single-tool focus.** We only terminate the turn when the *last* message is a successful `send_message_to_agent` ToolMessage. If the model called multiple tools in parallel (e.g. `send_message_to_agent` + `execute`), after tool execution the last message will be one of the two ToolMessages — whichever came last. The model will still re-enter on the other one normally in most multi-tool fan-outs, and if the last happens to be our tool we end the turn. Edge-case details below (§7).
- **Failure-aware.** Delivery failures (`success: False`) bypass the middleware — the error message reaches the model so it can react.
- **No disruption to HITL.** Approval runs on the `AIMessage(tool_calls=[...])` node, *before* the tool. By the time our middleware looks at state, approval has either succeeded (tool ran, we see its ToolMessage) or rejected (no ToolMessage from our tool at all, middleware returns None).

### 5.3 Wiring

`agent.py::create_cli_agent`:

```python
from deep_worker_cli.middleware.send_message_completion import (
    SendMessageCompletionMiddleware,
)

agent_middleware.append(SendMessageCompletionMiddleware())
```

No parameters are needed — the middleware reads everything it needs from the ToolMessage content that the tool itself writes.

### 5.4 TaskManager interaction

TaskManager (`agent_process.py::_handle_invoke`, both stream and non-stream paths) takes the final `AIMessage.content` as `full_response` and feeds it to `task_manager.on_subordinate_response(response=...)`. With this middleware, `full_response` will be `"agent:<self.name>:> <text>"` for inter-agent turns.

That is **not** a real task response. If we hand it to TaskManager, the manager LLM will evaluate it against the initial user request and almost certainly say `in_progress` + "you didn't complete the task, you just sent a message" — triggering a corrective loop that was not asked for.

**Fix:** in both the stream-mode and non-stream-mode TaskManager branches of `_handle_invoke`, skip `task_manager.on_subordinate_response(...)` **iff** `full_response` starts with `f"agent:{self.name}:> "`. Treat such turns as *"waiting for peer"* — the task is neither done nor unable, we simply have no new subordinate output to evaluate, so we exit the TM loop without a corrective prompt.

Code sketch (stream mode, current `agent_process.py:1135-1169`):

```python
if self.task_manager:
    # An outgoing inter-agent message is not a response to the initial task —
    # skip TaskManager evaluation and exit the loop, waiting for a peer reply.
    if full_response.startswith(f"agent:{self.name}:> "):
        print(
            f"[STREAM] Skipping TaskManager: outgoing inter-agent message",
            file=sys.stderr,
        )
        break

    corrective_prompt = await self.task_manager.on_subordinate_response(...)
    ...
```

Same guard in the non-stream branch.

### 5.5 `default_agent_prompt.md`

Append one sentence to the existing "Inter-Agent Messaging" section:

> When you call `send_message_to_agent`, the call itself **is** your final response for this turn — the system will record your outgoing message as the assistant reply. Do NOT also produce a textual preamble like "I will now send a message" or a follow-up like "Waiting for the reply"; those add noise. Just call the tool. If the delivery fails, you will see the error in the tool result and can decide what to do next.

## 6. Data & State

- **New state keys:** none. The middleware uses the standard `jump_to` key already declared on `AgentState`.
- **Persistence:** the echo `AIMessage` is a regular LangGraph message and flows through the normal checkpointer (SQLite / memory). `dwcli --history`, Full Dialog, TUI subscribers all see it without any extra code.

## 7. Edge Cases

| Situation | Behaviour |
|---|---|
| Delivery fails (`success: False`) | Middleware returns `None`; model runs, sees error, reacts normally. |
| `send_message_to_agent` + other tool called in the same AIMessage | Middleware only triggers if the **last** ToolMessage in state is our tool. If both tools ran and `send_message_to_agent` came last, turn ends (which is fine — model would just produce filler anyway). If our tool came first and another one last, middleware does nothing and model runs; user still sees the echo on the NEXT loop iteration if the remaining flow produces one, OR the outgoing message simply won't be echoed. This is acceptable for v1; multi-tool parallel calls to `send_message_to_agent` are not an expected case. |
| Tool rejected at HITL | No ToolMessage produced for our tool; middleware returns `None`. |
| Multiple consecutive `send_message_to_agent` calls (model calls it repeatedly) | Each call that ends with a successful ToolMessage as the last state message terminates the turn right there. No runaway. |
| Checkpointed history contains a dangling `send_message_to_agent` tool call from a previous process | `PatchToolCallsMiddleware` (already installed) adds a synthetic ToolMessage with a "cancelled" content. Our middleware sees it, fails to parse it as JSON → returns `None`. No interference. |
| `content` is a content-block list (Claude multimodal) instead of a plain string | `ToolMessage.content` from our tool is always a dict (serialized). `AIMessage.content` we set is a plain string — compatible with every provider. |
| LangGraph version without `hook_config`/`jump_to` | The project pins `langchain.agents.middleware` via the vendored `deepagents` dependency tree, which already ships `hook_config` (see `middleware/tool_call_limit.py`). No extra pinning needed. |

## 8. Testing Plan

### 8.1 Unit tests (new file)

`tests/unit_tests/middleware/test_send_message_completion.py`

1. `test_non_tool_message_passes_through` — state ends with a HumanMessage → returns `None`.
2. `test_tool_message_from_other_tool_passes_through` — ToolMessage with `name="execute"` → returns `None`.
3. `test_failed_delivery_passes_through` — ToolMessage from our tool with `{"success": false, ...}` → returns `None`.
4. `test_successful_delivery_ends_turn_with_echo` — ToolMessage from our tool with `{"success": true, "delivered_content": "agent:alpha:> hi"}` → returns `{"messages": [AIMessage(content="agent:alpha:> hi")], "jump_to": "end"}`.
5. `test_tool_content_as_string_json` — same as (4) but `content` is a JSON-encoded string.
6. `test_tool_content_as_python_repr_string` — same as (4) but content is `str(dict)` (single quotes).
7. `test_unparseable_content_passes_through` — ToolMessage whose content is `"weird"` → returns `None`.
8. `test_empty_delivered_content_passes_through` — `delivered_content=""` → returns `None`.
9. `test_empty_messages_state_passes_through` — `state["messages"] = []` → returns `None`.
10. `test_missing_messages_key_passes_through` — `state = {}` → returns `None`.
11. `test_hook_config_allows_jump_to_end` — verify the `before_model` method has `__can_jump_to__ = ["end"]` metadata (so LangGraph actually routes to end).

### 8.2 Wiring test

`tests/unit_tests/test_agent.py` (extend):

- `test_create_cli_agent_registers_send_message_completion_middleware` — patch `create_deep_agent` in the deepagents module, call `create_cli_agent`, assert that one of the middlewares passed in is an instance of `SendMessageCompletionMiddleware`.

### 8.3 TaskManager-skip test

`tests/unit_tests/test_agent_process_metadata.py` or a new file:

- `test_task_manager_skipped_for_outgoing_inter_agent_message` — construct an `AgentProcess` with a mocked `task_manager`, drive `_handle_invoke` through a path where the final `full_response` starts with `f"agent:{self.name}:> "`, and assert `task_manager.on_subordinate_response` is NOT called.
- `test_task_manager_runs_for_normal_response` — same setup, `full_response="normal reply"`, assert `on_subordinate_response` IS called.

### 8.4 Integration-ish test (optional, stretch)

Use a `GenericFakeChatModel` like in existing `deepagents/tests/unit_tests/test_subagents.py`: script it to produce `AIMessage(tool_calls=[send_message_to_agent(beta, "hi")])` followed by `AIMessage(content="unused filler")`. Monkey-patch `AgentClient` to accept the send without network. Invoke the agent. Assert that the final state's last `AIMessage.content == "agent:alpha:> hi"` (ours) and the filler from the second scripted response is **not** in the messages list — proving that `jump_to=end` skipped the model call.

## 9. Implementation Checklist (for the green phase)

1. Create `deep_worker_cli/middleware/__init__.py` and `deep_worker_cli/middleware/send_message_completion.py`.
2. Add `SendMessageCompletionMiddleware` per §5.2.
3. Wire it into `agent.py::create_cli_agent` as an unconditional entry in `agent_middleware` (same level as `LocalContextMiddleware`, `ProjectMemoryMiddleware`).
4. In `process/agent_process.py::_handle_invoke`, add the TaskManager-skip guard (§5.4) in **both** stream and non-stream branches.
5. Append the sentence in §5.5 to `default_agent_prompt.md`.
6. Tests per §8.
7. Commit.

## 10. Non-Goals

- Changing `send_message_to_agent` tool behaviour (the tool still delivers + returns the same dict; middleware is purely additive).
- Changing the web UI client code. The fix is transparent to it — Full Dialog's pair parser will now find a non-empty `AIMessage.content` and render it as-is.
- Rewriting the history format or adding new marker types. The echo uses the existing `agent:<sender>:> ` convention.
- Handling the exotic case of multiple parallel `send_message_to_agent` calls in one AIMessage (see §7). Left for a future iteration if it ever becomes a real use case.

## 11. Resolved Decisions

- **Q: tool return vs middleware?** Middleware. Tools can't end a LangGraph turn. Confirmed by reading LangChain stock middlewares.
- **Q: echo format?** `agent:<sender>:> <text>` — identical to what the recipient sees. No new marker.
- **Q: TaskManager?** Skip evaluation for outgoing inter-agent turns (content starts with `agent:<self.name>:> `). User confirmed this is the right semantics.
- **Q: HITL compat?** Works naturally; rejection produces no ToolMessage for our tool, middleware stays out of the way.
- **Q: Multiple send_message calls in one AIMessage?** Out of scope for v1. Documented in §7.

**Path of this design doc:**
`deep_worker_cli/docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md`
