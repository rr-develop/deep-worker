# Design: `reply_back` Parameter for Inter-Agent Messaging

**Status:** Draft — awaiting review
**Author:** AI Assistant
**Date:** 2026-04-18
**Related:**
- `docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md` (v1 of the inter-agent messaging tool)
- `docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md` (turn-end echo middleware)
- `deep_worker_cli/tools.py` (tool definition)
- `deep_worker_cli/default_agent_prompt.md` (system prompt)
- `deep_worker_cli/middleware/send_message_completion.py` (existing middleware)
- `deep_worker_cli/client/agent_client.py` (RPyC transport)
- `deep_worker_cli/rpyc_service/agent_service.py` (enqueue_message endpoint)
- `deep_worker_cli/process/agent_process.py` (agent invocation)
- `deep_worker_cli/agent.py` (middleware registration)

---

## 1. Problem

When agent A sends a message to agent B via `send_message_to_agent("B", "text")`,
agent B sometimes replies to the human user instead of replying back to agent A
via the same `send_message_to_agent` tool. This happens because:

1. The system prompt's instruction is a soft suggestion ("If a reply is
   appropriate, respond by calling `send_message_to_agent`"), not an imperative.
2. No middleware on the receiving side detects incoming inter-agent messages
   and reminds the agent how to reply.
3. The sending agent has no way to express whether it expects a reply back
   (tool channel) or a response to the user (direct assistant output).

## 2. Goal

Give the sending agent explicit, structured control over the reply channel:

- **`reply_back=True` (default)**: The receiving agent MUST reply via
  `send_message_to_agent` back to the sender.
- **`reply_back=False`**: The receiving agent SHOULD respond directly to the
  user as its final assistant message — the sender expects delivery to the user,
  not a tool-mediated reply.

This is achieved by:

1. Adding `reply_back: bool = True` to the tool schema.
2. Transporting the flag to the receiving side alongside the message.
3. Adding a middleware on the receiving side (`InterAgentReplyMiddleware`) that
   injects a cognitive hint into the system prompt based on `reply_back`.
4. Updating the system prompt docs to reflect the new parameter and sharpen
   the reply-behavior instructions.

## 3. Design

### 3.1 Tool Schema Change (`tools.py`)

```python
class _SendMessageToAgentArgs(BaseModel):
    agent_name: str = Field(...)
    message: str = Field(...)
    reply_back: bool = Field(
        default=True,
        description=(
            "Whether the recipient should reply back to this agent via "
            "send_message_to_agent (True, default) or respond directly "
            "to the user as its final assistant output (False). "
            "Use False when the message is meant for the user to see "
            "and the recipient should answer the user, not this agent."
        ),
    )
```

The `reply_back` value is included in the tool's return dict and in the
command sent to the target agent.

### 3.2 Transport Layer

**`tools.py` (sending side)**: Include `reply_back` in the command dict:

```python
resp = await client._send_command({
    "command": "enqueue_message",
    "message": prefixed,
    "reply_back": reply_back,
})
```

**`agent_client.py`**: Route `reply_back` from command to
`exposed_enqueue_message` call as an additional positional argument.

**`agent_service.py`**: `exposed_enqueue_message(message, reply_back=True)` —
pass `reply_back` to `_handle_invoke`.

**`agent_process.py`**: `_handle_invoke` stores `reply_back` in the
`HumanMessage.additional_kwargs` so middleware can read it:

```python
input_message = HumanMessage(
    content=message,
    additional_kwargs={"reply_back": reply_back} if reply_back is not None else {},
)
```

### 3.3 New Middleware: `InterAgentReplyMiddleware`

**File:** `deep_worker_cli/middleware/inter_agent_reply.py`

**Hook:** `before_model` (fires before each model call).

**Logic:**

```
1. Get the last message in state.
2. If it's a HumanMessage whose content matches the inter-agent prefix
   pattern `<name> -> <our-name> :> ...`:
   a. Extract `reply_back` from `additional_kwargs` (default True).
   b. If reply_back == True:
        Inject system-prompt-level hint:
        "You just received a message from agent <sender>. You MUST reply
        using send_message_to_agent(\"<sender>\", \"<your reply>\"). Do NOT
        respond with a text assistant message — use the tool."
   c. If reply_back == False:
        Inject system-prompt-level hint:
        "You just received a message from agent <sender>. This agent expects
        you to respond directly to the user. Provide your final answer as a
        regular assistant message."
3. In all other cases → return None (no-op).
```

**Important:** The middleware only fires once per incoming inter-agent message
(i.e., only on the first `before_model` after the message arrives). On
subsequent iterations it checks that it hasn't already injected a hint for
this message (e.g., by tracking `message.id`).

### 3.4 System Prompt Updates (`default_agent_prompt.md`)

The `send_message_to_agent` section is updated to:

1. Add the `reply_back` parameter to the signature and description.
2. Strengthen the receiving-side instruction from soft suggestion to imperative:
   "When you receive a message from another agent, you MUST reply using
   `send_message_to_agent(\"<sender>\", \"<your reply>\")` — unless
   `reply_back=False` was specified, in which case you should respond
   directly to the user."

### 3.5 `SendMessageCompletionMiddleware` — No Change

This middleware only fires on the **sending** side (the tool call result). It
does not need changes.

### 3.6 Registration

`InterAgentReplyMiddleware` is added to the middleware stack in
`create_cli_agent()` in `agent.py`, alongside `SendMessageCompletionMiddleware`.

---

## 4. Control Flow

### 4.1 Flow with `reply_back=True` (default)

```
Agent A                             Agent B
  |                                    |
  |  send_message_to_agent("B",        |
  |    "hello", reply_back=True)       |
  |  ──────────────────────────────>    |
  |                                    |
  |  [SendMessageCompletionMiddleware  |
  |   ends A's turn]                   |
  |                                    |
  |                          [InterAgentReplyMiddleware
  |                           detects incoming message:
  |                           reply_back=True]
  |                                    |
  |                          Injects cognitive hint:
  |                          "Reply via send_message_to_agent"
  |                                    |
  |                           [Model sees hint and
  |                            calls send_message_to_agent
  |                            back to A]
  |                                    |
  |  <─────────────────── send_message_to_agent("A", "...")
  |                                    |
```

### 4.2 Flow with `reply_back=False`

```
Agent A                             Agent B
  |                                    |
  |  send_message_to_agent("B",        |
  |    "tell the user X",              |
  |    reply_back=False)               |
  |  ──────────────────────────────>    |
  |                                    |
  |                          [InterAgentReplyMiddleware
  |                           detects incoming message:
  |                           reply_back=False]
  |                                    |
  |                          Injects cognitive hint:
  |                          "Respond directly to the user"
  |                                    |
  |                           [Model responds with
  |                            text assistant message
  |                            to the user]
  |                                    |
```

---

## 5. Transport Detail

### 5.1 `agent_client.py` — Command Routing

Current code:

```python
elif command_type == "enqueue_message":
    message = command.get("message", "")
    async_result = self._conn.async_request(
        consts.HANDLE_ASYNC_CALLATTR,
        self._root,
        "exposed_enqueue_message",
        (message,),
        ()
    )
```

Changed to:

```python
elif command_type == "enqueue_message":
    message = command.get("message", "")
    reply_back = command.get("reply_back", True)
    async_result = self._conn.async_request(
        consts.HANDLE_ASYNC_CALLATTR,
        self._root,
        "exposed_enqueue_message",
        (message, reply_back),
        ()
    )
```

### 5.2 `agent_service.py` — Signature Change

```python
async def exposed_enqueue_message(self, message: str, reply_back: bool = True) -> str:
```

Pass `reply_back` to `_handle_invoke`:

```python
await agent_process._handle_invoke(
    {"command": "invoke", "message": message, "stream": True, "reply_back": reply_back},
    ...
)
```

### 5.3 `agent_process.py` — State Injection

In `_handle_invoke`, extract `reply_back` from command and pass to the
existing ``_next_user_message`` closure (which creates ``HumanMessage``
with the ``input_message_id``):

```python
reply_back = command.get("reply_back", True)

# _next_user_message already exists in _handle_invoke and creates
# an id (input_message_id). We extend it to propagate reply_back.
def _next_user_message(content: str) -> HumanMessage:
    nonlocal input_message_id
    input_message_id = f"user-{uuid.uuid4().hex}"
    return HumanMessage(
        content=content,
        id=input_message_id,
        additional_kwargs={"reply_back": reply_back},
    )
```

---

## 6. Middleware Detail: `InterAgentReplyMiddleware`

### 6.1 Location

```
deep_worker_cli/
└── middleware/
    ├── __init__.py
    ├── send_message_completion.py
    └── inter_agent_reply.py    # NEW
```

### 6.2 Class

```python
class InterAgentReplyMiddleware(AgentMiddleware):
    """Inject cognitive hints on the receiving side of inter-agent messages.

    When an agent receives a ``HumanMessage`` that matches the inter-agent
    pattern ``<sender> -> <receiver> :> <body>``, this middleware injects
    a system-prompt hint before the model call based on the ``reply_back``
    flag in the message's ``additional_kwargs``:

    * ``reply_back=True`` (default) → instruct the agent to reply via
      ``send_message_to_agent``.
    * ``reply_back=False`` → instruct the agent to respond directly to
      the user.

    The hint is injected only once per incoming message (tracked by
    message id, or by (sender, content_prefix) as fallback for messages
    without an id).
    """

    tools: list = []

    def __init__(self, name: str) -> None:
        """Args:
            name: The receiving agent's own name (``assistant_id``),
                  used to recognize incoming messages addressed to us.
        """
        self._name = name
        self._handled_message_ids: set[str] = set()
        # Fallback tracking for messages without an id:
        # key = (sender_name, content_prefix)
        self._handled_message_fallbacks: set[tuple[str, str]] = set()

    @hook_config(can_jump_to=[])
    def before_model(self, state: AgentState, runtime=None) -> dict | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None

        # Find the latest HumanMessage that looks like an inter-agent message.
        # Walk backwards to find the most recent incoming inter-agent message.
        for msg in reversed(messages):
            if not isinstance(msg, HumanMessage):
                continue
            if not _is_inter_agent_message(msg, self._name):
                continue

            # Extract sender name from content: "<sender> -> <receiver> :> ..."
            sender = msg.content.split(" -> ")[0] if " -> " in msg.content else ""

            # --- Check if already handled (by id or by fallback) ---
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id in self._handled_message_ids:
                continue  # Already handled by id
            if not msg_id:
                # Fallback: use (sender, content_prefix)
                content = getattr(msg, "content", "") or ""
                prefix = content[:80]  # first 80 chars as fingerprint
                key = (sender, prefix)
                if key in self._handled_message_fallbacks:
                    continue  # Already handled by fallback

            # --- Not yet handled ---

            # Extract reply_back from additional_kwargs (default True)
            kwargs = getattr(msg, "additional_kwargs", {}) or {}
            reply_back = kwargs.get("reply_back", True)

            # Build hint based on reply_back
            if reply_back:
                hint = (
                    f"You just received a message from agent '{sender}'. "
                    f"You MUST reply using send_message_to_agent(\"{sender}\", \"<your reply>\"). "
                    f"Do NOT respond with a regular text assistant message — "
                    f"use the send_message_to_agent tool to reply back to '{sender}'."
                )
            else:
                hint = (
                    f"You just received a message from agent '{sender}' with "
                    f"reply_back=False. This means '{sender}' expects you to "
                    f"provide your final answer as a regular assistant message — "
                    f"do NOT use send_message_to_agent to reply."
                )

            # --- Track as handled ---
            if msg_id:
                self._handled_message_ids.add(msg_id)
            else:
                content = getattr(msg, "content", "") or ""
                prefix = content[:80]
                self._handled_message_fallbacks.add((sender, prefix))

            # --- Find SystemMessage (do NOT create one) ---
            system_msg = _find_system_message(messages)
            if system_msg is None:
                # No system message to extend — skip hint injection
                return None

            # Inject hint into the system message
            updated_system = _append_hint_to_system(system_msg, hint)

            return {"messages": [updated_system if m is system_msg else m for m in messages]}
```

### 6.3 Helper functions

```python
import re

INTER_AGENT_PATTERN = re.compile(
    r"^(.+?) -> (.+?) :> .+"
)

def _is_inter_agent_message(msg: HumanMessage, receiver_name: str) -> bool:
    """Check if a HumanMessage is an inter-agent message addressed to us."""
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    match = INTER_AGENT_PATTERN.match(content)
    if not match:
        return False
    # The receiver must be this agent
    return match.group(2) == receiver_name

def _find_system_message(messages: list) -> SystemMessage | None:
    """Find the first SystemMessage. Returns None if not found."""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return msg
    return None

def _append_hint_to_system(system_msg: SystemMessage, hint: str) -> SystemMessage:
    """Append a cognitive hint to a SystemMessage's content."""
    current = getattr(system_msg, "content", "") or ""
    separator = "\n\n" if current else ""
    new_content = current + separator + hint
    return SystemMessage(content=new_content, id=system_msg.id)
```

### 6.4 Registration in `agent.py`

```python
from deep_worker_cli.middleware.inter_agent_reply import InterAgentReplyMiddleware

# In create_cli_agent(), alongside SendMessageCompletionMiddleware
agent_middleware.append(SendMessageCompletionMiddleware())
agent_middleware.append(InterAgentReplyMiddleware(name=assistant_id))

**Middleware ordering requirement:** ``InterAgentReplyMiddleware`` must come
BEFORE anything that might modify or summarise system messages (e.g.
``SummarizationMiddleware``), but AFTER ``DanglingToolCallMiddleware`` and
``SendMessageCompletionMiddleware``. This ensures the cognitive hint is
visible to the model before summarization truncates it.
```

---

## 7. Backward Compatibility

- `reply_back` defaults to `True` — existing callers that don't pass it
  get the current behavior (reply expected).
- The `exposed_enqueue_message` signature change is backward-compatible:
  `reply_back: bool = True` with a default.
- Old senders (without `reply_back`) talking to new receivers: the
  `command.get("reply_back", True)` fallback in `agent_client.py` ensures
  `True` is used.
- New senders (with `reply_back`) talking to old receivers: the
  default parameter `reply_back: bool = True` in `exposed_enqueue_message`
  ensures the extra argument is accepted.

---

## 8. System Prompt Updates (`default_agent_prompt.md`)

### Current text (line 81-108):

The `send_message_to_agent` section is updated:

**Signature line** changes from:
```
Signature: `send_message_to_agent(agent_name, message)`.
```
to:
```
Signature: `send_message_to_agent(agent_name, message, reply_back=True)`.
```

**New parameter description** added after the existing `message` description:

```
**reply_back** (bool, default=True):
When True (default), the receiving agent MUST reply via the same
send_message_to_agent tool back to you. When False, the receiving agent
should respond directly to the user — use this when you want the recipient
to deliver information to the user rather than replying to you.
```

**Receiving-side instruction** strengthened:

Old:
```
If a reply is appropriate, respond by calling `send_message_to_agent("<some-name>", "<your reply>")`.
```

New:
```
When you receive a message from another agent, you MUST reply using
send_message_to_agent("<sender>", "<your reply>") — unless reply_back=False
was specified by the sender, in which case you should respond directly
to the user instead.
```

---

## 9. System Prompt: Removing Cognitive Conflict

The current `default_agent_prompt.md` (lines 99-104) contains:

```
**Usage policy (IMPORTANT — follow strictly).** Do NOT use this tool by default.
Use it ONLY when one of the following is true:
1. The human user has explicitly asked you to pass a message or a command to
   another agent...
2. You received an agent-to-agent message earlier in this conversation...
```

This creates a **cognitive conflict**: line 97 says "reply via `send_message_to_agent`"
but lines 99-104 say "Do NOT use this tool by default." An LLM resolving this
tension often defaults to the prohibition (more emphatic, first in the list).

### Fix

Restructure the usage policy into two distinct sections — one for **outgoing**
(initiating messages) and one for **incoming** (replying to peers):

```
**Usage policy for OUTGOING messages (initiating contact):**
Do NOT use this tool to initiate conversations on your own, broadcast to
multiple agents, or send unsolicited status updates. Only use it when:
1. The human user explicitly asks you to pass a message to another agent.
2. You were previously instructed by another agent or by the user to do so.

**Usage policy for INCOMING messages (replying to a peer):**
When you RECEIVE a message from another agent (content starts with
`<sender> -> <your-name> :> ...`), you MUST reply using
`send_message_to_agent("<sender>", "<your reply>")` unless the sender
specified `reply_back=False`. This is NOT optional — it is the only
way to respond to a peer.
```

This removes the conflict:
- **Initiating**: restricted, "Do NOT use by default"
- **Replying**: mandatory, "you MUST reply"

---

## 10. Open Questions

1. **Middleware on sending side?** Should `SendMessageCompletionMiddleware`
   also check `reply_back` and skip the turn-end echo when `reply_back=False`?
   — No change needed: the middleware fires on any successful send, regardless
   of `reply_back`. The echo is about recording the outgoing message in
   history, not about reply semantics.

2. **Multiple inter-agent messages in same history?** The middleware tracks
   handled message IDs to avoid re-injecting the same hint. This is handled
   by the `_handled_message_ids` set.

3. **InterAgentReplyMiddleware ordering?** Must run AFTER the system prompt
   is assembled (which happens in earlier middlewares) but BEFORE any
   summarization middleware that might collapse the system message. It should
   be placed right before `SummarizationMiddleware`.
