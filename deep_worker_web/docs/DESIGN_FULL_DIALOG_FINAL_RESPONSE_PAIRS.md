# Design: Full Dialog — User → Final Assistant Response Pairs

**Status:** 📋 PROPOSED
**Date:** 2026-04-19
**Related bug:** Full Dialog truncates conversation, dropping the agent's
final answer (observed on a production agent: 96 of 104 text-bearing AIMessages lost).

---

## 1. Problem

`GET /api/agents/{name}/dialog-text` returns a plain-text rendering of the
conversation built by `AgentService.get_agent_dialog_formatted()`
(`app/services/agent_service.py:1089–1170`). It uses a naive **pair parser**:

```python
for message in history:
    if msg_type == 'HumanMessage':
        current_user_query = content
    elif msg_type == 'AIMessage' and current_user_query is not None:
        if content and content.strip():
            dialog_pairs.append((current_user_query, content))
            current_user_query = None        # ← closes pair, drops the rest
```

This assumes a single `HumanMessage → AIMessage` exchange. Real agent turns
look like:

```
HumanMessage("...")
AIMessage("Let me check…")            ← parser captures THIS as "the answer"
ToolMessage(grep results)
AIMessage("No matches for X")         ← LOST
ToolMessage(...)
AIMessage("Let me also check Y")      ← LOST
ToolMessage(...)
AIMessage("Confirmed: zero ...")      ← FINAL ANSWER, LOST
```

Because the first text-bearing AIMessage usually says something trivial
like *"I'll search for these phrases and fix them."*, **the user sees the
preamble and never the actual conclusion**.

### Measured impact (production agent, 2026-04-19)

| Metric | Value |
|---|---|
| Total messages in checkpoint | 394 |
| `HumanMessage` | 8 |
| `AIMessage` total | 197 |
| `AIMessage` with text content | **104** |
| `AIMessage` shown by current parser | **8** |
| `AIMessage` with text **silently dropped** | **96** |

Per-turn breakdown (loss by turn):

```
After Human[ 0]:  5 text AIs → shown 1, lost  4
After Human[14]: 14 text AIs → shown 1, lost 13
After Human[72]: 34 text AIs → shown 1, lost 33
After Human[206]:34 text AIs → shown 1, lost 33
After Human[326]: 5 text AIs → shown 1, lost  4
After Human[342]: 4 text AIs → shown 1, lost  3
After Human[360]: 3 text AIs → shown 1, lost  2
After Human[374]: 5 text AIs → shown 1, lost  4   ← final answer dropped
```

The user complaint that triggered this design ("there is no last model
message") is exactly turn `374`: the actual final report
`"Confirmed: zero occurrences of «trained*»…"` was message `[393]` but the
parser closed the pair at `[375]` ("I'll search for these phrases and fix
them.") and discarded everything after.

### Three sources of "user" messages

The `HumanMessage` slot in agent history is **not just direct user input**.
Three sources legitimately deliver "user-side" messages, and **all three
must appear in the dialog as first-class user turns** at every verbosity
level — they all drive the agent's behaviour and the reader needs to see
what the agent was asked at every step:

1. **Direct user input** — what the user typed in dwcli/web.
2. **TaskManager corrective prompts** — content begins with `TaskManager:>`.
   These are inputs the in-process TaskManager sends to keep the agent on
   task.
3. **Inter-agent messages** — content begins with `agent:<sender>:> `.
   Sent by other agents via `send_message_to_agent()`.

The current parser does process them but loses them just like real user
messages because of the larger pair-closing bug. The fix MUST keep them
visible at every verbosity level. They must be **labelled** so the reader
can tell which source initiated each turn (the `TaskManager:>` and
`agent:<name>:> ` prefixes are reliable, unambiguous markers — we use
them precisely to choose the right header), but they must NEVER be
filtered out — that visibility is exactly what makes the dialog useful for
post-mortem inspection of what happened to the agent.

### Summarization injections — the only thing we DO drop

When the agent's own context overflows, the model's summarization step
injects a synthetic `HumanMessage` whose body begins with
*"You are in the middle of a conversation that has been summarized…"*
(visible as `msg[0]` in the affected-agent fixture). This is **not** a message
anyone sent — it is the agent's internal context-management bookkeeping.
Rendering it as a user turn would actively mislead the reader. It is the
only `HumanMessage` flavour that is filtered.

### System messages are first-class in history rendering (POLICY)

**Rendering full conversation history WITHOUT system prompts is misleading
and is an outright lie. It is ABSOLUTELY UNACCEPTABLE and is forbidden.**

`SystemMessage`s carry two things that are essential to understanding what
the agent actually saw:

1. The agent's **base system prompt** (the leading SystemMessage), and
2. **Runtime middleware hints** injected mid-conversation — most notably the
   inter-agent reply hint from `InterAgentReplyMiddleware`, which tells the
   receiving agent to reply via `send_message_to_agent`. This hint is
   injected as a SystemMessage right after the incoming peer message.

Historically both the agent-side serializer (`serialize_message` in
`process/message_dict_cache.py`) and the web renderer
(`dialog_formatter.split_into_turns`) silently dropped every SystemMessage:
the serializer emitted `{"type": "SystemMessage", "id": ...}` with **no
`content`**, and the renderer matched only `HumanMessage`/`AIMessage`, so the
SystemMessage fell through and vanished. A reader inspecting the Full Dialog
therefore saw an inter-agent request followed directly by the agent's reply,
with **no trace** of the hint that was actually in the agent's context — a
rendering that misrepresents history.

**Required behaviour (binding):**

- System messages MUST be rendered at every verbosity **at or above `text`**
  (i.e. `text` — "All assistant text" — and `full`).
- They are rendered under a dedicated `--- System ---` header, in
  chronological order relative to the assistant text and tool-calls of the
  same turn (so a mid-turn hint appears between the user message and the
  assistant response that followed it).
- A leading system prompt (before any HumanMessage) is rendered as a
  standalone `--- System ---` preamble block.
- They are intentionally omitted only from the most compact `final` view,
  whose contract is strictly "one final assistant text per user turn".

Any future history-rendering surface (web, CLI, exports) MUST honour this
policy. Producing a "full" or "text" history view that hides system messages
is a regression, not an optimisation.

### Test coverage gap

`get_agent_dialog_formatted` has **zero unit tests** today:

```
$ grep -r "get_agent_dialog_formatted\|dialog-text" tests/unit/
(no matches)
```

The only existing tests (`tests/test_full_dialog.py`) are Playwright UI
smoke tests that just check the endpoint returns *some* text containing
`---`. They would have happily passed with the current bug.

---

## 2. Goals

1. **Default rendering** is "user query ↔ final assistant text response"
   pairs in chronological order, where "user query" includes direct user
   input, TaskManager prompts, and inter-agent messages — all three are
   first-class turn drivers and MUST appear at every verbosity level.
2. The **last** pair must include the **last text-bearing AIMessage**
   *after* the most recent user-side message — i.e. the genuinely final
   answer, not the first preamble.
3. If the most recent user-side message has no text-bearing AIMessage yet
   (agent is still working / interrupted), that user-side message MUST
   still be rendered with an explicit "(no response yet)" placeholder.
4. Label each user-side turn so the reader can see who initiated it
   (`User` / `TaskManager` / `agent:<name>`).
5. Drop only summarization injections (see §1).
6. **Verbosity control** via query parameter — opt-in, never default:
   - `?verbosity=final` (default): every user-side turn + that turn's
     final assistant text only.
   - `?verbosity=text`: every user-side turn + **all** intermediate
     text-bearing AIMessages of that turn + **all SystemMessages**
     (base prompt and middleware hints), under `--- System ---` headers.
   - `?verbosity=full`: like `text` plus **compact** tool-call summaries
     (one line per call: tool name + truncated args).
   - System messages are first-class at `text`/`full` and forbidden to hide
     there (see §1 "System messages are first-class in history rendering").
7. **Comprehensive unit tests** must accompany the change.

### Non-goals

- Not changing the wire format (still `text/plain` from the endpoint).
- Not changing dialog.html / Full Dialog UI in this design (separate task
  if we ever switch to JSON+HTML rendering).
- Not changing the underlying RPyC `get_history` payload — the fix is
  purely in `get_agent_dialog_formatted` and its caller endpoint.

---

## 3. Design

### 3.1 New algorithm — "last text wins" within a turn

A *turn* is the slice of messages from a `HumanMessage` (inclusive) to the
next `HumanMessage` (exclusive). For every turn:

1. Identify the leading `HumanMessage` and classify it (User / TaskManager
   / AgentEcho / Summarization). Summarization turns are dropped entirely;
   the other three are all rendered.
2. Walk the rest of the turn collecting:
   - all text-bearing AIMessages (in order),
   - all tool calls (name + truncated args), regardless of which AIMessage
     attached them,
   - tool-only AIMessages have no text contribution.
3. Emit one rendered turn block whose shape depends on `verbosity`:
   - `final`  → header + **last** text AIMessage (or "(no response yet)").
   - `text`   → header + **all** text AIMessages, in order.
   - `full`   → header + interleaved text AIs and tool-call summaries
     in their original order.

### 3.2 Header formatting

| Source | Header | Body |
|---|---|---|
| Real user | `--- User ---` | message content as-is |
| TaskManager | `--- TaskManager ---` | content with `TaskManager:>` prefix stripped |
| Inter-agent | `--- From agent:<name> ---` | content with `agent:<name>:> ` prefix stripped |
| Assistant (default & text) | `--- Assistant ---` | one or more text blocks (joined with blank line for `text` mode) |
| Assistant (full mode, tool call) | `--- Assistant (tool: <name>) ---` | one-line `args` summary, truncated to 200 chars |

Headers serve as labels only — TaskManager and agent-echo turns are
otherwise treated identically to direct user turns (same algorithm,
same placeholder for missing reply, same paired layout).

A turn with no response yet renders, regardless of source:

```
--- TaskManager ---
<query>

--- Assistant ---
(no response yet)
```

### 3.3 Detection rules (HumanMessage classification)

Order matters — first match wins:

```python
def classify_human(content: str) -> Literal["user","taskmanager","agent_echo","summarization"]:
    if content.startswith("TaskManager:>"):
        return "taskmanager"
    if content.startswith("agent:") and ":> " in content[:64]:
        return "agent_echo"
    if "Context Extraction Assistant" in content[:200] \
       or content.startswith("You are in the middle of a conversation that has been summarized"):
        return "summarization"
    return "user"
```

Only the `summarization` branch is filtered. The other three (`user`,
`taskmanager`, `agent_echo`) all produce visible turns in the rendered
dialog. The classifier exists solely to choose the header label and to
strip prefixes from the body.

### 3.4 Text extraction from AIMessage

The current parser only handles `content: str`. AIMessage content can also
be `list[ContentBlock]`. The fix must handle both:

```python
def extract_text(msg) -> str:
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join(
            b.get('text', '') for b in c
            if isinstance(b, dict) and b.get('type') == 'text'
        )
    return ''
```

Empty string after `.strip()` ⇒ "no text", treat as tool-only.

### 3.5 Tool-call summary (verbosity=`full` only)

```python
def format_tool_call(tc: dict) -> str:
    name = tc.get('name', '?')
    args = tc.get('args', {})
    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 200:
        args_str = args_str[:197] + '...'
    return f"→ {name}({args_str})"
```

Only emitted in `full` mode. Output shape:

```
--- Assistant (tool: execute) ---
→ execute({"command": "grep -rn 'foo' src/"})
```

### 3.6 Public API surface

#### Endpoint

```
GET /api/agents/{agent_name}/dialog-text?verbosity=final
                                                =text
                                                =full
```

Default: `verbosity=final`. Unknown values → 422 Unprocessable Entity.
Backward-compatible: existing callers (no query param) keep working but
**now see the correct content** (final answers, not preambles, plus
TaskManager / inter-agent messages that were always supposed to be there).

Response stays `text/plain; charset=utf-8`.

**No size limits.** The endpoint always returns the entire conversation.
There is no `limit`/`offset`/`max_messages` knob. The dialog grows only
with the agent's actual context, and the natural ceiling is whatever the
agent itself can hold.

#### Service method

```python
async def get_agent_dialog_formatted(
    self,
    agent_name: str,
    verbosity: Literal["final", "text", "full"] = "final",
) -> str: ...
```

The method body is rewritten around the §3.1 algorithm. Backward callers
without `verbosity=` get `"final"` — same default as the endpoint.
The method requests **all** messages from RPyC `get_history` (no `limit`).

#### RPyC server contract change (mandatory)

`AgentProcess._handle_get_history` and `AgentService.exposed_get_history`
must accept "no limit" as a sentinel meaning "return everything":

- `limit=None` → return all messages (preferred new contract)
- `limit=0` or `limit<0` → return all messages (defensive)
- `limit=N` (positive int) → return last N messages (legacy callers)
- the previously-hardcoded `limit=50` default at server side becomes
  "no limit" (i.e. `None`).

Existing callers (`dwcli`, etc.) that explicitly pass a positive `limit`
keep working unchanged.

#### UI: verbosity dropdown

The Full Dialog button in `static/index.html` becomes a small button
group: a `<select id="full-dialog-verbosity">` with three options
(`Final answers only` / `All assistant text` / `Full (with tool calls)`)
and the existing `📜 Full Dialog` button. Clicking the button reads the
selected verbosity and opens

```
/api/agents/{name}/dialog-text?verbosity=<selected>
```

in a new tab. The default selected option is `final` so the no-click
flow matches the no-query-param flow.

Persistence: the chosen verbosity is saved in `localStorage` so the user
doesn't have to reselect it every time. Key: `dialog_verbosity`.

### 3.7 Pseudocode (reference implementation)

```python
async def get_agent_dialog_formatted(
    self, agent_name: str, verbosity: str = "final"
) -> str:
    history = await self._fetch_history(agent_name, limit=10000)
    turns = self._split_into_turns(history)
    rendered = []
    for turn in turns:
        kind = classify_human(turn.user_text)
        if kind == "summarization":
            continue
        rendered.append(self._render_turn(turn, kind, verbosity))
    return "\n\n".join(rendered) if rendered else "No conversation history yet."

def _split_into_turns(self, history):
    turns = []
    current = None
    for msg in history:
        if msg["type"] == "HumanMessage":
            if current is not None:
                turns.append(current)
            current = Turn(human=msg, ai_messages=[], tool_calls=[])
        elif current is not None:
            if msg["type"] == "AIMessage":
                current.ai_messages.append(msg)
                for tc in msg.get("tool_calls", []) or []:
                    current.tool_calls.append((len(current.ai_messages)-1, tc))
            # ToolMessage payload is not rendered; it's noisy and the
            # corresponding tool-call already appears via AIMessage.tool_calls.
    if current is not None:
        turns.append(current)
    return turns
```

### 3.8 Edge cases

| Case | Behaviour |
|---|---|
| Empty history | `"No conversation history yet."` (unchanged) |
| Only summarization HumanMessages | `"No conversation history yet."` |
| Trailing user-side message (any source) with no AI yet | Render header + `(no response yet)` |
| Two consecutive HumanMessages (e.g. user + immediate TaskManager prompt) | Two turns, second one shows `(no response yet)` if no AI between them |
| AIMessage with `content=""` and tool_calls | Counts as no-text in `final`/`text`; emitted as tool-call line in `full` |
| AIMessage with `content` as block list, no `text` block | Counts as no-text |
| `tool_calls` arg containing huge blobs | Truncated to 200 chars |
| HumanMessage starts with `agent:` but prefix malformed (no `:> `) | Falls through to `user` |
| TaskManager prompt rendered without `TaskManager:>` prefix | Already stripped, kept readable |
| Multi-byte / Cyrillic content | `ensure_ascii=False` everywhere |

---

## 4. Test plan (TDD)

All tests live in `tests/unit/test_dialog_formatting.py`.
They feed `get_agent_dialog_formatted` (or its pure helper) a synthetic
`history` list — no RPyC, no real agent, fast and deterministic.

### 4.1 Default mode (`verbosity=final`)

| # | Name | Setup | Expectation |
|---|---|---|---|
| 1 | `test_empty_history` | `[]` | exact string `"No conversation history yet."` |
| 2 | `test_single_user_no_response` | 1 HumanMessage | `--- User ---\n<q>\n\n--- Assistant ---\n(no response yet)` |
| 3 | `test_single_user_with_simple_answer` | Human + AIMessage("ok") | one pair, AI = `"ok"` |
| 4 | `test_takes_last_text_ai_not_first` | Human + AIMsg("preamble") + AIMsg("middle") + AIMsg("FINAL") | renders FINAL only |
| 5 | `test_skips_tool_only_ai_messages` | Human + AIMsg(tool_calls=...) + AIMsg("done") | renders "done" |
| 6 | `test_handles_content_blocks` | Human + AIMsg(content=[{type:text,text:"hi"}]) | renders "hi" |
| 7 | `test_handles_mixed_content_blocks` | Human + AIMsg(content=[{type:text,text:"a"},{type:tool_use,...}]) | renders "a" |
| 8 | `test_intermediate_tool_calls_skipped_in_final` | Human + AI("p1") + Tool + AI("p2") + Tool + AI("FINAL") | only FINAL |
| 9 | `test_multiple_turns_each_takes_last_text` | 3 turns, each like #4 | 3 pairs, each shows the FINAL of its turn |
| 10 | `test_multiple_turns_chronological_order` | 3 turns | output order matches input order (verified by content match) |
| 11 | `test_trailing_user_no_ai_renders_no_response` | 2 complete turns + dangling Human | last block ends with `(no response yet)` |

### 4.2 Non-direct user sources (visible at all verbosity levels)

| # | Name | Setup | Expectation |
|---|---|---|---|
| 12 | `test_taskmanager_turn_visible_default` | Human("TaskManager:> do X") + AI("done") | header `--- TaskManager ---`, body has prefix stripped, AI shown |
| 13 | `test_taskmanager_turn_visible_text_mode` | same, verbosity=text | rendered |
| 14 | `test_taskmanager_turn_visible_full_mode` | same, verbosity=full | rendered |
| 15 | `test_taskmanager_no_response_renders_placeholder` | Human("TaskManager:> do X") only | `--- TaskManager ---` + `(no response yet)` |
| 16 | `test_agent_echo_turn_visible_default` | Human("agent:eva:> hi") + AI("ok") | header `--- From agent:eva ---`, body without prefix |
| 17 | `test_agent_echo_turn_visible_text_mode` | same, verbosity=text | rendered |
| 18 | `test_agent_echo_turn_visible_full_mode` | same, verbosity=full | rendered |
| 19 | `test_agent_echo_no_response_renders_placeholder` | Human("agent:eva:> hi") only | placeholder rendered |
| 20 | `test_mixed_sources_all_visible` | User + TaskManager + agent_echo + AI replies for each | all three turns appear in chronological order with correct headers |
| 21 | `test_taskmanager_and_user_never_filtered` | 4-turn fixture mixing real user and TaskManager | every non-summarization turn produces a header; counts match |

### 4.3 Summarization (the only filtered case)

| # | Name | Setup | Expectation |
|---|---|---|---|
| 22 | `test_summarization_human_filtered` | Human("Context Extraction Assistant ...") + AI(...) + Human(real) + AI("ok") | summarization turn dropped, only real turn shown |
| 23 | `test_summarization_legacy_marker_filtered` | Human("You are in the middle of a conversation that has been summarized…") + AI(...) | dropped |
| 24 | `test_summarization_only_history_returns_no_history` | only summarization Human | `"No conversation history yet."` |
| 25 | `test_summarization_does_not_swallow_following_turn` | summarization-Human + real-Human + AI | only the real turn appears, intact |

### 4.4 `verbosity=text`

| # | Name | Expectation |
|---|---|---|
| 26 | `test_text_mode_includes_all_text_ai_messages` | turn with 3 text AIs → all 3 appear, joined with blank line, FINAL last |
| 27 | `test_text_mode_still_skips_tool_only_ai` | tool-only AIs still skipped |
| 28 | `test_text_mode_no_response_renders_placeholder` | dangling Human still shows `(no response yet)` |
| 29 | `test_text_mode_does_not_emit_tool_calls` | a tool_calls-only AI is silent |

### 4.5 `verbosity=full`

| # | Name | Expectation |
|---|---|---|
| 30 | `test_full_mode_renders_tool_call_lines` | `--- Assistant (tool: execute) ---` line present |
| 31 | `test_full_mode_truncates_long_args` | args >200 chars truncated with `...` |
| 32 | `test_full_mode_preserves_chronological_order` | text and tool-call lines interleaved in original order |
| 33 | `test_full_mode_handles_unicode_args` | Cyrillic args not mangled (`ensure_ascii=False`) |
| 34 | `test_full_mode_no_response_still_renders_placeholder` | placeholder unchanged in full mode |

### 4.6 HTTP endpoint contract

| # | Name | Expectation |
|---|---|---|
| 35 | `test_endpoint_default_is_final` | `GET /api/agents/{name}/dialog-text` (no query) → behaves identically to `?verbosity=final` |
| 36 | `test_endpoint_accepts_text_verbosity` | `?verbosity=text` returns 200 |
| 37 | `test_endpoint_accepts_full_verbosity` | `?verbosity=full` returns 200 |
| 38 | `test_endpoint_rejects_unknown_verbosity` | `?verbosity=zzz` returns 422 |
| 39 | `test_endpoint_returns_plain_text_content_type` | response header `content-type: text/plain; charset=utf-8` |
| 40 | `test_endpoint_404_for_unknown_agent` | (regression) unknown agent → 404 |

### 4.7 No-limit history retrieval

| # | Name | Expectation |
|---|---|---|
| 41a | `test_handle_get_history_no_limit_returns_all` | `_handle_get_history({})` (no `limit` key) returns every message in the checkpoint |
| 41b | `test_handle_get_history_zero_limit_returns_all` | `limit=0` → all messages |
| 41c | `test_handle_get_history_negative_limit_returns_all` | `limit=-1` → all messages |
| 41d | `test_handle_get_history_positive_limit_still_works` | `limit=5` → last 5 messages (legacy contract preserved) |
| 41e | `test_dialog_endpoint_does_not_pass_limit` | the new service path calls RPyC `get_history` with `limit=None` (asserted via mock) |

### 4.8 Regression — the actual user-reported bug

| # | Name | Setup | Expectation |
|---|---|---|---|
| 41 | `test_regression_final_answer_visible` | replay the §1 fixture (TaskManager-Human + 5 text AIs after, including `"Confirmed: zero..."`) | rendered output **contains** `"Confirmed: zero"`; turn header is `--- TaskManager ---` |

### 4.8 Helper purity

| # | Name | Expectation |
|---|---|---|
| 42 | `test_classify_human_user` | plain text → `"user"` |
| 43 | `test_classify_human_taskmanager` | starts with `TaskManager:>` |
| 44 | `test_classify_human_agent_echo` | `agent:eva:> ...` |
| 45 | `test_classify_human_agent_echo_malformed_falls_to_user` | starts with `agent:` but no `:> ` → `"user"` |
| 46 | `test_classify_human_summarization_explicit` | starts with `"You are in the middle of a conversation that has been summarized"` |
| 47 | `test_classify_human_summarization_context_extraction` | contains `"Context Extraction Assistant"` in first 200 chars |
| 48 | `test_extract_text_from_string_content` | direct string |
| 49 | `test_extract_text_from_block_list` | mixed text + tool blocks |
| 50 | `test_extract_text_returns_empty_for_tool_only` | only `tool_use` blocks |

**Total: 55 unit tests** (50 formatter/contract + 5 no-limit tests).
All synchronous-callable for the pure formatter; endpoint tests need
`AsyncClient`; the no-limit tests need `AgentProcess` with an in-memory
checkpointer (no RPyC).

---

## 5. Implementation steps (TDD red → green)

1. Add `verbosity` parameter to `agent_schemas` (literal type or query
   model).
2. Write `tests/unit/test_dialog_formatting.py` with all 50 tests; confirm
   most fail (red).
3. Extract pure helpers into `app/services/dialog_formatter.py`:
   - `classify_human(content) -> str`
   - `extract_text(msg_dict) -> str`
   - `format_tool_call(tc) -> str`
   - `split_into_turns(history) -> list[Turn]`
   - `render_dialog(turns, verbosity) -> str`
4. Rewrite `AgentService.get_agent_dialog_formatted` to delegate to the
   pure renderer.
5. Add `verbosity: Literal["final","text","full"] = "final"` query param
   to `GET /api/agents/{agent_name}/dialog-text` in `agents.py`.
6. Run full suite — all 50 new + existing tests must pass (green).
7. Run pyright (project policy) — 0 new errors.
8. Update `USER_GUIDE.md`: document the three verbosity modes and the
   three user-side message sources.
9. Commit with TDD-style message body and reference to this design.

---

## 6. Resolved decisions (per user direction)

- **Verbosity in UI** — **IN SCOPE.** The Full Dialog click flow exposes
  a dropdown so the user can choose `final` / `text` / `full` without
  hand-editing URLs. See §3.6 UI for details.
- **JSON/HTML renderer** — **NOT NEEDED for this fix.** The current
  text-format rendering in a new browser tab stays as-is for all three
  text verbosities. The helpers in `dialog_formatter.py` are still
  factored as pure functions in case a JSON mode is added later.
- **Live updates** — unchanged. WebSocket streaming continues independently.
- **Pagination / size limits** — **REMOVED entirely.** Per explicit user
  direction, no artificial limits are allowed. The implicit limit comes
  from the agent's context window; we add no additional caps.
  Concretely: `AgentService.get_agent_dialog_formatted` no longer passes
  any `limit` to RPyC `get_history`. The server-side `_handle_get_history`
  treats absent / `None` / `0` / negative `limit` as "return everything"
  (this is the fix for the artificial 50-default and 1000-cap that
  previously truncated long histories).

---

## 7. Risk & rollback

- **Risk**: changing default behaviour might surprise scripted callers
  that parsed the previous (broken) format. Mitigation: the format still
  uses the same `--- User ---` / `--- Assistant ---` markers; only the
  *content* is now correct, plus two new header variants
  (`--- TaskManager ---`, `--- From agent:<name> ---`) for messages that
  were already present in the stream but rendered with the wrong header.
- **Rollback**: revert the commit. The pure helpers live in their own
  file and the endpoint signature changes are additive (`verbosity` query
  param is optional). No DB / RPyC changes.

---

## 8. Acceptance criteria

- [ ] All 55 tests in `test_dialog_formatting.py` pass.
- [ ] Existing `tests/test_full_dialog.py` Playwright tests still pass.
- [ ] `pyright` reports no new errors.
- [ ] **Live verification**: hitting `GET /api/agents/{name}/dialog-text`
      (default) on the running server shows the string
      `"Confirmed: zero occurrences"` somewhere in the body.
- [ ] The same live response shows a `--- TaskManager ---` header for the
      TaskManager-driven turn (turn `374` in the §1 fixture).
- [ ] `?verbosity=text` on a real agent shows intermediate "Let me also
      check…" lines.
- [ ] `?verbosity=full` on a real agent additionally shows tool-call
      summary lines.
- [ ] `?verbosity=zzz` returns HTTP 422.
- [ ] No `limit` query param is exposed; the endpoint always returns all
      messages.
- [ ] UI: Full Dialog area exposes a verbosity dropdown that persists
      its selection in `localStorage`.
- [ ] `USER_GUIDE.md` documents the verbosity modes, the three
      user-side message sources, and the no-limit guarantee.

---

## 9. 2026-06-23 follow-up — two introspection modes: `system_prompt` and `tools`

**Status:** ✅ IMPLEMENTED
**Request:** add two more entries to the Full Dialog verbosity combo (4th and
5th, after `full`) so the operator can inspect, on demand:

1. **`system_prompt`** — the *invariant (startup) system prompt* the agent was
   built with. This is the provider-AGNOSTIC role/instruction text produced by
   the CLI's `agent.get_system_prompt()` and passed to
   `create_deep_agent(system_prompt=...)`. It is the SAME text regardless of
   provider (claude-code-sdk / deepseek / openai / …). It deliberately does NOT
   include the per-provider tool surface — that is mode #5.
2. **`tools`** — the list of tools currently available to the agent. This MAY
   depend on the active provider (the tool surface is bound per provider /
   per session). Rendered as `name` + one-line description, one per tool.

### 9.1 Why these are NOT history turns

Modes 1–3 (`final`/`text`/`full`) render the conversation, which lives in the
checkpoint `messages` channel. The system prompt and tool list do NOT live in
the checkpoint:

- The system prompt is regenerated from agent configuration on every model
  request (LangChain factory prepends `request.system_message`); it is never
  persisted to `state["messages"]`. (See the cross-provider analysis in a
  related internal incident analysis (not included here) — the invariant
  base prompt is a function of agent
  config, not of stored history.)
- The tool inventory is computed at agent-build time
  (`AgentProcess._extract_final_tool_metadata`, which reads the compiled
  graph's `tools` node) and is provider-dependent.

Therefore these two modes bypass `get_history` entirely and call dedicated
RPyC methods, then format the result as plain text. They are NOT routed
through `split_into_turns` / `render_dialog`.

### 9.2 Wire contract

```
GET /api/agents/{name}/dialog-text?verbosity=system_prompt
GET /api/agents/{name}/dialog-text?verbosity=tools
```

Same `text/plain` response. `verbosity` Literal is widened to
`final | text | full | system_prompt | tools`; unknown values → 422.

### 9.3 Agent-side handlers (source of truth)

- `AgentProcess._handle_get_system_prompt()` →
  `{"success": True, "system_prompt": <str>, "provider": <str>, "model": <str>}`.
  The invariant text is captured at build time into `self._invariant_system_prompt`
  (set in `start()` right where the system prompt is resolved) so the reported
  value is EXACTLY what the agent was constructed with — not a re-derivation
  that could drift. Falls back to recomputing via `get_system_prompt()` if the
  attribute is missing (e.g. legacy process).
- `AgentProcess._handle_get_tools()` →
  `{"success": True, "provider": <str>, "tools": [{"name","description"}...]}`.
  Tool list is read from the compiled graph via the existing
  `_extract_final_tool_metadata(self.agent)` helper (single source of truth,
  same inventory the model actually binds). Each metadata entry is reduced to
  `name` + `description`; the JSON input schema is intentionally omitted from
  the plain-text view (too noisy) but the name/description pair is enough to
  answer "does the agent currently have tool X?".

Exposed over RPyC as `exposed_get_system_prompt` / `exposed_get_tools`
(callable as `root.get_system_prompt()` / `root.get_tools()`), each returning
a JSON-serialized string like the other `exposed_*` methods.

### 9.4 Web rendering

`AgentService.get_agent_dialog_formatted` branches BEFORE the history fetch:

- `verbosity == "system_prompt"` → call `root.get_system_prompt()`, render:

  ```
  --- System Prompt (invariant, provider=<p>, model=<m>) ---
  <full invariant system prompt text>
  ```

- `verbosity == "tools"` → call `root.get_tools()`, render:

  ```
  --- Available Tools (provider=<p>, count=<n>) ---
  - <name>: <description first line>
  - ...
  ```

Both reuse the same non-blocking connect gate, 30s timeout, and legacy-error
handling as the history path. A pure formatter pair
(`render_system_prompt`, `render_tools`) lives in `dialog_formatter.py` so it
is unit-testable without RPyC.

### 9.5 UI

`static/index.html` adds two `<option>`s to `#full-dialog-verbosity` after
`full`:

```html
<option value="system_prompt">System prompt (invariant)</option>
<option value="tools">Available tools</option>
```

`main.js` widens the accepted set in `getFullDialogVerbosity()` /
`restoreFullDialogVerbosity()` to include the two new values so the choice
persists and is forwarded on the button click.

### 9.6 Tests (TDD)

- `dialog_formatter`: `render_system_prompt` / `render_tools` pure-function
  tests (header shape, empty/edge cases, Cyrillic, missing description).
- agent-side: `_handle_get_system_prompt` returns the captured invariant and
  falls back when the attribute is absent; `_handle_get_tools` reduces
  metadata to name/description and tolerates a tool-less agent.
- web service: `get_agent_dialog_formatted(verbosity="system_prompt"|"tools")`
  calls the right RPyC method and renders via the pure formatter
  (RPyC root mocked).
- router: `verbosity=system_prompt` / `tools` accepted; `verbosity=zzz` → 422.
