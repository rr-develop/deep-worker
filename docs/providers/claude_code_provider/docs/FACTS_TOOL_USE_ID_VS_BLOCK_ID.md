# FACTS: `tool_use_id` vs "block id" — measured, not assumed

**Status:** reference (ground truth). Measured 2026-06-16 over the real
Claude CLI session transcripts on this host.

This document exists to kill a recurring false belief that has already
caused one deadlock (mj1_init, 2026-06-15) and risks future regressions:
the idea that a "block id" and a tool call's `tool_use_id` are the same
thing, or that one `tool_use_id` can be derived from / equated to the id
of the message block that contains it. **They are different namespaces,
and one block contains many tool calls.**

## The three ids — do not conflate them

A Claude assistant turn, as recorded in the CLI session `.jsonl`, has
THREE distinct id namespaces:

| Name in data | Prefix | What it identifies |
|---|---|---|
| `message.id` | `msg_*` | the **assistant message block** (one model response) |
| `content[].id` on a `tool_use` block | `toolu_*` | **ONE tool call** — this IS the `tool_use_id` |
| `_meta.claudecode/toolUseId` (MCP `tools/call`) | `toolu_*` | the **same tool call's** `tool_use_id`, carried over the MCP control channel |

So "the tool_use block's id" (`content[].id`, `toolu_*`) **is** the
`tool_use_id`. The "message block id" (`message.id`, `msg_*`) is a
DIFFERENT thing entirely.

The dangerous ambiguity is the word **"block"**: in the API a `tool_use`
*content block* has a `toolu_*` id (= the tool_use_id); in the session
file the *message block* has a `msg_*` id. Anyone who writes "block id"
without saying which one invites the confusion this document corrects.

### "block" means TWO different things — pin them down

There is no third "tool block" id in the data. There are exactly two
container levels:

| "block" | What it is | id | # of tool calls |
|---|---|---|---|
| **message block** | one assistant response | `message.id` = `msg_*` | MANY (0..52) |
| **`ToolUseBlock`** (a `tool_use` *content* element) | exactly **ONE tool call** | `.id` = `content[].id` = `toolu_*` | exactly 1 |

The SDK's `ToolUseBlock` is parsed from a single `tool_use` content
element (`message_parser.py`: `ToolUseBlock(id=block["id"], …)`). So
**`ToolUseBlock` IS one tool call**, and `ToolUseBlock.id` **is** that
call's `tool_use_id` (`toolu_*`) — NOT a container of several calls and
NOT the `msg_*` message id.

Therefore, when code/docs say `ToolUseBlock.id == _meta.claudecode/
toolUseId`, that is TRUE — it is *one call's* `toolu_*` appearing on two
transports (visible stream vs MCP control channel). It does **not**
contradict "`msg_* != toolu_*`" (the message block id vs a call id),
which is the false equality this document kills. Keep the two statements
separate:

* TRUE : `ToolUseBlock.id` (a `toolu_*`) == `_meta.claudecode/toolUseId`
  (the same `toolu_*`) — same single call, two transports.
* FALSE: `message.id` (`msg_*`) == any `tool_use_id` (`toolu_*`) — never
  (0/46683), and one `msg_*` owns many `toolu_*`.

## Measured facts (1796 session files, 69 168 assistant messages)

Script: group every `tool_use` block's `content[].id` by its enclosing
`message.id`, within each session file.

```
assistant messages with >=1 tool_use:           46 683 (unique message.id)
messages where message.id == some tool_use_id:        0     ← NEVER equal
```

* **`message.id` (`msg_*`) is NEVER equal to any `tool_use_id`
  (`toolu_*`).** 0 out of 46 683. They are different namespaces and
  cannot be equal.

* **One message block routinely contains MANY tool calls, each with its
  own unique `tool_use_id`.** Distribution of tool calls per
  `message.id`:

  ```
  1 call : 45 457
  2 calls:    881
  3 calls:    215
  4 calls:     54
  5 calls:     25
  6 calls:     24
  …
  up to 52 calls in a single message block
  ```

  1 226 message blocks carried **more than one** tool call. In none of
  them did `message.id` equal any contained `tool_use_id`.

## What this means (the invariant)

* The identity of a tool call is **its `tool_use_id` and nothing else**.
* A message block is NOT a tool call. A block is "done" only when **all**
  of its `tool_use_id`s have been resolved (it can hold up to dozens).
* Never key a wait/pending/result on a message/block id, and never assume
  "one block = one call". Always correlate by the specific
  `tool_use_id`.

## The SAME `toolu_*` of ONE call appears on two transports

For a SINGLE tool call, the same `toolu_*` value appears in two places:
the visible-stream `ToolUseBlock.id` AND the MCP control-channel
`_meta.claudecode/toolUseId`. That is one id for one call on two
transports — equal **for that one call**. This is a different statement
from "message-block id == tool_use_id" (which is false). When code/docs
say "`ToolUseBlock.id` == `_meta.claudecode/toolUseId`", they mean the
per-call `toolu_*` on two transports — never read it as "msg_* ==
toolu_*".

> ⚠️ This per-call equality applies ONLY within one and the same call.
> Two DIFFERENT `toolu_*` values are simply two DIFFERENT calls — NOT one
> call that "got two ids". Do not use the two-transports idea to explain
> away two different ids (that was an analysis error in the mj1
> investigation — corrected below).

### The actual mj1_init deadlock (2026-06-15) — TWO different calls

The mj1 hang was **not** one call with two ids. It was two distinct tool
calls, measured from the raw transcript
(`955e5e68-…jsonl`):

| `tool_use_id` | `name` | `tool_result` |
|---|---|---|
| `toolu_01BHB9…` | `execute` (Claude **built-in**) | `is_error:true` — *"No such tool available: execute"* |
| `toolu_011Um…` | `mcp__deepworker__execute` (**our** proxy tool) | none — this is the call that hung |

Sequence: the model first called the **built-in** `execute` (`01BHB9`);
the CLI rejected it ("no such tool"); the model then re-issued the call
correctly as `mcp__deepworker__execute` (`011Um`), which reached our
control channel and hung.

The deadlock: `chat_model` saw the first visible `tool_use` block
(`01BHB9`, name `execute`) and **misclassified the built-in `execute` as
our proxy tool** — because classification strips the
`mcp__deepworker__` prefix and then matches on the bare base name
`execute`, which collides with our proxy tool's base name. It set
`expected_proxy_tool_call_id = 01BHB9` and waited
`wait_for_pending_call_by_id(01BHB9)`. But `01BHB9` is a built-in call —
it NEVER comes over our MCP control channel. Our real call `011Um`
arrived and registered in `pending_calls`, but `chat_model` was parked
waiting for `01BHB9`. Live proof from the running proxy:

```
pending_calls:        ['toolu_011Um…']   ← our mcp__deepworker__execute (control channel)
_pending_id_waiters:  ['toolu_01BHB9…']  ← built-in execute that chat_model wrongly waits for
```

Correct rule: identify a call ONLY by the `tool_use_id` that arrives on
our MCP control channel for a tool whose name actually starts with
`mcp__deepworker__`; never treat a bare-name match (`execute`) as one of
ours; and treat the message as complete only when every such call is
resolved.

## Fix implemented (2026-06-16)

**Tools are visible to the model ONLY under their long MCP names.**
Claude Code exposes any MCP server's tools as `mcp__<server>__<tool>`
(here `mcp__deepworker__<tool>`) — there is no way to present them under a
bare/short name (verified against the Claude Code MCP docs). A past
attempt to "expose short names" (commit `e89cef6`: a system-prompt note
telling the model to use bare names + a bare-name short-circuit) was
**already reverted** (`790f3e1`) because bare-name calls are silently
dropped by the CLI (they never route to our MCP server). We do **not**
nudge the model toward short names; the canonical call is the long name.

**Prefix-only classification (the structural fix).** `chat_model.ainvoke`
now classifies a `ToolUseBlock` as one of ours **only if**
`block.name.startswith("mcp__deepworker__")` AND the stripped name is a
registered proxy tool. A bare block (e.g. a built-in `execute`, or a
model hallucination of a short name) is **not** classified as ours, so we
never capture its `tool_use_id` and never park on a wait that can't
complete. This is the structural cure for the mj1_init deadlock — not a
timeout. (Earlier `#126` band-aids — bare-twin dedup, replay-dedup — were
reverted; this is the root fix.)

**Safeguard against the wrong-id wait (defence in depth).**
`proxy_mcp_server.parse_no_such_tool(text)` detects the CLI's
`No such tool available: <name>` rejection, and
`ProxyMCPServer.cancel_pending_waiter(tool_use_id)` cancels an id-keyed
waiter so a turn can't park forever if a wait was somehow started for a
call that won't route. A tool not registered on the LangGraph side never
enters the wait in the first place (the classification gate above).

Tests: `test_event_based_tool_detection.py::TestBareNameNotMisclassified
AsProxy` (bare name not waited on; bare-then-prefixed waits only on the
prefixed id), `test_no_such_tool_safeguard.py` (parse + cancel
primitives).

## Reproduce the statistics

```python
import json, glob, collections
files = glob.glob("/home/<user>/.claude/projects/*/*.jsonl")
msg_to_tools = {}
for f in files:
    for line in open(f, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") != "assistant": continue
        m = o.get("message", {})
        mid = m.get("id")
        if not (isinstance(mid, str) and mid.startswith("msg_")): continue
        for b in m.get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                msg_to_tools.setdefault((f, mid), set()).add(b["id"])
dist = collections.Counter(len(v) for v in msg_to_tools.values())
eq = sum(1 for (f, mid), tids in msg_to_tools.items() if mid in tids)
print("unique (file, message.id) with tools:", len(msg_to_tools))
print("tool calls per message.id:", dict(sorted(dist.items())))
print("message.id == some tool_use_id:", eq)   # 0
```

## Related (annotated, not deleted)

* a related internal incident analysis (not included here) — the
  per-call two-transport equality is correct there; see the annotation
  added pointing back to this document so "block id" is never read as
  `message.id`.
* `DESIGN_restore_proxy_architecture.md` — early design that used
  `ToolUseBlock.id` as the pending-calls key; annotated.
* `chat_model.py` / `proxy_mcp_server.py` — the wait-by-id path; comments
  annotated to reference this document and the multi-call/base-name trap.
