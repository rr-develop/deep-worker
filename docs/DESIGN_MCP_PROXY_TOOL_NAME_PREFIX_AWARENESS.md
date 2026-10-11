# DESIGN — provider-dependent hints for calling base tools through the MCP proxy

Related incident: a related internal incident analysis (not included here).

## 1. Problem

The system prompt and dynamic hints refer to base deep-worker tools by their
**short** names (`send_message_to_agent`, `restart_self`, …), whereas the
actual invocation channel is the MCP proxy, and the name form **depends on the
provider**:

| Provider | proxy server name | How the model SHOULD call our tool | Does the short (native) name work? |
|-----------|-------------------|----------------------------------|-----------------------------------|
| claude-code-sdk | `deepworker` (static) | `mcp__deepworker__<tool>` | unstable (breaks after restart/resume) |
| codex | `deepworker_<pid>` (dynamic) | `mcp__deepworker_<pid>__<tool>` | NO (short names are reserved for Codex's native tools) |
| grok | `deep_worker_proxy_<pid>` (dynamic) | by the tool name directly — the grok CLI itself resolves it as an MCP tool on our server (the `CallMcpTool` mechanism); **no string prefix** | a bare name = grok's native built-ins; our tool is invoked as an MCP tool, not a native one |

Result: claude-code-sdk/codex agents get
`No such tool available: send_message_to_agent` and cannot reply to
inter-agent messages / restart. grok has a separate problem: not recognizing
the tool, the model goes looking for its schema on disk (`_build_tool_contract`,
lines 88-97) instead of calling it directly.

## 2. The essence of the fix (we do NOT touch tool mechanics — only prompts)

The goal is to make **system prompts honestly reflect the specifics of each
provider**. No new public methods on models, no changes to MCP tool name
resolution. We change ONLY the text the model sees.

### Goals
1. Inform the model: the user and other agents may refer to tools by their
   short names; for the current provider the short name may NOT resolve —
   in that case use the provider-specific MCP form.
2. Give the correct form for each of the three CLI providers
   (claude-code-sdk/codex/grok), including the dynamic PID for codex.
3. Fix the dynamic hint of `InterAgentReplyMiddleware` (it dictates how to
   answer an incoming `send_message_to_agent`) — also provider-dependent.

### Non-goals
- Change the MCP proxy mechanism / name resolution inside the CLI providers.
- Touch tool names in `ToolMetadata` — they remain bare names there.
- Introduce new API methods on models (`proxy_tool_invocation_*` and the like) —
  rejected by the user.

## 3. Prompt texts per provider

For the three "command" tools — `send_message_to_agent`, `restart_self`,
`restart_agent` — we give a DETAILED provider-dependent form. Plus ONE general
rule for the other base tools (§3.4).

### 3.1 claude-code-sdk
```
Base deep-worker tools are available through the MCP server `deepworker`. If
you refer to a tool by its short name `<X>` and get "No such tool available",
call its MCP version: `mcp__deepworker__<X>`. Example:
`mcp__deepworker__send_message_to_agent`.
```

### 3.2 codex
```
Short names are reserved for Codex's native tools. Call base deep-worker tools
ONLY through the MCP server: `mcp__deepworker_<pid>__<X>`,
where `<pid>` is the PID of the current agent process. Example:
`mcp__deepworker_<pid>__send_message_to_agent`.
```
Source of `<pid>`: the same process that holds the proxy — `os.getpid()`; or
the ready-made server name `self._proxy_mcp.server_name` (== `deepworker_<pid>`),
from which the prefix is assembled as `mcp__{server_name}__`.

### 3.3 grok  (RC-B: strengthened with the disambiguation "MCP tool, not a native built-in")
```
Base deep-worker tools (`send_message_to_agent`, `restart_self`, … and all
file/shell/web tools) are deep-worker MCP tools, NOT grok's built-in tools
(Shell/Read/Write/Edit/Glob/Grep/WebSearch/Task). Call them by name directly:
the grok CLI routes the call to our MCP server
`deep_worker_proxy_<pid>` (via its CallMcpTool mechanism) automatically.
Do NOT search for their schema on disk (no Glob/Grep/Read over
`mcps/…/tools/*.json`) and do NOT consider them missing.
```
(grok has no string prefix; the "tool name" here is the inner `toolName`
inside grok's `CallMcpTool`, which fills in the `server` itself. This is NOT
the bare name of a grok native built-in — our tools are available EXCLUSIVELY
through the MCP channel.)

### 3.4 General rule (claude/codex) for the other base tools
```
The same rule applies to all base deep-worker tools (read_file,
execute, ls, …): if the short name does not resolve — use the
provider-specific MCP form (see above).
```
grok needs no separate section — the wording of §3.3 is already general.

## 4. Points of application (where short names are currently hard-coded)

1. **`default_agent_prompt.md`** — static template (section
   `### send_message_to_agent`, ~lines 81-115; the `restart_*` sections). The
   names here remain short (the template is provider-agnostic); the provider
   specifics are carried by the dynamic note (item 2) the template refers to.
   **RC-C (mandatory).** A single reference line is added to the
   `### send_message_to_agent` section: "The CALL form of this tool for your
   provider is in the "Your Agent Identity" section below (for some providers
   the short name does not resolve and the MCP form is required)". This way the
   static template explicitly redirects to the variable part instead of leaving
   the short name without a caveat.
2. **`agent.py` → `get_system_prompt`**, the "Your Agent Identity" block
   (~lines 177-193). A PROVIDER-DEPENDENT note (§3), selected by
   `settings.model_provider`, is added here. This is the only place with a
   variable part (prefix / PID / grok wording). For providers without an MCP
   proxy (deepseek/openai/anthropic/google) no note is added — their tools are
   native, the short name is the working one.

   **RC-A (mandatory).** In the current `agent_identity_section` (~lines 183-191)
   the call examples with the short name
   `send_message_to_agent(<peer>, …)` are hard-coded. If we merely add the
   provider-note NEXT to it, the model may follow the first (short) instruction
   and ignore the note. Therefore **the call examples in the Identity block are
   parameterized** via `proxy_tool_form(provider, "send_message_to_agent")`
   (and similarly for `restart_self`/`restart_agent`, if mentioned there):
   - claude-code-sdk → the example shows `mcp__deepworker__send_message_to_agent(...)`;
   - codex → `mcp__deepworker_<pid>__send_message_to_agent(...)`;
   - grok → the tool name as-is + the §3.3 disambiguation (MCP tool, not a native built-in);
   - providers without proxy → the short name (as now).
   Thus the short name no longer stays as the "call name" where it does not
   resolve. Additionally, the provider-note opens with a precedence phrase:
   "for CALLING on this provider use the forms below; short names in the
   template denote the logical identity of the tool, not its call name".
3. **`middleware/inter_agent_reply.py`** (`InterAgentReplyMiddleware`,
   lines 154-171). The hint `send_message_to_agent("{sender}", …)` is formed
   provider-dependently:
   - claude-code-sdk → `mcp__deepworker__send_message_to_agent("{sender}", …)`;
   - codex → `mcp__deepworker_<pid>__send_message_to_agent("{sender}", …)`;
   - grok → the §3.3 wording: call the tool `send_message_to_agent` directly as
     a deep-worker MCP tool (NOT a grok native built-in), the CLI will route it
     to the server `deep_worker_proxy_<pid>`; do not search for the schema on
     disk. (The middleware hint is the critical moment of answering a peer
     message, so the short phrase "as MCP tool, do not search disk" is added
     directly into it.)
   - others → as now, the short name.
   The middleware receives the form through the constructor:
   `InterAgentReplyMiddleware(name, *, reply_tool_form="send_message_to_agent")`.
   In `agent.py` (~line 898), where `model`/provider is available, the form is
   computed by the same provider dispatcher (§5) and passed to the constructor.
   The `reply_tool_form` default stays short — backward compatibility with the
   existing unit tests.

## 5. Provider dispatcher (one pure function, no methods on models)

All provider specifics are collected in a single string helper in
the CLI package (e.g. `prompts/proxy_tool_hints.py`):

```python
import os

def proxy_tool_form(provider: str, bare_name: str, *, pid: int | None = None) -> str:
    """The call form of a base tool for the given provider —
    a string ready to be inserted into a hint."""
    if provider == "claude-code-sdk":
        return f"mcp__deepworker__{bare_name}"
    if provider == "codex":
        p = pid if pid is not None else os.getpid()
        return f"mcp__deepworker_{p}__{bare_name}"
    if provider == "grok":
        return bare_name      # the CLI resolves it as an MCP tool; no string prefix
    return bare_name          # deepseek/openai/… — native tools
```

And a second helper — the FULL text of the provider note for the prompt:
claude/codex → "if the short name does not resolve → the MCP form (§3.1/§3.2)"
with explicit forms for the three command tools + the general rule §3.4; grok →
the §3.3 wording; deepseek/openai/… → an empty string. Pure string
functions — trivially testable, never fail.

## 6. TDD plan

### Unit (CLI package)
- `proxy_tool_form`:
  - `claude-code-sdk` → `mcp__deepworker__send_message_to_agent`;
  - `codex` (pid=1234) → `mcp__deepworker_1234__send_message_to_agent`;
  - `grok` → `send_message_to_agent` (bare, without `mcp__`);
  - `deepseek`/unknown → `send_message_to_agent` (fallback).
- The note text for the prompt:
  - claude-code-sdk: contains `mcp__deepworker__` and all three command tools;
  - codex: contains `mcp__deepworker_<pid>__` (the real pid) and the wording
    "reserved for Codex's native tools";
  - grok: contains "call by name directly" and "do not search for their schema
    on disk", does NOT contain `mcp__`;
  - deepseek: an empty string.
- `get_system_prompt` (via `settings.model_provider`): with claude — the
  Identity section contains `mcp__deepworker__send_message_to_agent`; with
  grok — the §3.3 text; with deepseek — no provider note.
- `InterAgentReplyMiddleware(reply_tool_form=…)`: the hint contains the passed
  form; the default stays `send_message_to_agent` (backward compatibility).

### Integration (no mocks)
- Extend
  `tests/integration_ai/test_claude_code_inter_agent_send_and_reply.py`:
  bring up a REAL claude-code-sdk agent, send an incoming inter-agent message, and
  verify that the system prompt/hint features
  `mcp__deepworker__send_message_to_agent` (not the bare name), and that the
  agent replies live without `No such tool available`.
- **RC-D (strongly recommended).** Similarly extend the existing
  `tests/integration_ai/test_codex_inter_agent_send_and_reply.py`: verify that
  the system prompt/hint of the codex agent features
  `mcp__deepworker_<pid>__send_message_to_agent` (the real pid of the process).

## 7. Compatibility and risks

- All changes are string/additive; tool contracts and names in
  `ToolMetadata` are unchanged → architecture and API are unaffected.
- Providers without an MCP proxy and mock models degrade to the short name
  (fallback) → existing tests do not break.
- The codex PID is taken from the live process (`os.getpid()` / `server_name`),
  known at prompt-build and middleware-run time — correct.
- grok: the §3.3 wording removes both the "No such tool" problem and the
  problem of the model going to search for the tool schema on disk.

## 8. Implementation element (explicitly pinned)

The comment in `grok_provider/chat_model.py:_build_tool_contract` (lines 84-102),
asserting that "grok exposes deep-worker MCP tools under BARE names without a
prefix", is OUTDATED/INCORRECT and contradicts the actual dispatch via
`CallMcpTool` (`acp_types.is_proxy_mcp_dispatch`, `chat_model.py:912`). Its
wording should be brought in line with §3.3 as part of the implementation
(grok sees our tools as MCP tools, calls them by name, the CLI routes them to
the server `deep_worker_proxy_<pid>`).
