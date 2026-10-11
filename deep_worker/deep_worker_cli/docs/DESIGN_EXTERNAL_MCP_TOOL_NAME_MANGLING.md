# DESIGN: External MCP tool-name mangling (provider-safe bijection)

**Status:** APPROVED
**Date:** 2026-06-24
**Related:** DESIGN_EXTERNAL_MCP_TOOLS.md (§3.3 MCPToolAdapter, §6 tool name glossary)

## 1. Problem (diagnosis, confirmed live)

External MCP tools have names with dots/slashes for namespacing
(e.g. the `chrome` server returns `browser.list_tabs`, `admin.allowlist_add`).
In deep-worker the adapter name was assembled as `<server>.<tool>` =
`chrome.browser.list_tabs` (dot notation, DESIGN_EXTERNAL_MCP_TOOLS §6).

**The MCP specification DOES ALLOW dots** (official spec 2025-11-25,
Tool Names section: `A-Za-z0-9_-.` are allowed, a valid name example —
`admin.tools.list`; SEP-986 finalized — dots and slashes for namespacing).
Real MCP servers use them, and we must support that.

**LLM providers do NOT accept dots** — that is their limitation on top of MCP
(an acknowledged divergence of the two standards, OpenAI dev-community
"Clarifying tool name rules for MCP and OpenAI APIs"):

| Provider class | Reaction to a dot in the tool name | Confirmation |
|---|---|---|
| OpenAI-compatible (deepseek/openai/gemini) | **HTTP 400**, the whole request is rejected: `Invalid 'tools[N].function.name': does not match '^[a-zA-Z0-9_-]+$'` | live repro on deepseek |
| claude-code-sdk CLI | silently mangles `.`→`_`: we register `chrome.browser.status`, the model calls `mcp__deepworker__chrome_browser_status` → our match against `proxy_tool_names` (with dots) does not fire → tool_call does NOT reach ToolNode → Future is not resolved → 120s in_progress → job disappears | live repro + py-spy + internal diagnostics |

The symptom for the user is the same on both classes: the external
MCP-tool call hangs and/or the agent crashes.

## 2. Goal

Keep **canonical MCP names with dots/slashes** as the source
of truth (wire call `call_tool`, UI/catalog, logs), and always feed the
LLM provider a **provider-safe** name `^[a-zA-Z0-9_-]+$`, with an
**exact bijection** mangled ↔ canonical. No heavily distorted names
(no `__DOT__`): only replace separators with a double underscore.

## 3. Mangling rule

```
mangle(name):
    replace EVERY '.' and '/' with '__' (double underscore)
```

- A name WITHOUT `.` and `/` does not change at all (`status` → `status`).
- `browser.list_tabs` → `browser__list_tabs`.
- `chrome.browser.status` → `chrome__browser__status`.
- `user-profile/update` → `user-profile__update`.

### 3.1 Bijection via map (option B, chosen by the user)

Mangling by itself is NOT injective (`a.b` and `a_b` → different sources,
but `a.b`→`a__b`, and `a__b`→`a__b` — a collision is possible only if the
source set already contains a name with a ready-made `__`). Therefore we keep
the bijection as an **explicit table**, not a computed one:

```
class MCPToolNameMangler:
    forward: dict[str, str]   # canonical -> mangled (only mangled ones)
    reverse: dict[str, str]   # mangled   -> canonical (only mangled ones)
    _all_mangled: set[str]    # ALL issued final names (mangled AND not)

    def add(canonical) -> mangled:
        m = mangle(canonical)
        if m in self._all_mangled:        # see §3.2 — single collision check
            raise MCPToolNameCollisionError(canonical, m)
        if m != canonical:                # name is mangled → into the map
            forward[canonical] = m
            reverse[m] = canonical
        # an unmangled name is NOT put into forward/reverse, but...
        self._all_mangled.add(m)          # ...it ALWAYS goes into the common set
        return m

    def to_mangled(canonical)  -> str:  return forward.get(canonical, canonical)
    def to_canonical(mangled)  -> str:  return reverse.get(mangled, mangled)
```

Exactly as requested: `mangled = forward.get(canonical, canonical)` —
unmangled names are used as-is and are not stored in the map.

### 3.2 Collisions = fail-loud (single `_all_mangled` check)

The final (issued) tool name is always `m = mangle(canonical)`,
regardless of whether it was mangled or not. Uniqueness is needed on exactly
this final name, because ToolNode routes by it
(`BaseTool.name`). Therefore we keep `_all_mangled` — the set of ALL already
issued final names — and in `add()` we do ONE check
`if m in self._all_mangled: raise`. This covers all three collision
scenarios (per a review comment):

1. **mangled ↔ mangled**: `a.b` and `a/b` both → `a__b`.
2. **mangled (added first) ↔ literal**: first `a.b`→`a__b` (in set),
   then the literal tool `a__b` (`m == canonical == 'a__b'`, already in set).
3. **literal (added first) ↔ mangled**: first the literal `a__b`
   (went into the set, although not into the map), then `a.b`→`a__b` (already in set).

In all cases — **`MCPToolNameCollisionError`** (fail-loud); the pool
logs and does NOT raise the conflicting server (as it already does when a server
fails to start, §3.4). No silent overwrites. Collisions in real MCP
are extremely unlikely (a double underscore in source names is rare), but
they are handled explicitly.

## 4. Where to mangle — ONE place (the adapter), zero changes in chat models

Key observation: **both ToolNode and all providers route the tool by
`BaseTool.name`**:
- OpenAI class: `bind_tools` serializes `tool.name` into the API; the
  returned tool_call comes with the same name → ToolNode looks up the BaseTool by it.
- CLI providers (claude-code-sdk/codex/grok): `bind_tools` puts `tool.name`
  into `ToolMetadata`; ProxyMCPServer registers the tool under that name; the
  stream match goes against `tools_metadata` names; `pending_calls[...]["name"]`
  (= the name from metadata) is surfaced into ToolNode.

This means that if the name of the **`MCPToolAdapter` itself (`BaseTool.name`) is made
mangled**, the same mangled name passes through end-to-end routing at
ALL providers without a single edit in their chat models. And the real wire
call to the MCP server goes by the unchanged `_bare_tool_name` (canonical
bare tool name, WITHOUT the server prefix) — it is already stored separately.

### 4.1 Change in `MCPToolAdapter.__init__`

Was (current):
```python
super().__init__(name=f"{server_name}.{tool_name}", ...)
object.__setattr__(self, "_bare_tool_name", tool_name)
```

Will be:
```python
canonical = f"{server_name}.{tool_name}"      # canonical (for UI/logs)
mangled   = mangler.add(canonical)            # provider-safe
super().__init__(name=mangled, ...)
object.__setattr__(self, "_bare_tool_name", tool_name)   # wire name UNCHANGED
object.__setattr__(self, "_canonical_name", canonical)   # for UI/catalog
```

- `BaseTool.name` = mangled → providers/ToolNode are happy.
- `_bare_tool_name` = `browser.list_tabs` (as before) → `call_tool` to the
  server goes with the CANONICAL name (MCP allows dots). ✓
- `_canonical_name` = `chrome.browser.list_tabs` → available for UI/catalog.

We extend `description` with the canonical name string, so that the model sees
the namespacing in a human-readable form (the mangled name in `name`, the canonical one
in the description). This is not required for operation, but improves the model UX.

### 4.2 Where the mangler is created

One `MCPToolNameMangler` per agent lives in `ExternalMCPClientPool`
(in the same place where adapters are collected from `list_tools()`). The pool:
- creates the mangler,
- passes it to each `MCPToolAdapter`,
- stores it as `pool.name_mangler` (read-only property) for future needs
  (UI/catalog, diagnostics, translating names in both directions).

`auto_approved_tool_names` (§3.4 of the base design) now contains
**mangled** names (since HITL/interrupt_on matches by `BaseTool.name`).
This is correct: both interrupt_on and middleware see the mangled name.

## 5. Why we do NOT touch provider chat models

The alternative (egress-mangle/ingress-demangle in each of the 4
providers) would require:
- claude-code-sdk: edits in `bind_tools` + `proxy_blocks` classification +
  surfacing `pending → tool_calls`;
- codex/grok: similar edits in their `chat_model` + proxy;
- openai class: a wrapper subclass over `ChatOpenAI`/`ChatGoogleGenerativeAI`
  intercepting `bind_tools` and `AIMessage.tool_calls`.

That is 4 risky places versus ONE in the adapter. Mangling at the
`BaseTool.name` level gives the same bijection (the map lives in the pool), but routing
"just works" everywhere, since the mangled name is the only name the
graph sees. The canon is preserved for wire (`_bare_tool_name`) and UI
(`_canonical_name` + map). This is exactly the user's option B
(explicit map, `get(x, x)`), implemented minimally invasively.

## 6. TDD plan

### Unit tests
1. `test_mangler.py`:
   - `mangle('status') == 'status'` (no mangling, not in the map).
   - `mangle('browser.list_tabs') == 'browser__list_tabs'`.
   - `mangle('a/b') == 'a__b'`.
   - `to_canonical(to_mangled(x)) == x` (bijection) for a set of names.
   - unmangled name: `to_mangled('status') == 'status'`, not in the map.
   - collision `a.b` + `a/b` → `MCPToolNameCollisionError`.
   - collision `a.b` + literal `a__b` → `MCPToolNameCollisionError`.
2. `test_adapter.py` (update existing):
   - `MCPToolAdapter.name == 'chrome__browser__list_tabs'` (mangled).
   - `_bare_tool_name == 'browser.list_tabs'` (canonical bare, unchanged).
   - `_canonical_name == 'chrome.browser.list_tabs'`.
   - `_arun` calls `session.call_tool('browser.list_tabs', ...)` (canonical).
3. Update `test_interrupt_on_policy.py`, `test_pool_stdio_integration.py`:
   assertions on mangled names in the catalog, canonical in the wire call.

### Integration (NO mocks, real providers)
4. `test_external_mcp_deepseek_live.py`: raise a deepseek agent with a real
   chrome MCP, invoke the tool, check that there is NO 400 and a result comes back.
5. `test_external_mcp_claude_code_live.py`: the same on claude-code-sdk, check
   that the tool_handler reaches AND the Future is resolved (no 120s in_progress).

## 7. Compatibility and risks

- The tool name visible to the model changes `chrome.browser.status` →
  `chrome__browser__status`. This is intentional (a provider requirement).
- Old sessions/checkpoints where a tool_call with the dotted
  name is saved in history: on resume, ToolNode will look up the tool by the dotted name, while the tool
  is now mangled → a possible "no such tool". Mitigation: external MCP tools are
  new functionality, and production histories practically have no
  dotted-name calls to them; DanglingToolCallMiddleware closes dangling tool_calls on restart.
  (If needed — an alias resolution via `to_canonical` can be added,
  but that is a separate step, not needed for the base fix.)
- Internal tools (`mcp__deepworker__*`, web_tools, etc.) are NOT affected:
  mangling applies ONLY to `MCPToolAdapter`.

## 8. What we do NOT do

- We do not forbid dots in MCP names (MCP allows them).
- We do not introduce heavily distorted encodings (`__DOT__` and the like).
- We do not edit provider chat models and do not bypass their name validation.
- We do not change the ProxyMCPServer contract (the handler still only parks
  the Future).
