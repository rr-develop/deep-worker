# Research: a unified MCP over LSP for deep-worker agents

**Date:** 2026-07-30  
**Branch:** `dev`  
**Status:** research only — deep-worker code **has not been changed**  
**Goal:** evaluate how to give agents IDE navigation (definition, references,
rename, hover, diagnostics, …) through a **single MCP interface** for
Python → Kotlin → other languages; optional, like a semantic index.

---

## 1. Problem statement

| Requirement | Meaning |
|-------------|---------|
| Unified MCP | One tool surface for all languages, not N different APIs |
| Precision | Not grep/semantic "similar", but the LSP symbol graph |
| Optionality | Like `.vector_index` / semantic search: enabled — works |
| Python first | pyright / basedpyright / pylsp |
| Kotlin second | JetBrains kotlin-lsp or fwcd |
| Open source | Can fork/extend |
| Agent UX | One call → definition/usages, no manual tree traversal |

**Do not confuse** with the providers' `ProxyMCPServer` (Claude/Grok/Codex):
that is a LangGraph→CLI bridge, the handler does **not** execute tools. Here
we mean an **external MCP server** that executes LSP requests.

---

## 2. How the MCP ↔ LSP "bridge" works

```
Agent (deep-worker)
    │  MCP tools/call  (stdio or HTTP)
    ▼
MCP server (Serena / mcpls / mcp-language-server / …)
    │  JSON-RPC LSP (usually stdio)
    ▼
Language Server(s): pyright, kotlin-lsp, gopls, rust-analyzer, …
    │
    ▼
Workspace (project root)
```

Critical details for agents:

1. **Addressing model.** "Raw" LSP is `file + line + character`.
   The agent is often more comfortable with **a symbol by name**
   (`find_symbol("Foo.bar")`), otherwise a preliminary read/grep is
   needed to find the position.
2. **LS lifecycle.** Workspace indexing (especially Java/Kotlin/C++)
   can take seconds to minutes; cold start ≠ "instant grep".
3. **Multi-root / multi-language.** Monorepo: several LS at once,
   routing by extension / languageId.
4. **Document sync.** Many LS require `didOpen` before requests;
   a good bridge does this transparently.
5. **UTF-16 vs UTF-8 positions.** LSP defaults to UTF-16 code units;
   the bridge must convert (mcpls explicitly writes about this).

---

## 3. Candidate overview (OSS)

### 3.1 Summary table

| Project | Stars (≈) | Language | Activity | Multi-LS | API level | License |
|---------|-----------|----------|----------|----------|-----------|---------|
| **[oraios/serena](https://github.com/oraios/serena)** | **~27k** | Python | 2026-07-30 | yes, 40+ languages | **agent-first** (symbols) | MIT |
| **[isaacphi/mcp-language-server](https://github.com/isaacphi/mcp-language-server)** | ~1.6k | Go | 2026-03 | 1 LS per process | position-based + edit | BSD-3 |
| **[bug-ops/mcpls](https://github.com/bug-ops/mcpls)** | ~55 | Rust | 2026-07-29 | yes (heuristics) | position-based, LSP 3.17 | MIT/Apache-2 |
| **[jonrad/lsp-mcp](https://github.com/jonrad/lsp-mcp)** | ~191 | TS | 2025-03 | multi (POC) | dynamic LSP methods | MIT |
| **[Tritlo/lsp-mcp](https://github.com/Tritlo/lsp-mcp)** | ~123 | TS | 2025-07 | 1 LS | hover/completion/diagnostics | MIT |
| **[nzrsky/lsp-mcp-server](https://github.com/nzrsky/lsp-mcp-server)** | ~11 | Zig | 2025-12 | config multi | hover/definition/completion | MIT |
| BumpyClock/lsp-mcp (Rust crates `lsp-mcp`) | ~16 | Rust | 2026-01 | multi (claimed) | unified access | — |
| quintenkasteel/lsp-mcp | — | — | **repo 404** | — | mentioned in reviews | — |

> **Note:** `quintenkasteel/lsp-mcp` as of 2026-07-30
> **was not found** as a public GitHub repository; on docs.rs/crates
> the name `lsp-mcp` appears (BumpyClock and others). Rely on the
> verifiable repositories above.

### 3.2 Serena (oraios) — the leader in maturity and agent UX

**Essence:** not a "bare LSP→MCP", but an **IDE toolkit for the agent**:
symbol search, file overview, references, rename, symbolic edit
(`replace_symbol_body`, insert before/after), diagnostics. The backend is
an abstraction over LS (and optionally the paid JetBrains plugin).

**Tools (LS backend, excerpt):**

| Tool | IDE equivalent |
|------|----------------|
| `find_symbol` | Workspace symbol search |
| `find_declaration` | Go to definition |
| `find_referencing_symbols` | Find all references / Show usages |
| `find_implementations` | Go to implementations |
| `get_symbols_overview` | Document outline |
| `rename_symbol` | Rename refactoring |
| `replace_symbol_body` / `insert_*_symbol` | Structural edit |
| `get_diagnostics_for_file` | Problems panel |
| `restart_language_server` | (optional) |

**Languages:** 40+, including **Python** (pyright default) and **Kotlin**
(JetBrains Kotlin Language Server, managed download). Multi-LS in
a single process for polyglot.

**Pros for deep-worker:**

- The agent thinks in **symbol names**, not line/col → fewer tool chains.
- Already MCP stdio/HTTP; installed via `uv tool install serena-agent`.
- Active development, huge community, agent evaluations (Opus/GPT)
  in the official evals.
- Open source MIT → can fork/limit the tool set.

**Cons / risks:**

- Heavy stack (uv, LS download, memory for Java/Kotlin LS).
- Its own "thick" surface (memory, file tools) overlaps with
  deep-worker's built-in tools — **need to disable the extras** via
  modes/config (Serena can do this).
- Dependence on upstream lifecycle; the "don't install from the
  marketplace" policy (they officially recommend uv).
- JetBrains backend — a separate paid path; for us the LS is primary.

### 3.3 mcpls (bug-ops) — the best "thin" multi-LSP bridge

**Essence:** universal MCP↔LSP, **one tool set** for any LSP 3.17.
Rust, prebuilt binaries, TOML config, heuristics by project markers
(starts only the needed LS).

**Tools (unified interface):**

- Intelligence: `get_hover`, `get_definition`, `get_references`,
  `get_completions`, `get_document_symbols`, `workspace_symbol_search`,
  `get_signature_help`, `go_to_implementation`, `go_to_type_definition`,
  `get_inlay_hints`
- Diagnostics: `get_diagnostics`, `get_cached_diagnostics`, `get_code_actions`
- Refactor: `rename_symbol`, `format_document`, call hierarchy
  (`prepare_call_hierarchy`, `get_incoming_calls`, `get_outgoing_calls`)

**Built-in heuristics:** Rust, Python (pyright), TS, Go, C/C++, Zig;
claimed to work with **any** LSP 3.17 (incl. jdtls; Kotlin via
`[[lsp_servers]]` + extension mapping `kt/kts → kotlin`).

**Pros:**

- Exactly what was asked: **one API, many languages**.
- Active (July 2026), docs, security note about `./mcpls.toml` trust.
- Lightweight runtime compared to Serena.
- Graceful degradation: one LS crashes — the rest stay alive.

**Cons:**

- **Position-based** API (needs line/col) — the agent will still often
  need `workspace_symbol_search` / `get_document_symbols` as the entry
  point (present in the API).
- Less community than Serena/isaacphi.
- Kotlin is not "zero-config" out of the box (needs an explicit LS `command`).

### 3.4 isaacphi/mcp-language-server

**Essence:** one process = one LS (`--lsp pyright-langserver -- --stdio`)
+ workspace. Go, mcp-go, ~1.6k stars.

**Tools:** `definition`, `references`, `diagnostics`, `hover`,
`rename_symbol`, `edit_file`.

**Pros:** proven, simple, good for **one** language per agent.  
**Cons:** multi-language = **several MCP server entries** (or several
instances) — not a "unified" multi-LS process; updates are rarer
(last push ~March 2026).

### 3.5 nzrsky/lsp-mcp-server (Zig)

Mentioned in the original review. Generic bridge, JSON config, `--server zls`.
Few stars, README is heavily "marketing" (a very broad packaging claim).
For production deep-worker — **low priority** (maturity/activity).

### 3.6 Tritlo / jonrad

- **Tritlo:** hover/completions/diagnostics/code_actions; **no**
  full go-to-definition/references as a first-class focus;
  mandatory `start_lsp`.
- **jonrad:** POC multi-LS, dynamic schema; stale, "POC" warning.

Useful as an API reference, not as a primary dependency.

### 3.7 Language servers (backend, not MCP)

| Priority | Language | Recommended LS | Note |
|----------|----------|----------------|------|
| 1 | Python | **pyright** / basedpyright | stdio: `pyright-langserver --stdio`; already in isaacphi/mcpls/Serena examples |
| 1 alt | Python | python-lsp-server (pylsp) | plugins, weaker type inference |
| 2 | Kotlin | **Kotlin/kotlin-lsp** (JetBrains, ~3.4k★) | used by Serena; JVM, memory |
| 2 alt | Kotlin | fwcd/kotlin-language-server (~2k★) | community, less "official" |
| 3 | Go | gopls | mature |
| 3 | TS/JS | typescript-language-server | mature |
| 3 | Rust | rust-analyzer | mature |
| 3 | Java | jdtls | slow cold start |
| 3 | C/C++ | clangd | needs compile_commands.json |

---

## 4. Fit against the target tool surface

The desired agent minimum (unified across languages):

| Capability | LSP method | Serena | mcpls | isaacphi |
|------------|------------|--------|-------|----------|
| Go to definition | textDocument/definition | `find_declaration` | `get_definition` | `definition` |
| Show usages | textDocument/references | `find_referencing_symbols` | `get_references` | `references` |
| Find symbol by name | workspace/symbol | `find_symbol` | `workspace_symbol_search` | — (no direct) |
| Outline file | textDocument/documentSymbol | `get_symbols_overview` | `get_document_symbols` | — |
| Hover / type | textDocument/hover | (via find/overview) | `get_hover` | `hover` |
| Implementations | textDocument/implementation | `find_implementations` | `go_to_implementation` | — |
| Rename | textDocument/rename | `rename_symbol` | `rename_symbol` | `rename_symbol` |
| Diagnostics | textDocument/publishDiagnostics | `get_diagnostics_*` | `get_diagnostics` | `diagnostics` |
| Call hierarchy | callHierarchy/* | — | yes | — |

**UX conclusion:** Serena is closer to "one request — what the agent needs".
mcpls is closer to "full LSP 3.17 as is". Both give a unified interface
over different LS.

---

## 5. How this fits deep-worker

### 5.1 Already there: External MCP

Implemented (see `deep_worker/deep_worker_cli/docs/DESIGN_EXTERNAL_MCP_TOOLS.md`
and the `deep_worker_cli/external_mcp/` package):

- Layers: `<state_dir>/mcp.json` → project `.deep_worker/mcp.json` →
  per-agent → CLI.
- `ExternalMCPClientPool` starts stdio/HTTP MCP, gives `BaseTool` to
  `create_cli_agent(..., external_mcp_tools=...)`.
- Name mangling, HITL, `auto_approve` per server.

**Consequence:** the first LSP experiment does **not** need a new
framework — an entry in `mcp.json` is enough, e.g.:

```json
{
  "mcpServers": {
    "serena": {
      "command": "serena",
      "args": ["start-mcp-server", "--transport", "stdio"],
      "cwd": "${workspace}",
      "auto_approve": true
    }
  }
}
```

or:

```json
{
  "mcpServers": {
    "mcpls": {
      "command": "mcpls",
      "args": [],
      "env": {
        "MCPLS_CONFIG": "/path/to/mcpls.toml"
      },
      "auto_approve": true
    }
  }
}
```

(the concrete args/cwd — to be refined in the PoC; the workspace root
must match the agent's `project_memory_dir`.)

### 5.2 Analogy with semantic search (optionality)

| Aspect | Semantic search | LSP (proposal) |
|--------|-----------------|----------------|
| Role | "similar" code / NL query | exact symbols / graph |
| Enabling | `.vector_index/` index, optional deps | mcp.json + installed LS |
| Package in repo | semantic search module | `lsp/` (currently research; later design/code) |
| Tool path | LangChain `@tool` in `tools/` **or** MCP | preferably **External MCP** |
| Weight | embedding model | LS processes (CPU/RAM) |

**Complement, not replace:** semantic — "roughly where auth is";
LSP — "where `AuthMiddleware.authenticate` is declared and who calls it".

### 5.3 What not to do

- Do not embed LS inside a provider's ProxyMCPServer.
- Do not add threads beyond the AgentProcess lifecycle (the pool is already per-agent).
- Do not change architecture/contracts without a separate design and approval.
- Do not enable write-refactor tools (`rename`, edit body) without a HITL policy
  (by default external MCP is already gated if there is no `auto_approve`).

### 5.4 Integration risks

| Risk | Mitigation |
|------|------------|
| LS cold start (Kotlin/Java) | ~~lazy start; prompt: "there may be a delay on the first request"~~ — **the assumption was disproved by measurement**, see design §7.1: the LS starts together with MCP, the cost is paid at agent start |
| RAM (several LS) | heuristics / only python+kotlin in the config |
| Tool name overlap | mangler `server.tool`; disable Serena basic file tools |
| Wrong workspace root | cwd = agent's project_memory_dir; explicit root in args |
| UTF-16 offsets | choose a bridge with conversion (mcpls) or agent-first (Serena) |
| Write tools break code | HITL; or read-only tools mode |
| No-polling policy | OK: event-driven JSON-RPC; no busy-wait |

---

## 6. Recommendations

### 6.1 Strategy (preferred)

**Phase A — PoC without our own code (recommended immediately):**

1. Install **Serena** (Python first) and/or **mcpls** (thin multi-LS).
2. Connect through the existing **External MCP** (`mcp.json` project/agent).
3. Disable Serena's duplicate file/shell tools (modes), keep
   symbol/navigation/refactor.
4. Compare on real deep-worker tasks (refactor rename, find
   usages across packages, Kotlin module later).

**Phase B — the "like semantic index" option policy:**

1. Document: "LSP navigation — optional capability".
2. Per-project/per-agent enable (mcp.json already).
3. Prompt fragment: when to call `find_symbol` / `get_definition` instead of
   `grep` (for default_agent_prompt or the project AGENTS.md).
4. Limit auto_approve: read-only tools vs rename.

**Phase C — our own MCP (only if A/B is not enough):**

Write our own bridge if we need:

- a rigid unified schema baked into deep-worker;
- a symbol-first API without dependence on Serena;
- built-in lifecycle (restart LS, per-worktree root) and CI tests;
- a minimal binary without uv/node.

Base for fork/inspiration: **mcpls** (thin, multi-LS, Rust) +
the agent-first layer of ideas from **Serena** (find by name, overview).
**Don't** start from scratch with a JSON-RPC LSP client if vendoring/fork is possible.

### 6.2 Product ranking

| Place | Product | When |
|-------|---------|------|
| **1** | **Serena** | default for agents: best UX, Python+Kotlin, maturity |
| **2** | **mcpls** | need a thin multi-LSP, full 3.17, less "IDE suite" |
| **3** | **isaacphi** | one language per agent, simplicity, Go binary |
| — | nzrsky / Tritlo / jonrad | not primary; reference only |
| — | own MCP | after the PoC, if the gaps are critical |

### 6.3 Unified interface (target contract for a future design)

Regardless of the backend, the agent should see stable names:

```
lsp_find_symbol(name, relative_path?)
lsp_definition(symbol | file+line+col)
lsp_references(symbol | file+line+col)
lsp_implementations(...)
lsp_hover(...)
lsp_document_symbols(path)
lsp_diagnostics(path?)
lsp_rename(...)   # gated
```

An adapter implementation can map:

- Serena tools → this contract, or
- mcpls tools → this contract,

and live either as a thin MCP wrapper, or as LangChain tools in
`tools/lsp_*.py` (like `tools/semantic_search.py`).

---

## 7. Relationship to the codebase (integration points)

| Component | Path | Role |
|-----------|------|------|
| External MCP pool | `deep_worker_cli/external_mcp/` | **main** integration point for OSS MCP |
| Config layers | `deep_worker_cli/external_mcp/config.py` | mcp.json merge |
| Agent factory | `deep_worker_cli/agent.py` `external_mcp_tools=` | tools into the graph |
| Design | `deep_worker_cli/docs/DESIGN_EXTERNAL_MCP_TOOLS.md` | contract |
| Semantic (option analog) | `tools/semantic_search.py` | optional capability pattern |
| Provider ProxyMCP | `*_provider/proxy_mcp_server.py` | **do not** touch for LSP |

---

## 8. Open questions (to agree before design/implement)

1. **Default backend:** Serena vs mcpls vs dual (Serena primary)?
2. **Write tools** (rename/edit body): auto_approve or always HITL?
3. **Kotlin:** official kotlin-lsp (JVM weight) vs fwcd?
4. **Own wrapper** over OSS for stable tool names — needed right away,
   or are native Serena/mcpls names + prompt enough?
5. **Per-worktree:** one LS pool per AgentProcess (cwd) — enough?
6. **CI:** integration tests with pyright on a fixture repo — in scope?

---

## 9. Sources

- https://github.com/oraios/serena  
- https://oraios.github.io/serena/  
- https://github.com/bug-ops/mcpls  
- https://github.com/isaacphi/mcp-language-server  
- https://github.com/nzrsky/lsp-mcp-server  
- https://github.com/Tritlo/lsp-mcp  
- https://github.com/jonrad/lsp-mcp  
- https://github.com/Kotlin/kotlin-lsp  
- https://github.com/fwcd/kotlin-language-server  
- https://microsoft.github.io/language-server-protocol/  
- Internal: `DESIGN_EXTERNAL_MCP_TOOLS.md`, a related internal design doc for semantic search (not included here)

---

## 10. One-phrase conclusion

**A ready open-source path exists:** connect **Serena** (agent-first,
Python+Kotlin+40) or **mcpls** (a unified thin multi-LSP) through
deep-worker's already existing **External MCP**; our own MCP makes sense
only as a thin unification/policy after a PoC. No application code was
changed in this research; artifacts are in `lsp/`.
