# DESIGN: External MCP tools for deep-worker agents

**Status:** IMPLEMENTED
**Date:** 2026-06-23

## 1. Purpose

Allow attaching an arbitrary set of executable tools from external
**MCP servers** (Model Context Protocol) to each agent. Configuration is
layered, as already done for `AGENTS.md` and skills:
user-global → project → per-agent → CLI ad-hoc.

This is NOT related to our provider-side `ProxyMCPServer`
(`claude_code_provider`, `grok_provider`, `codex_provider`): that one is a
LangGraph→CLI provider bridge, whose invariant "handler does NOT execute the
tool, only parks the Future" remains unchanged.

## 2. Architecture in one picture

```
                     project_root/.deep_worker/mcp.json   ← shared in repo
                                    ▲
~/.deep_worker/mcp.json  ───────────┤
                                    │   load_mcp_config(layers)
~/.deep_worker/agents/<a>/mcp.json ─┤   →  dict[name → ServerSpec]
                                    │
CLI --mcp-config X ─────────────────┘
CLI --mcp-server N=spec
                            │
                            ▼
              ExternalMCPClientPool.start()       ← per-agent
                            │
                            ▼
              for each server: list_tools()
                            │
                            ▼
              MCPToolAdapter(server, tool_name)  ← BaseTool
                            │
                            ▼
              create_cli_agent(tools=[…adapters])
                            │
                            ▼
              LangGraph graph: tools visible as ordinary BaseTool.
              SafeToolCallMiddleware catches exceptions,
              BoundedToolCallMiddleware gives a 2-min wait,
              interrupt_on HITL applies as for all other
              side-effect tools — EXCEPT in skip_permissions
              (auto_approve=True) mode, where NO HITL is needed.
```

Each agent holds its own client pool; there is no global state. A
crash/slow response of an external MCP server is an ordinary tool error;
the graph survives.

## 3. Exact contracts

### 3.1 `mcp.json` format

Compatible with the de-facto MCP standard (cursor / claude desktop /
codex), so that config can be copied "as is":

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${env:GH_TOKEN}" }
    },
    "intranet": {
      "type": "http",
      "url": "http://127.0.0.1:9999/mcp",
      "headers": {"Authorization": "Bearer ${env:INTRANET_TOKEN}"}
    },
    "legacy": {
      "command": "/usr/local/bin/old-mcp",
      "args": ["--mode=foo"],
      "disabled": true
    }
  }
}
```

The parser accepts both roots — `mcpServers` (preferred, de-facto
standard) and a flat dict (for compatibility with
`~/.cursor/mcp.json`-like files without a wrapper) — but the same
`ServerSpec` comes out of the output model.

Fields:
- `name` (key) — unique within the merged map. This is the server name;
  the canonical tool name at the deep-worker level is formed as
  `<server>.<tool>` (see §6), and its provider-safe variant for the
  graph/HITL/providers — via the mangler bijection
  (`DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md`).
- `type` — `"stdio"` (default if `command` is present) or
  `"http"` (if `url` is present).
- `command` + `args` — for stdio.
- `url` — for http (streamable HTTP MCP).
- `headers` — HTTP headers; values support `${env:...}`.
- `env` — map of env-vars, passed to the child process.
  Values go through `${env:VAR}` substitution.
- `cwd` — working directory of the child process (optional).
- `disabled: true` — skip (same as cursor/claude
  desktop), for quick disable without removing the entry.
- `auto_approve: true` — this server is entirely auto-approved
  even without `--skip-permissions` (optional; by default
  HITL applies as to any side-effect tool).

Variable substitution:
- `${env:VAR}` — `os.environ["VAR"]`. If the variable is missing → fail
  loud when building the config (cannot start a server with an empty
  token).
- `${env:VAR:-fallback}` — fallback if the variable is missing.

### 3.2 Layers and merge

```
1.  ~/.deep_worker/mcp.json                         (user-global)
2.  {project_root}/.deep_worker/mcp.json            (project)
3.  ~/.deep_worker/agents/<agent_name>/mcp.json     (per-agent)
4.  CLI: --mcp-config <path> (repeatable)
        --mcp-server '<name>={"command":"..."}'     (repeatable)
```

Merge — **by server name**: the higher layer fully replaces the
entry with the same name from the lower one. This gives a clear override —
no partial `env` merges (which easily break silently). An
empty/non-existent file is softly ignored.

### 3.3 `MCPToolAdapter`

A wrapper around `mcp.client.session.ClientSession`, implementing
`langchain_core.tools.BaseTool`:

- `canonical_name` = the original `<server_name>.<tool_name>` (with the dot) —
  source of truth for UI/catalog/wire call (see §6).
- `name` = **provider-safe** name (`.` and `/` → `__`); ToolNode and each
  provider (claude-code/codex/grok/deepseek/openai/gemini) route by it.
  Satisfies `^[a-zA-Z0-9_-]+$`. See
  `DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md`.
- `description` taken from MCP `tool.description`.
- `args_schema` — assembled from MCP `inputSchema` (JSON Schema)
  via a simple dynamic pydantic-V2 model.
- `_arun(**kwargs)` → `await session.call_tool(bare_tool_name, kwargs)`
  — the bare tool_name always goes to the MCP server (no server prefix,
  no mangling). Result → string (json.dumps if structured).
- Exceptions outward — let the existing
  `SafeToolCallMiddleware` catch them. Long calls — let the
  existing `BoundedToolCallMiddleware` catch them.

### 3.4 `ExternalMCPClientPool`

One per agent. `start()`:
1. Collects the final `dict[name, ServerSpec]` via
   `load_mcp_config(...)`.
2. For each `spec` raises a `ClientSession` (stdio or http)
   via the `mcp` SDK, runs `initialize()`, reads `list_tools()`.
3. Returns a list[BaseTool] of adapters.

`stop()` — properly closes all clients (`__aexit__`), kills
child processes. Called in `AgentProcess.stop()` (via the
existing lifecycle hooks).

If a particular server did not start — log-warn, the other tools
keep working. Hard-fail starts ONLY on an unresolvable environment
variable (see 3.1).

### 3.5 Integration in `create_cli_agent`

In `deep_worker_cli/agent.py`, right before `tools = tools or []`:

```python
external_mcp_pool: ExternalMCPClientPool | None = None
if not _is_running_tests():            # unit tests don't raise
    external_mcp_pool = await ExternalMCPClientPool.create(
        assistant_id=assistant_id,
        project_memory_dir=project_memory_dir,
        cli_overrides=mcp_cli_overrides,
    )
    tools = list(tools or []) + external_mcp_pool.tools
```

Since `create_cli_agent` is currently sync, async initialization
is done in one of two ways:

- **A. If `create_cli_agent` is called from an async context**
  (`AgentProcess.start()` — yes): the pool is raised via `await` BEFORE
  calling `create_cli_agent`, and a ready `tools=` list is passed.
  This is the default path.
- **B. If called synchronously** (tests, benchmarks): the pool is not
  initialized, the config is ignored. This matches the current
  situation where tools like web_search etc. are not needed by tests.

In practice — we put all async logic in `AgentProcess.start()`
(calling `await ExternalMCPClientPool.create(...)` + passing the
ready list to `create_cli_agent(tools=...)`), and do NOT touch
lifecycle in `create_cli_agent`.

### 3.6 HITL / interrupt_on / `--skip-permissions`

A hard rule at the user's request:

> If the agent is started with the `-sp` option (skip permissions), then
> no additional HITL approvals are required for tool calls — i.e. the
> agent simply calls the tool and that's it.

`create_cli_agent` already has exactly the needed fork:

```python
if auto_approve:
    interrupt_on = {}     # <- nothing gets interrupted
else:
    interrupt_on = _add_interrupt_on(...)
```

Our integration:
- With `auto_approve=True` (caused by `--skip-permissions`) external
  MCP tools are NOT added to `interrupt_on`. Period.
- With `auto_approve=False` external MCP tools go into
  `interrupt_on` by default (like any side-effect tools), EXCEPT servers
  with `auto_approve: true` in `mcp.json`.

This naturally fits the existing `_add_interrupt_on`:
we extend its signature with an optional `extra_tool_names: set[str]`
(adapter tool names) and an optional `auto_approved_tool_names`
(tools from servers with `auto_approve: true`).

### 3.7 CLI

The `deep-worker mcp` subcommand (new, does not break existing):
- `deep-worker mcp list [--scope user|project|agent:<n>|all]`
  — prints the final merged config and, for each server,
  tries to read the tool list (with a timeout).
- `deep-worker mcp add <name> --command "npx -y @x/foo" \
       [--env K=V] [--scope user|project|agent:<n>] \
       [--disabled] [--auto-approve]`
  — writes to the appropriate json (creates the directory if needed).
  Files are created with permissions `0o600` (they may contain tokens).
- `deep-worker mcp remove <name> --scope ...`
- Flags that directly launch the agent:
  `deep-worker <name> --mcp-config <path>` (repeatable) and
  `--mcp-server '<n>={...json...}'` (repeatable). These flags
  are saved in `launch_config` the same way as already done for
  boolean options (via `_BOOLEAN_FLAGS`-like registration).

### 3.8 What we do NOT do (anti-patterns under policies)

1. We do not write to global configs of CLI providers
   (`~/.grok/`, `~/.codex/`) — lesson from an earlier internal incident.
2. We do not create threads — only asyncio tasks in the agent loop.
3. We do not use polling — the `mcp` SDK is already event-driven.
4. We do not `pip install` — `mcp` is already available.
5. We do not mix these tools into the provider's `ProxyMCPServer`
   (a different role; breaks the invariant "handler does NOT execute the tool").
6. We do not expose secrets in logs — env values are redacted.

## 4. TDD plan

1. **Unit:** `load_mcp_config` — empty/non-existent file,
   layers in the correct order, override by name, parsing
   stdio/http forms, `${env:VAR}` happy-path and failure, parsing
   of both root forms (`mcpServers` and flat).
2. **Unit:** `MCPToolAdapter` — correct `name`,
   `description`, proxying `_arun → call_tool`,
   result serialization.
3. **Unit:** `_add_interrupt_on` — external tools correctly
   do/do NOT get into `interrupt_on` depending on
   `auto_approve` and per-server `auto_approve: true`.
4. **Integration:** a local fake stdio-MCP server
   (on our own `mcp.server` SDK) with the `add(a,b)` tool. Raise
   the pool, invoke the tool, check the result, correct
   shutdown.
5. **Smoke live:** a real `@modelcontextprotocol/server-everything`
   (see §6 of this file) — the tool list is read, one of the tools
   is invoked — out of CI, gated by an env variable.

## 5. Compatibility

- Old agents that have no `mcp.json` and no CLI
  flags — unchanged behavior (empty pool, zero adapters).
- Full Dialog `verbosity=tools` (my previous fix) automatically
  shows external tools — `_extract_final_tool_metadata` reads
  the compiled graph, no edits needed.
- Agent restart: since the config is file-based + flags saved in
  `launch_config`, after `deep-worker --restart <name>` the tools
  come up again idempotently.

## 6. Tool name glossary

Inside deep-worker, TWO names live for each external MCP tool:

- `canonical_name` = `<server_name>.<tool_name>` (e.g.
  `chrome.browser.list_tabs`) — the canonical dot-notation of the MCP spec
  ([MCP spec 2025-11-25 §Tool Names](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#tool-names)
  and [SEP-986 final](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986)
  explicitly allow `.` and `/`). Source of truth for UI/catalog/wire
  (the tool is registered on the actual MCP server under the bare `<tool_name>`).
- `name` (== final `BaseTool.name`) — **provider-safe** (`.`/`/` →
  `__`, see `DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md`). ToolNode,
  HITL `interrupt_on`, per-server auto-approve, as well as the
  chat_model of each CLI provider and the standard `bind_tools` of
  OpenAI-class providers route by it.

The canonical ↔ name bijection lives in `ExternalMCPClientPool.mangler`
(`MCPToolNameMangler`). Only actually mangled names are stored there;
for the rest `mangled = canonical`. Collisions — fail-loud.

Tools provided through the internal ProxyMCPServer
(deepworker tools): on the deep-worker side they are usually without such a
prefix or native (execute, read_file, etc.). Providers
(claude-code etc.) may add their own
prefix `mcp__deepworker__*` when exporting to their MCP — that is their
responsibility and does not affect naming of external MCP in deep-worker.

Format history: `mcp__<server>__<tool>` (rejected — unreadable) →
dot-notation `<server>.<tool>` (broke on claude-code: the CLI silently
mangles `.`→`_` on top of the MCP spec, plus the OpenAI API rejects HTTP 400)
→ **currently**: canonical with dots + provider-safe `__` via bijection
(one tool, two names for two audiences).

## 7. Security

- `mcp.json` is always created with permissions `0o600` (it often
  contains tokens).
- Env values are NOT logged (we log only the variable name).
- `--mcp-server '<n>={...}'` forbids inline secrets: at
  parse time, if there's `${env:...}` inside — ok; if a plain
  token in `env` (heuristic — the key matches `*_TOKEN|*_KEY|*_SECRET`
  and the value is longer than 16 characters without the `${env:` prefix) — warning
  on stderr, but no fail.

### 7.1 Sources of `${env:VAR}` values — drop-in `env.d`

`mcp.json` refers to secrets via `${env:VAR}`. So that the agent does not
need to hardcode values and so that canonical secret sources
(e.g. `~/.config/my-chrome-mcp/admin.env`, the same file that
raises the systemd unit of the bridge itself) are not duplicated in
`~/.deep_worker/.env`, the agent additionally loads
**drop-in env files** from the `~/.deep_worker/env.d/` directory at startup.

Contract (see `deep_worker_cli/process/env_loader.py`):

* The directory is optional. None — no-op, the agent starts as before.
* Files with the `*.env` extension are picked up in **sorted**
  order (determinism).
* Each file (or symlink target) must have permissions `0o600` (like
  `mcp.json`). Files with group/world bits — **skipped** with a warning.
  This protects against accidental secret leakage on first run.
* Already-set variables (from `~/.deep_worker/.env` or inherited
  from the shell) are NEVER overwritten (`override=False`). This means:
    - the main `~/.deep_worker/.env` remains the highest priority — a drop-in
      cannot accidentally "beat" it;
    - if two drop-in files set the same variable, the one whose
      name comes earlier in `sorted()` wins.
* Symlinks **allowed and recommended**: typical setup —
    ```bash
    mkdir -p ~/.deep_worker/env.d
    ln -s ~/.config/my-chrome-mcp/admin.env ~/.deep_worker/env.d/my-chrome-mcp.env
    chmod 0600 ~/.config/my-chrome-mcp/admin.env   # once
    ```
  The real file with the token stays at its canonical location (where the
  bridge's systemd unit uses it); deep-worker only picks up its
  value via the symlink. No manual copies of values, no edits to the
  agent launch/restart workflow.

Behavior is insensitive to the launch command (`deep-worker <name>` /
`deep-worker --restart <name>` / web restart) — the drop-in applies
identically on every launch.
