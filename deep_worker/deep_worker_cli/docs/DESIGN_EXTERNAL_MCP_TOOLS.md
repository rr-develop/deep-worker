# DESIGN: External MCP tools for deep-worker agents

**Status:** IMPLEMENTED
**Date:** 2026-06-23

## 1. Goal

Give every agent the ability to attach an arbitrary set of
executable tools from external **MCP servers** (Model Context Protocol).
Configuration is layered, as already done for `AGENTS.md` and skills:
user-global → project → per-agent → CLI ad-hoc.

This is NOT related to our provider-side `ProxyMCPServer`
(`claude_code_provider`, `grok_provider`, `codex_provider`): that one is
a LangGraph→CLI-provider bridge whose invariant — "the handler does NOT
execute the tool, it only parks the Future" — remains unchanged.

## 2. Architecture in one picture

```
                     project_root/.deepagents/mcp.json   ← shared in the repo
                            ▲
~/.deepagents/mcp.json  ────┤
                            │   load_mcp_config(layers)
~/.deepagents/<a>/mcp.json ─┤   →  dict[name → ServerSpec]
                            │
CLI --mcp-config X ─────────┘
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
              LangGraph graph: the tools are visible as ordinary BaseTools.
              SafeToolCallMiddleware catches exceptions,
              BoundedToolCallMiddleware gives a 2-min wait,
              interrupt_on HITL applies as to all other
              side-effect tools — EXCEPT in skip_permissions
              (auto_approve=True) mode, where NO HITL is needed.
```

Each agent keeps its own client pool; there is no global
state. A crash or slow response from an external MCP server is an
ordinary tool error; the graph survives.

## 3. Exact contracts

### 3.1 `mcp.json` format

Compatible with the de-facto MCP standard (cursor / claude desktop /
codex), so the config can be copied "as is":

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

The parser accepts both roots — `mcpServers` (preferred,
de-facto standard) and a flat dict (for compatibility with
`~/.cursor/mcp.json`-style files without the wrapper) — but the
output model receives the same `ServerSpec` either way.

Fields:
- `name` (key) — unique within the merged map.
  Matches the visible tool name as
  `mcp__<name>__<tool>` (see §6).
- `type` — `"stdio"` (default when `command` is present) or
  `"http"` (when `url` is present).
- `command` + `args` — for stdio.
- `url` — for http (streamable HTTP MCP).
- `headers` — HTTP headers; values support `${env:...}`.
- `env` — a map of env-vars, passed to the child process.
  Values go through `${env:VAR}` substitution.
- `cwd` — the child process's working directory (optional).
- `disabled: true` — skip it (same as cursor/claude
  desktop), for quick disabling without deleting the entry.
- `auto_approve: true` — this whole server is auto-approved
  even without `--skip-permissions` (optional; by default
  HITL applies as to any side-effect tool).

Variable substitution:
- `${env:VAR}` — `os.environ["VAR"]`. If the variable is missing → fail
  loud when building the config (a server must not start with an empty
  token).
- `${env:VAR:-fallback}` — fallback if the variable is missing.

### 3.2 Layers and merge

```
1.  ~/.deepagents/mcp.json                          (user-global)
2.  {project_root}/.deepagents/mcp.json             (project)
3.  ~/.deepagents/<agent_name>/mcp.json             (per-agent)
4.  CLI: --mcp-config <path> (repeatable)
        --mcp-server '<name>={"command":"..."}'     (repeatable)
```

The merge is **by server name**: a higher layer fully replaces the
record with the same name from a lower layer. This gives a clear
override — no partial `env` merges (which easily break
silently). An empty/non-existent file is softly ignored.

### 3.3 `MCPToolAdapter`

A wrapper around `mcp.client.session.ClientSession` that implements
`langchain_core.tools.BaseTool`:

- `name` = `mcp__<server_name>__<tool_name>` (the same prefix
  as our proxy — a single format for the LLM).
- `description` comes from the MCP `tool.description`.
- `args_schema` — built from the MCP `inputSchema` (JSON Schema)
  via a simple dynamic pydantic-V2 model.
- `_arun(**kwargs)` → `await session.call_tool(tool_name, kwargs)`,
  result → string (json.dumps if structured).
- Exceptions propagate outward — let the existing
  `SafeToolCallMiddleware` catch them. Long calls — let the existing
  `BoundedToolCallMiddleware` catch them.

### 3.4 `ExternalMCPClientPool`

One per agent. `start()`:
1. Builds the final `dict[name, ServerSpec]` via
   `load_mcp_config(...)`.
2. For each `spec`, brings up a `ClientSession` (stdio or http)
   via the `mcp` SDK, runs `initialize()`, reads `list_tools()`.
3. Returns the list[BaseTool] of adapters.

`stop()` — properly closes all clients (`__aexit__`), kills the
child processes. Called in `AgentProcess.stop()` (through the
existing lifecycle hooks).

If a specific server fails to start — log a warning; the other tools
keep working. A hard fail starts ONLY on an unresolvable environment
variable (see 3.1).

### 3.5 Integration into `create_cli_agent`

In `deep_worker_cli/agent.py`, right before `tools = tools or []`:

```python
external_mcp_pool: ExternalMCPClientPool | None = None
if not _is_running_tests():            # unit tests don't bring up
    external_mcp_pool = await ExternalMCPClientPool.create(
        assistant_id=assistant_id,
        project_memory_dir=project_memory_dir,
        cli_overrides=mcp_cli_overrides,
    )
    tools = list(tools or []) + external_mcp_pool.tools
```

Since `create_cli_agent` is currently sync, the async initialization
is done one of two ways:

- **A. If `create_cli_agent` is called from an async context**
  (`AgentProcess.start()` — yes): bring up the pool via `await` BEFORE
  calling `create_cli_agent`, and pass the ready `tools=` list.
  This is the default path.
- **B. If the caller is sync** (tests, benchmarks): the pool is not
  initialized, the config is ignored. This behavior
  matches the fact that tools like web_search are not needed by
  tests.

In practice — we put all the async logic into `AgentProcess.start()`
(the `await ExternalMCPClientPool.create(...)` call + passing the
ready list into `create_cli_agent(tools=...)`), and do NOT touch
lifecycle in `create_cli_agent`.

### 3.6 HITL / interrupt_on / `--skip-permissions`

A hard rule at the user's request:

> If the agent is started with the `-sp` option (skip permissions), then
> no additional HITL approvals are required for a tool call —
> i.e. the agent simply calls the tool and that's it.

`create_cli_agent` already has exactly the needed fork:

```python
if auto_approve:
    interrupt_on = {}     # <- nothing gets interrupted
else:
    interrupt_on = _add_interrupt_on(...)
```

Our integration:
- With `auto_approve=True` (triggered by `--skip-permissions`), external
  MCP tools are NOT added to `interrupt_on`. Period.
- With `auto_approve=False`, external MCP tools go into
  `interrupt_on` by default (like any side-effect tools), EXCEPT servers
  with `auto_approve: true` in `mcp.json`.

This naturally fits into the existing `_add_interrupt_on`:
we extend its signature with an optional `extra_tool_names: set[str]`
(the adapter tool names) and an optional `auto_approved_tool_names`
(tools from servers with `auto_approve: true`).

### 3.7 CLI

The `deep-worker mcp` subcommand (new, does not break existing ones):
- `deep-worker mcp list [--scope user|project|agent:<n>|all]`
  — prints the final merged config and, for each server,
  tries to read the tool list (with a timeout).
- `deep-worker mcp add <name> --command "npx -y @x/foo" \
       [--env K=V] [--scope user|project|agent:<n>] \
       [--disabled] [--auto-approve]`
  — writes into the corresponding json (creating the directory if needed).
  Files are created with `0o600` permissions (they may contain tokens).
- `deep-worker mcp remove <name> --scope ...`
- Flags that directly launch an agent:
  `deep-worker <name> --mcp-config <path>` (repeatable) and
  `--mcp-server '<n>={...json...}'` (repeatable). These flags are
  saved in `launch_config` the same way as is already done for
  boolean options (via a `_BOOLEAN_FLAGS`-like registration).

### 3.8 What we do NOT do (anti-patterns under the policies)

1. We do not write to the CLI providers' global configs
   (`~/.grok/`, `~/.codex/`) — a lesson learned live.
2. We do not create threads — only asyncio tasks in the agent's loop.
3. We do not use polling — the `mcp` SDK is already event-driven.
4. We do not `pip install` — `mcp` is already available.
5. We do not mix these tools into the provider's `ProxyMCPServer`
   (a different role; it breaks the "handler does NOT execute the tool"
   invariant).
6. We do not expose secrets in logs — env values are redacted.

## 4. TDD plan

1. **Unit:** `load_mcp_config` — empty/non-existent file,
   layers in the correct order, override by name, parsing of
   stdio/http forms, `${env:VAR}` happy-path and failure, parsing of
   both root forms (`mcpServers` and flat).
2. **Unit:** `MCPToolAdapter` — correct `name`,
   `description`, `_arun → call_tool` proxying,
   result serialization.
3. **Unit:** `_add_interrupt_on` — external tools correctly
   do/don't land in `interrupt_on` depending on
   `auto_approve` and per-server `auto_approve: true`.
4. **Integration:** a local fake stdio MCP server
   (on our own `mcp.server` SDK) with an `add(a,b)` tool. Bring up the
   pool, call the tool, verify the result, and shut down
   correctly.
5. **Live smoke:** a real `@modelcontextprotocol/server-everything`
   (see §6 of this file) — the tool list is read, one of the tools
   is invoked — outside CI, gated by an env variable.

## 5. Compatibility

- Old agents that have neither any `mcp.json` nor any CLI flag —
  behavior unchanged (empty pool, zero adapters).
- Full Dialog `verbosity=tools` (a previous fix) automatically
  shows the external tools — `_extract_final_tool_metadata` reads
  the compiled graph; no edits needed.
- Agent restart: since the config is file-based and the flags are saved
  in `launch_config`, after `deep-worker --restart <name>` the tools
  come back up idempotently.

## 6. Prefix glossary

The LLM sees our own tools as `mcp__deepworker__<name>`
(the prefix is set by the provider SDK, see `claude_code_provider`).
External MCP tools will similarly appear as
`mcp__<server_name>__<tool>`. This is deliberate: one
format for the model, simpler prompt writing.

## 7. Security

- `mcp.json` is always created with `0o600` permissions (it often
  contains tokens).
- Env values are NOT logged (we log only the variable name).
- `--mcp-server '<n>={...}'` forbids inline secrets: during
  parsing, if there is `${env:...}` inside — ok; if a plain
  token is in `env` (heuristic — the key matches `*_TOKEN|*_KEY|*_SECRET`
  and the value is longer than 16 chars without the `${env:` prefix) — a
  warning goes to stderr, but it is not a fail.
