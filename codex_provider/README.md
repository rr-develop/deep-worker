# codex_provider

LangChain `BaseChatModel` wrapper around OpenAI's **Codex CLI**, designed as
a drop-in sibling of `claude_code_provider/` for deep-worker.

Use it as:

```bash
deep-worker my-agent --provider codex
```

No API key needed: Codex CLI uses its own cached login at
`~/.codex/auth.json`.

## What this package provides

| Module | Purpose |
|---|---|
| `chat_model.py` | `CodexChatModel(BaseChatModel)` — the public surface |
| `client_state.py` | Process-global singleton: per-`agent_thread_id` `(codex_session_uuid, sent_count)` |
| `exec_runner.py` | `asyncio.create_subprocess_exec` of `codex exec --experimental-json` + JSONL parser; zero runtime dep on the OpenAI Agents SDK |
| `proxy_mcp_server.py` | In-agent-process HTTP MCP server (asyncio task, **not** an external process) that proxies tool calls to LangGraph's ToolNode via `asyncio.Future`s |
| `custom_types.py` | Shared TypedDicts / dataclasses |
| `docs/DESIGN_CODEX_PROVIDER.md` | Full design (rev 3, implemented) |

## Architecture in one paragraph

`CodexChatModel.ainvoke()` mirrors only the **tail delta**
`messages[sent_count:]` into a per-turn `codex exec` subprocess
(`resume <uuid>` after the first turn), captures the session UUID
Codex assigns, and reads its JSONL stdout event-driven. Deep-worker
tools are exposed to Codex via a **single in-agent-process HTTP MCP
server** (`deepworker_<pid>`). It is injected **per-invocation** into
each `codex exec` (`--ignore-user-config` + `--config
mcp_servers.<name>.url=...`), NOT registered into the shared
`~/.codex/config.toml`. Writing the global config would leak a dead
MCP entry per agent and make every `codex exec` hang on stale ports —
see `docs/DESIGN_CODEX_PROVIDER.md` §4.3.1 (observed 2026-05-20).
Codex native tools that overlap the base deep-worker tool set are
disabled with Codex CLI's `disabled_tools=[...]` config and guarded at
runtime: if a native command/file/search item still appears, the turn
fails instead of bypassing LangGraph.
When Codex calls a tool, our handler parks an `asyncio.Future`; the
matching `ToolMessage` from LangGraph's ToolNode resolves it.
`ainvoke()` waits for the handler to register a pending call via a
one-shot Future waiter, not a sleep loop, so the bridge stays
event-driven and cancellable. This guarantees every tool call AND
every tool result lands in `result["messages"]` — the same path dwcli
broadcast, the checkpointer, and TaskManager already use. On LangGraph
summarization (`sent_count > len(messages)`) we drop the cached
`codex_session_uuid` and start fresh; **`agent_thread_id` is never
mutated** — that's the deep-worker contract.

## Type checking

This package is checked under **pyright strict** by the pre-commit
hook in `.githooks/pre-commit`. The provider source (`codex_provider/`)
runs under full strict mode. The tests directory has one targeted
relaxation in `pyrightconfig.json`:

```json
"executionEnvironments": [
  { "root": "tests", "reportPrivateUsage": "none" }
]
```

Reason: tests inject mocks into Pydantic `PrivateAttr` fields
(`_proxy_mcp`, `_exec_runner`, `_client_state`, `_tools_metadata`, …)
and assert on them. That's legitimate test-side DI — Pydantic's own
convention for private attributes is the leading underscore. We turn
`reportPrivateUsage` off for the `tests/` directory only. The provider
package itself is still strict-checked.

## Tests

* `tests/test_*.py` — unit tests with mocked subprocess + mocked
  proxy. Always run.
* `tests/test_integration.py` — live tests against a real `codex`
  binary / deep-worker agent. Skipped unless
  `CODEX_INTEGRATION_TESTS=1`.

Run all units:

```bash
cd /path/to/codex_provider
python3 -m pytest tests/ -v
```

Run integration:

```bash
CODEX_INTEGRATION_TESTS=1 python3 -m pytest tests/test_integration.py -v
```
