# grok_provider

xAI **Grok** CLI provider for deep-worker — a sibling of `codex_provider` /
`claude_code_provider`. Exposes `GrokChatModel`, a LangChain `BaseChatModel`
that drives the Grok CLI in its `grok agent stdio` mode (the **Agent Client
Protocol**, ACP v1, JSON-RPC over stdio).

Authentication is by the user's **supergrok** subscription cache at
`~/.grok/auth.json` — no separate API key.

## Architecture (see `docs/DESIGN_GROK_PROVIDER.md`)

- **One long-lived `grok agent stdio` subprocess per agent process**
  (`acp_client.py`), shared by every `agent_thread_id`; each thread owns a
  distinct `grok_session_id` (separate `session/new`). One turn = one
  `session/prompt`.
- **Two tool channels, both recorded in LangGraph history:**
  - **Channel A — deep-worker MCP tools** via the in-agent proxy MCP server
    (`proxy_mcp_server.py`). Grok dispatches them through `CallMcpTool`; the
    handler parks an `asyncio.Future` resolved when the `ToolMessage`
    returns from LangGraph's ToolNode. The handler NEVER executes a tool.
  - **Channel B — Grok's local built-ins** (`Shell`/`Read`/`Edit`/…). Grok
    runs these itself and cannot be told not to; we observe them on the ACP
    stream and record them via `builtin_tool_passthrough.py` so they land in
    `result["messages"]`.
- **State** (`client_state.py` + `custom_types.py`): a process-global
  singleton keyed by `agent_thread_id` (never mutated). On summarization the
  slot contents reset and the stale session is evicted from the reverse
  `sessionId → agent_thread_id` index; the key is untouched.
- **Persistence** (`session_state.py`): `(grok_session_id, sent_count)` to
  `~/.deep_worker/<agent>/grok_session_state.json` (dir `0o700`, file
  `0o600`, atomic write).

## Tests

```bash
# Unit tests (hermetic — no grok binary):
cd grok_provider && pytest tests -q

# Live integration tests (need grok CLI + ~/.grok/auth.json):
GROK_INTEGRATION_TESTS=1 pytest tests/test_integration.py -v
```

## Registration in deep-worker

`--provider grok`. Wired in `deep_worker_cli`: `agent_config.py`,
`agent_command.py`, `config.py` (factory + default model), `provider_chain.py`
(failover `claude-code → codex → grok → deepseek`), and the built-in
passthrough binding in `agent_process.py` / `task_manager.py`.
