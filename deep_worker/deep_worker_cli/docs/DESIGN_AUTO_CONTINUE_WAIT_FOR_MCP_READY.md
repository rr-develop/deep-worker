# Auto-continue after `restart_self` must wait for MCP tool registration

**Status:** ❌ SUPERSEDED 2026-06-14 by deterministic prewarm — see
a related internal incident analysis (not included here).
**Authors:** original 2026-06-12 polling design; superseded 2026-06-14.

## Why this document is no longer authoritative

The 2026-06-12 design replaced a blind 0.5s sleep in
`_auto_continue_after_restart` with a 30s polling barrier on
`ClaudeCodeChatModel.tools_metadata`
(`AgentProcess._wait_for_mcp_tools_ready`). The barrier returned `True`
on the **first non-empty** snapshot of `tools_metadata`.

That design had a subtle race that surfaced on restart and caused
incident 2026-06-13 (#2):

* On a regular start, the first non-empty `tools_metadata` came from
  the LangGraph turn-node's `bind_tools(final_tools=23)` — the FULL
  inventory — so polling happened to release at the right time.
* On a restart, the first non-empty snapshot came from
  `deepagents/graph.py:252` `_bind_tools_for_middleware(model, tools=7)`
  — the caller-supplied `web_tools` SUBSET, computed
  synchronously inside `create_deep_agent` BEFORE LangGraph could
  attach the middleware tools. Polling released early, the first
  `ainvoke` cached a `ProxyMCPServer` with the wrong (4-of-7) tool set,
  and `ClientManager.clients[name]` then served that stale proxy on
  every subsequent turn for the lifetime of the session.

The cluster post-mortem
(a related internal incident analysis (not included here))
established the architectural property that makes a deterministic fix
possible:

> The compiled LangGraph agent already knows its FINAL tool set
> synchronously at the moment `create_agent(...)` returns — see
> `langchain/agents/factory.py:880-905`:
> ```python
> available_tools = middleware_tools + regular_tools
> tool_node = ToolNode(tools=available_tools, ...)
> ```
> `tool_node.tools_by_name` is the inventory every turn will use; it
> does not change after compile. Polling for a side effect of
> `bind_tools` is therefore redundant — and, as shown above, racy.

## What replaced it

`AgentProcess.start()` now:

1. Reads `tool_node.tools_by_name` straight off the compiled agent via
   `_extract_final_tool_metadata(self.agent)`.
2. Calls `_prewarm_client_manager(client_manager, thread_id=self.name,
   tools_metadata=…, working_dir=…)` BEFORE
   `_auto_continue_after_restart` runs. This seeds
   `ClientManager.clients[self.name]` with a `ProxyMCPServer` built
   from the FINAL tool inventory, so the first real `ainvoke` hits the
   cache with the correct surface.

`_wait_for_mcp_tools_ready` and its test
(`test_auto_continue_waits_for_mcp_tools.py`) have been **deleted**.
The polling barrier in `_auto_continue_after_restart` has been
**removed**.

Defence in depth, on the `ClientManager` side:
`get_or_create_client` now checks the cached entry's
`ProxyMCPServer.tool_names` against the incoming `tools_metadata`. On
mismatch it disconnects the stale client and creates a fresh one with
the new inventory. See
`claude_code_provider/client_manager.py::_cache_entry_matches_tools` /
`_drop_stale_cache_entry`.

## Why not just "wait longer"

Tuning the timeout would have hidden the bug for `web_tools.len ==
final.len` cases and surfaced it again as soon as a middleware
contributed more tools. The deterministic prewarm is the proper fix:
no race window exists for the next maintainer to widen.

## Do not reintroduce polling here

If a regression ever lands that ends up with a stale `ProxyMCPServer`
again, fix the prewarm — not this entry point. Polling for a side
effect is the failure mode this document, in its previous incarnation,
helped land in production.
