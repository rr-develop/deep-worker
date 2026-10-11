# Architecture Overview

Deep Worker runs LLM agents as local background daemon processes and lets you
drive them interactively from a terminal. This document is a user-facing map
of how the pieces fit together; component READMEs linked from the root
[`README.md`](../README.md) and the [`docs/` index](README.md) carry the
operational detail.

## Runtime model

- **Daemon + CLI.** Each agent runs as its own background process
  (`deep-worker <agent>`). You talk to it through the interactive `dwcli`
  client, which connects to the daemon over **rpyc_async** — an
  AsyncioServer-based RPyC fork — on `127.0.0.1` (port `18861` by default).
  The rpyc_async service runs a single event loop with no worker threads.
- **Localhost only.** All network services bind exclusively to `127.0.0.1`
  and carry **no authentication** — the security model is localhost
  isolation on a trusted machine. See
  [`deep_worker/deep_worker_cli/SECURITY.md`](../deep_worker/deep_worker_cli/SECURITY.md).
- **No virtual environments.** The project runs with the system Python from
  a source checkout; rpyc_async is imported from a local source checkout
  (`rpyc-async`, an AsyncioServer-based RPyC fork) rather than from PyPI.
  Using a `.venv` or `pip install rpyc` breaks the proxy layer, so it is
  deliberately unsupported
  (see [`deep_worker/README.md`](../deep_worker/README.md)).

## Components

| Component | Purpose |
|---|---|
| [`deep_worker/deep_worker_cli`](../deep_worker/deep_worker_cli/README.md) | Core runtime: `deep-worker` daemon, `dwcli` client, storage, providers |
| [`claude_code_sdk_provider`](../claude_code_sdk_provider/README.md) | Claude Code provider — LangChain `BaseChatModel` backed by the Claude Code SDK |
| [`codex_provider`](../codex_provider/README.md) | OpenAI Codex CLI provider — drop-in sibling of `claude_code_sdk_provider` |
| [`grok_provider`](../grok_provider/README.md) | xAI Grok CLI provider (Agent Client Protocol, ACP v1) |
| [`dw_hist`](../dw_hist/README.md) | `dw-hist` — host-level CLI for semantically browsing agent history |
| [`lsp`](../lsp/README.md) | Optional LSP code navigation for agents (Serena MCP integration) |
| [`semantic_search`](../semantic_search/README.md) | Semantic search over history (LlamaIndex + LanceDB) and per-message embeddings |
| [`deep_worker_web`](../deep_worker_web/README.md) | Local web UI for working with deep-worker agents (127.0.0.1 only) |
| [`tools`](../tools/README.md) | Helper tools: custom agent tools, memory monitoring, RSS monitor |

## Conversation storage

- Agents persist conversation history to a per-agent **SQLite** database by
  default, or keep it in memory when asked (`--in-memory`).
- History is organised into **threads**; the CLI supports listing, resuming,
  and pruning old threads.
- When a model's context fills up, `SummarizationMiddleware` compacts the
  conversation: older messages are summarised, and the offloaded summaries
  are stored under `~/.deep_worker/<agent>/summaries/<thread>.md`, one
  append-only file per thread. `deep-worker --list-summaries` and
  `--remove-before DATE` manage those files.
- Storage is versioned; agents **auto-migrate to storage v4 on restart**.

## Notable features

These capabilities are described in depth by design documents under
`deep_worker/deep_worker_cli/docs/`; this section gives the one-paragraph
summary.

### External MCP tools

Attach an arbitrary set of executable tools from external **MCP servers**
(Model Context Protocol) to any agent. Configuration is layered, mirroring
the `AGENTS.md` / skills approach: user-global → project → per-agent →
CLI ad-hoc (`--mcp-config`, `--mcp-server`). A per-agent pool connects to
each server, lists its tools, and adapts them into LangGraph `BaseTool`s.
Design: `DESIGN_EXTERNAL_MCP_TOOLS.md`.

### A2A protocol support

Interoperate with other agents over the **A2A** (Agent-to-Agent) protocol:
an inbound A2A server exposes an agent to the network, and outbound tools
let an agent call remote agents. The implementation lives in
`deep_worker/deepworker_a2a/` (server, client, card, bridge, executor,
tools). Design: `deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`.

### Delta checkpointer

The default checkpointer stores **only the changes** between checkpoints
instead of a full state snapshot on every step, cutting storage and write
cost for long agent runs. Module: `deep_worker_cli/delta_checkpointer.py`.
Design: `claude_code_sdk_provider/docs/DESIGN_delta_checkpointer.md`.

### `read_image` tool

A built-in tool that reads an image file from disk and passes it to the
model as visual content — so vision-capable providers (e.g. Claude Code)
can analyse screenshots and diagrams. Design:
`DESIGN_READ_IMAGE_TOOL.md`.

### `send_message_to_agent` tool

A built-in tool for inter-agent messaging: an agent can deliver a message
to another running agent and receive its reply, enabling multi-agent
workflows from within a conversation. Design:
`DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md`.

### User-requested summarization

Besides automatic context-size compaction, you can compact a conversation
on demand:

```bash
deep-worker <agent> --summarize                 # default summarization
deep-worker <agent> --summarize --query TEXT    # one-shot custom prompt
```

`--summarize` is one-shot and never persisted; it uses the same prompt,
retention, offload, and failure policy as automatic compaction. Design:
`DESIGN_USER_REQUESTED_SUMMARIZATION.md`.

### Storage auto-migration (v4)

Agent storage is migrated to the current format automatically on restart,
so older checkpoints keep working without manual conversion. Relevant
modules: `storage_paths.py`, `agent_process.py`, `agent_command.py`,
`config.py`, `session_manager.py`.

## Security model

All services bind to `127.0.0.1` only and are designed for **local trusted
environments**: no authentication, `allow_public_attrs` and `allow_pickle`
enabled on the rpyc_async channel, and no external interfaces. Do not expose
the daemon port (`18861` by default) to a network. See
[`deep_worker/deep_worker_cli/SECURITY.md`](../deep_worker/deep_worker_cli/SECURITY.md).
