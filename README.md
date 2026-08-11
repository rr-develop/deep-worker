# Deep Worker

An AI agent framework for running LLM agents as local background processes and
interacting with them through a terminal CLI.

Deep Worker is a fork and continuation of
[`deepagents-cli`](https://github.com/langchain-ai/deepagents) (content-equivalent
to `langchain-ai/deepagents@8f8fc98`): agents run as daemon processes, persist
conversation history, and are driven interactively with `dwcli`.

## Components

| Directory | Purpose |
|---|---|
| `deep_worker/deep_worker_cli` | Core agent runtime: `deep-worker` daemon and `dwcli` interactive client |
| `claude_code_provider` | Claude Code provider — LangChain `BaseChatModel` backed by the Claude Code SDK |
| `codex_provider` | OpenAI Codex CLI provider — drop-in sibling of `claude_code_provider` |
| `grok_provider` | xAI Grok CLI provider (Agent Client Protocol, ACP v1) |
| `dw_hist` | `dw-hist` — host-level CLI for semantically browsing agent conversation history |
| `lsp` | Optional LSP code navigation for agents (Serena MCP integration) |
| `semantic_search` | Semantic search over history (LlamaIndex + LanceDB) and per-message embeddings |
| `deep_worker_web` | Local web UI for working with deep-worker agents (127.0.0.1 only) |
| `tools` | Helper tools: custom agent tools, memory monitoring, RSS monitor |

## Quick Start

Deep Worker is developed to run from a source checkout with the system Python —
no virtual environment (see `deep_worker/README.md` for why).

```bash
# 1. Make the launchers globally available (project policy: no pip install)
ln -sf "$PWD/deep_worker/deep_worker_cli/deep-worker" ~/.local/bin/deep-worker
ln -sf "$PWD/deep_worker/deep_worker_cli/dwcli" ~/.local/bin/dwcli

# 2. Start an agent as a background daemon
deep-worker my_agent

# 3. Connect to it interactively
dwcli my_agent
```

The agent daemon and the CLI communicate over rpyc_async — an
AsyncioServer-based RPyC fork — on `127.0.0.1`. All services bind to
localhost only and are designed for **local trusted environments** — see
`deep_worker/deep_worker_cli/SECURITY.md`.

## Notable features

- **External MCP tools** — attach tools from external MCP servers, layered
  configuration (user-global → project → per-agent → CLI)
- **A2A protocol support** — inbound A2A server plus outbound tools for
  agent-to-agent interoperability
- **Delta checkpointer** — store only checkpoint changes, not full state
- **`read_image` tool** — pass image files to vision-capable providers
- **`send_message_to_agent` tool** — inter-agent messaging with replies
- **Summarization** — automatic context compaction plus on-demand
  `deep-worker <agent> --summarize`
- **Storage auto-migration (v4)** — agents migrate to the current storage
  format automatically on restart

See [`docs/ARCHITECTURE_OVERVIEW.md`](docs/ARCHITECTURE_OVERVIEW.md) for the
full architecture map.

## Documentation

- [`docs/`](docs/) — component documentation, design notes and internal policies
- [`docs/ARCHITECTURE_OVERVIEW.md`](docs/ARCHITECTURE_OVERVIEW.md) — architecture overview: daemon model, components, storage, notable features
- [`deep_worker/deep_worker_cli/README.md`](deep_worker/deep_worker_cli/README.md) — CLI usage, storage options, thread management
- [`dw_hist/README.md`](dw_hist/README.md) — `dw-hist` usage
- [`lsp/README.md`](lsp/README.md) — LSP code navigation for agents
- [`semantic_search/docs/README_MESSAGE_EMBEDDINGS.md`](semantic_search/docs/README_MESSAGE_EMBEDDINGS.md) — per-message embeddings

## License

[MIT](LICENSE). Upstream portions are Copyright (c) LangChain, Inc.; local
contributions are Copyright (c) Roman Rodionov.
