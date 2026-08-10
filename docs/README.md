# Documentation

This directory collects project documentation, split into user-facing material
and internal design/process notes.

## User-facing

| Document | Purpose |
|---|---|
| [`deep_worker/deep_worker_cli/README.md`](../deep_worker/deep_worker_cli/README.md) | `deep-worker` / `dwcli` usage: storage, threads, history |
| [`deep_worker/deep_worker_cli/SECURITY.md`](../deep_worker/deep_worker_cli/SECURITY.md) | Security model (localhost-only) |
| [`deep_worker/README.md`](../deep_worker/README.md) | Core agent framework notes |
| [`dw_hist/README.md`](../dw_hist/README.md) | `dw-hist` — browse agent history |
| [`lsp/README.md`](../lsp/README.md) | LSP code navigation for agents |
| [`codex_provider/README.md`](../codex_provider/README.md) | Codex CLI provider |
| [`claude_code_provider/README.md`](../claude_code_provider/README.md) | Claude Code provider |
| [`grok_provider/README.md`](../grok_provider/README.md) | Grok CLI provider |

## Design and research

`DESIGN_*.md` in this directory are design documents for features and fixes,
one per topic. Historical research notes and internal operational instructions
of the development process are intentionally not shipped in this repository.

## Policies

- [`POLICIES.md`](POLICIES.md) — strict, non-negotiable project policies for
  contributors (human or agent). Reviewers must reject diffs that violate them.
