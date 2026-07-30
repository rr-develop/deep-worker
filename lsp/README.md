# LSP (Language Server Protocol) for deep-worker agents

Precise code navigation for agents: go-to-definition, show-usages,
find-symbol, outline, symbol refactoring — **in a single call**, without
`grep → read_file → grep` chains.

Optional capability: acts as a semantic index, enabled per project. If not
enabled, agent behavior is unchanged.

## Contents

| Path | Purpose |
|------|---------|
| [`docs/`](docs/) | Integration design |
| [`research/`](research/) | OSS research and findings |

## How it works

The backend is [Serena](https://github.com/oraios/serena) (MIT) — an
MCP server on top of language servers (40+ languages, Python and Kotlin
included). It connects through the **existing** deep-worker External MCP:
Serena lives as a separate stdio process, and its tools arrive to the
agent as regular `BaseTool`s. We don't write our own LSP client.

```
Agent → serena__find_symbol → Serena (MCP) → pyright / kotlin-lsp → project
```

## Installation

```bash
uv tool install -p 3.13 serena-agent
```

After that, `serena`, `serena-agent`, `serena-hooks` appear in PATH.

## Enabling

```bash
deep-worker lsp enable     # for the current project (default)
deep-worker lsp status     # status + binary presence
deep-worker lsp disable    # disable
```

The scope can be set explicitly:

```bash
deep-worker lsp enable --scope user            # for all projects
deep-worker lsp enable --scope agent:my_agent  # for a single agent
deep-worker lsp enable --config /path/mcp.json --project-root /path/repo
```

The command appends a `serena` entry to the corresponding `mcp.json`
(file `0600`, directory `0700`), leaving other MCP servers untouched.
Repeated calls are idempotent; to overwrite the entry use `--force`.

**Restart the agent** for the change to take effect.

## What the agent gets

| Tool | IDE equivalent |
|------|----------------|
| `serena__find_symbol` | find symbol by name |
| `serena__find_referencing_symbols` | Show usages / Find references |
| `serena__find_declaration` | Go to definition |
| `serena__find_implementations` | Go to implementations |
| `serena__get_symbols_overview` | file outline |
| `serena__get_diagnostics_for_file` | Problems |
| `serena__rename_symbol` | Rename refactoring |
| `serena__replace_symbol_body` | replace symbol body |

Names are mangled (`.` → `__`) — that's how the agent sees them. When LSP
is enabled, a short snippet is added to the system prompt explaining when
to prefer a symbol tool over `grep`.

Serena tools are marked `auto_approve` — no HITL is requested for them
(including writing ones). Disable with:
`deep-worker lsp enable --no-auto-approve`.

## Languages

- **Python** — pyright, works out of the box.
- **Kotlin** — official `Kotlin/kotlin-lsp`, Serena downloads it itself;
  first start is noticeably slower (JVM).
- The other 40+ languages — as needed, through the same unified interface.

## Highlights

- **The cost is paid at agent startup, not at the first call.** The
  language server starts when the MCP starts (~3.6 s on a Python project —
  measured in §11 of the design), after which the calls themselves are
  fast (~1 s). Kotlin/JVM is more expensive at startup. This is not a hang.
- **`.serena/` at the project root.** Serena itself creates `project.yml`,
  a cache and its own `.gitignore` there. It's the tool's internal
  directory; we don't commit it to git.
- **Complements, doesn't replace.** Semantic search answers "roughly where
  auth is", LSP answers "where `AuthMiddleware.authenticate` is declared
  and who calls it". `grep` remains for text, configs and logs.

## Tests

```bash
# unit
python3 -m pytest tests/unit_tests/lsp/ -q

# integration (real Serena, no mocks; skipped without the binary)
python3 -m pytest tests/integration_tests/lsp/ -q -m integration
```

In strict mode (`DEEP_WORKER_REQUIRE_INTEGRATION=1`) a missing `serena`
binary fails the test instead of skipping it.

## Documents

- [Integration design](docs/DESIGN_LSP_MCP_INTEGRATION.md)
- [OSS research](research/RESEARCH_LSP_MCP.md)
