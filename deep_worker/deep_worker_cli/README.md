# Deep Worker CLI

`deep-worker` is a Python package that runs LangChain agents as background
daemon processes and exposes them to the interactive `dwcli` client.

This is a fork of [LangChain's `deepagents-cli`](https://github.com/langchain-ai/deepagents),
renamed to `deep-worker` and extended with additional providers, tools, and
infrastructure (see the root [README](../../README.md) for the component map).

## Quick Install

There are two supported ways to use `deep-worker`:

**1. From a source checkout (recommended for development)** — no `pip install`:

```bash
# Point the launchers into your PATH (project policy: no virtual environments)
ln -sf /path/to/checkout/deep_worker/deep_worker_cli/deep-worker ~/.local/bin/deep-worker
ln -sf /path/to/checkout/deep_worker/deep_worker_cli/dwcli ~/.local/bin/dwcli
```

**2. From a built wheel (`pip install deep-worker`)** — for consuming the
package without the source tree. See
[`docs/BUILD_INSTALL_GUIDE.md`](docs/BUILD_INSTALL_GUIDE.md) for build,
check, and install instructions. Note the wheel's limitations
(`--import-claude` is unavailable; `deepagents` is vendored, so the official
PyPI `deepagents` must not be co-installed).

See [`deep_worker/README.md`](../../deep_worker/README.md) for the no-venv
runtime model.

## Providers & Capabilities (installed wheel)

The wheel ships its five provider packages **inside** the `deep_worker`
namespace — never as top-level packages (avoids collisions with real PyPI
distributions such as `claude-code-sdk-provider` / `semantic-search`):

```python
from deep_worker.providers.claude_code_sdk_provider import ClaudeCodeChatModel
from deep_worker.providers.semantic_search import SemanticSearchIndexer
```

Optional capabilities are gated behind extras and reported with actionable
`CapabilityError` remedies when missing:

| Extra / capability | Install | When missing |
|---|---|---|
| Vertex AI | `pip install "deep-worker[vertexai]"` | `pip install deep-worker[vertexai]` |
| Semantic search | `pip install "deep-worker[semantic-search]"` | `pip install deep-worker[semantic-search]` |
| A2A protocol | `pip install "deep-worker[a2a]"` | `pip install deep-worker[a2a]` |
| Claude Code / web-search delegation | SDK **fork** via `CLAUDE_AGENT_SDK_SRC` | set `CLAUDE_AGENT_SDK_SRC`; official SDK → `sdk_incompatible` |

The universal `web_search` delegation tool works with any LLM provider
(tri-state policy): default `auto` skips it with a single warning when the SDK
fork is absent; `--enable-web-search` fails loudly; `--disable-web-search`
never touches the SDK. `DEEP_WORKER_SOURCE_ROOT` / `DEEP_WORKER_PROVIDER_SOURCE`
force source-checkout provider resolution (development). Full details:
[`docs/BUILD_INSTALL_GUIDE.md`](docs/BUILD_INSTALL_GUIDE.md).

## Key Features

- 🚀 **Background Agent Processes**: Run agents as daemon processes with `deep-worker`
- 💬 **Interactive CLI**: Connect to running agents with `dwcli`
- 🧹 **Clean History**: Start agents with fresh state using `--clean` option
- 🔧 **Customizable**: Support for custom prompts, skills, and configurations
- 📊 **Session Management**: Conversation history and thread management with `-rm`, `--history`, `--remove-before`
- 💾 **Flexible Storage**: Choose between persistent SQLite or in-memory storage

## 🔒 Security Notice

**⚠️ IMPORTANT: deep-worker-cli is designed for LOCAL TRUSTED ENVIRONMENTS ONLY**

- All network services bind to **localhost (127.0.0.1) only**
- **NO external network access** - ports are not exposed outside your machine
- No authentication/authorization - relies on localhost isolation
- Suitable for personal development, NOT for production servers
- See [SECURITY.md](SECURITY.md) for detailed security requirements

**Never expose deep-worker ports to external networks!**

## 🤔 What is this?

Using an LLM to call tools in a loop is the simplest form of an agent. This architecture, however, can yield agents that are "shallow" and fail to plan and act over longer, more complex tasks.

Applications like "Deep Research", "Manus", and "Claude Code" have gotten around this limitation by implementing a combination of four things: a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

`deep-worker` implements these in a general purpose way so that you can easily create a Deep Agent for your application.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## 💾 Storage Options

By default, agents persist conversation history to SQLite database. You can control storage behavior:

```bash
# Default: SQLite persistence (production)
deep-worker my_agent

# Temporary in-memory storage (no persistence)
deep-worker --memory-only temp_agent

# Force SQLite even in test environment
deep-worker --use-sqlite test_agent
```

**List all conversation threads:**
```bash
# List all threads
deep-worker --list

# List with message counts (slower)
deep-worker --list --with-counts

# Filter by agent name
deep-worker --list --agent my_agent

# JSON output
deep-worker --list --format json
```

**Thread management commands:**
```bash
# Remove a specific thread
deep-worker -rm thread_id

# Dry run (preview without deleting)
deep-worker -rm thread_id --dry-run

# View thread history (works offline)
deep-worker --history thread_id

# View history as JSON
deep-worker --history thread_id --format json

# ⚠️ DANGER: --remove-before is GLOBAL and DESTRUCTIVE.
# It sweeps every per-agent checkpoint DB under ~/.deep_worker/ AND every
# ~/.deep_worker/<agent>/summaries/<thread>.md across ALL agents for the
# current $HOME. NEVER run this command from an ad-hoc shell / integration
# test without first isolating $HOME to a temp directory (see the
# isolated_home / fake_home fixtures under tests/).
#
# Safety guard: if DATE resolves to a future timestamp, the command
# refuses by default (exit code 2) and requires --force to proceed.
# Future cutoffs match every row and are almost always bugs.
#
# Always start with --dry-run.

# Remove old conversation data (threads AND offloaded summary sections)
# before a specific date. See the Summarization storage section below
# for what "summary sections" means.
deep-worker --remove-before 2026-01-01

# Remove data older than 30 days
deep-worker --remove-before 30d

# Other relative formats: 7d, 2w, 3m, 1y
deep-worker --remove-before 1y --dry-run
```

**Summarization storage — find and manage offloaded conversation history:**

When the model's context fills up, `SummarizationMiddleware` evicts older
messages and writes them to an on-disk file so the agent can later refer
back to them. Files live at:

```
~/.deep_worker/<agent_name>/summaries/<thread_id>.md
```

Each file is an append-only log with one `## Summarized at <ISO-timestamp>`
section per summarization event. The path quoted in the synthetic summary
message injected into the agent's history is exactly this absolute path —
you can `cat`/`grep`/`less` it directly. See
`deep_worker/docs/DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md`.

```bash
# List all offloaded summaries (text table)
deep-worker --list-summaries

# Filter by agent
deep-worker --list-summaries --agent my_agent

# Machine-readable JSON output
deep-worker --list-summaries --format json
```

The `--remove-before DATE` command also prunes summary files:
- If **every** section in a summary file is older than DATE → the file is
  deleted.
- If **some** are older and **some** are newer → the file is atomically
  rewritten (via `<file>.tmp` + `os.replace`) keeping only the newer
  sections.
- If **all** are newer → the file is untouched.

**Note:** Tests automatically use in-memory storage by default for faster execution and test isolation.

## 🏁 Turn-ending tools

Some tools *are* the agent's answer for a turn: once they succeed there is
nothing useful left for the model to say, and calling it again only produces
filler text. `send_message_to_agent` works this way out of the box — an
outgoing inter-agent message is the agent's response for that turn.

Any tool can be given the same behavior by declaring it explicitly:

```python
from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.middleware import TurnEndingTool

agent, backend = create_cli_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    assistant_id="extractor-42",
    project_memory_dir=work_dir,
    tools=[submit_fragments_tool],
    turn_ending_tools=[
        TurnEndingTool(
            tool_name="submit_fragments",
            is_success=lambda result: result.get("document_closed") is True,
            echo_field=None,
        ),
    ],
)
```

`TurnEndingTool` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `tool_name` | — | Exact tool name, compared with `==` against `ToolMessage.name`. |
| `is_success` | `result["success"]` is truthy | Predicate over the parsed tool result deciding whether this call ends the turn. |
| `echo_field` | `"delivered_content"` | Field whose non-empty string value becomes the final assistant message. `None` ends the turn without emitting any message. |

Notes:

- **Registration is explicit — a tool never becomes turn-ending because of its
  name.** There is no naming convention and no prefix matching, so an MCP
  server cannot alter an agent's control flow by choosing a suggestive tool
  name.
- `send_message_to_agent` is registered automatically; re-declaring it (or any
  name twice) raises `DuplicateTurnEndingToolError` at startup rather than
  being silently ignored.
- With `echo_field=None` the turn ends on the tool result itself and no
  assistant message is created — appropriate when the tool's output is not
  conversational. Use `echo_field` when the caller should see an answer.
- If the model made **parallel** tool calls, the turn is not ended, so it can
  see every result — including any sibling failure.
- A failing `is_success` predicate is treated as "not successful" and logged;
  it never stalls the agent.

### Declaring turn-ending tools from the CLI

Agents started with the stock `deep-worker` binary have no launcher code of
their own, so they use the repeatable `--turn-ending-tool` flag instead:

```bash
deep-worker rg_extractor --provider deepseek --model deepseek-chat \
    --dir /path/to/workdir -m -sp \
    --turn-ending-tool repo_graph__submit_fragments:document_closed
```

| Value | Predicate |
| --- | --- |
| `<name>` | ends the turn when `success` is truthy |
| `<name>:<field>` | ends the turn only when that top-level field is exactly `True` |

`<field>` is matched with `.get()`, so a missing key and `false` behave
identically — the turn simply continues. The check is `is True`, not
truthiness: `1`, `"true"` and `["yes"]` do NOT end the turn. CLI-declared
tools are always jump-only (`echo_field=None`); no assistant message is
injected.

`<field>` must be a single top-level key: a dot in the FIELD (`tool:a.b`)
is rejected, because nested paths are out of scope. A dot in the tool NAME
is fine (`a.b:c` is a valid spec for a tool literally named `a.b`) — only
the part after the `:` is constrained. Whitespace is rejected anywhere in
the spec rather than trimmed, so a stray space from shell quoting fails
loudly instead of silently producing a second, near-identical declaration.

⚠️ **`<name>` is the graph-side tool name, not the form the model calls.**
For a tool exposed by an external MCP server, that is the mangled
`<server>__<tool>` name — server `repo_graph` exposing `submit_fragments`
becomes `repo_graph__submit_fragments`. Do not pass the provider-visible
`mcp__…` form; the middleware matches on `ToolMessage.name`, which carries
the graph name.

The flag is repeatable, and each start logs one machine-greppable line per
declaration, so a launcher can prove its flag arrived:

```
turn_ending_tool_registered name=repo_graph__submit_fragments
turn_ending_tool_not_found name=repo_graph__submit_fragment candidates=['repo_graph__submit_fragments', ...]
```

A name that is not in the final tool inventory is a WARNING, never a startup
failure — an external MCP endpoint may come up after the agent, and the
declaration still matches if the tool appears later.

Declarations are persisted to `session_metadata.json` and survive
`--restart`. Passing `--turn-ending-tool` together with `--restart`
REPLACES the saved list wholesale (same semantics as `--mcp-config`);
passing nothing inherits it. Declaring the same name twice — including a
bare `foo` alongside `foo:done` — or declaring the built-in
`send_message_to_agent` fails at startup before the agent is constructed.

See [`docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md`](docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md)
and [`docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`](docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md)
for the full contract.

## 📖 Resources

- Root [README](../../README.md) — component map and quick start
- [docs/](../../docs/README.md) — documentation index
- [SECURITY.md](SECURITY.md) — security model
- [Upstream](https://github.com/langchain-ai/deepagents) — the `deepagents-cli` project this is forked from
