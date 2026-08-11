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
# It sweeps every sessions_*.db under ~/.deepagents/ AND every
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

## 📖 Resources

- Root [README](../../README.md) — component map and quick start
- [docs/](../../docs/README.md) — documentation index
- [SECURITY.md](SECURITY.md) — security model
- [Upstream](https://github.com/langchain-ai/deepagents) — the `deepagents-cli` project this is forked from
