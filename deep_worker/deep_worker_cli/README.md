# 🧠🤖 Deep Agents CLI

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents-cli?label=%20)](https://pypi.org/project/deepagents-cli/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents-cli)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents-cli)](https://pypistats.org/packages/deepagents-cli)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [Deep Agents CLI.js](https://github.com/langchain-ai/deepagentsjs).

To help you ship LangChain apps to production faster, check out [LangSmith](https://smith.langchain.com).
LangSmith is a unified developer platform for building, testing, and monitoring LLM applications.

<p align="center">
  <img src="https://raw.githubusercontent.com/langchain-ai/deepagents/master/libs/cli/images/cli.png" alt="Deep Agents CLI" width="600"/>
</p>

## Quick Install

```bash
uv tool install deepagents-cli
deepagents
```

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

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application. For a full overview and quickstart of Deep Agents, the best resource is our [docs](https://docs.langchain.com/oss/python/deepagents/overview).

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

# Remove old threads (before specific date)
deep-worker --remove-before 2026-01-01

# Remove threads older than 30 days
deep-worker --remove-before 30d

# Other relative formats: 7d, 2w, 3m, 1y
deep-worker --remove-before 1y --dry-run
```

**Note:** Tests automatically use in-memory storage by default for faster execution and test isolation.

## 📖 Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents/cli)** — Full documentation
- **[Deep Agents](https://github.com/langchain-ai/deepagents)** — The underlying agent harness
- **[Chat LangChain](https://chat.langchain.com)** - Chat interactively with the docs

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
# Test cleanup
