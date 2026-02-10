# Deep Worker - Quick Start

## ✅ Ready to use

Deep-worker works **right away, without installation**. The commands are available in your PATH:

```bash
dwcli --help          # Short alias (recommended)
deep-worker --help    # Agent launcher
deep-worker-cli --help # Client connector (full name)
```

## 🎯 Main commands

### Managing agents (dwcli)

```bash
# List running agents
dwcli --list

# Status of a specific agent
dwcli --status my-agent

# Stop an agent
dwcli --stop my-agent

# Connect to an agent (interactive mode — future)
dwcli my-agent
```

### Starting a new agent

```bash
# Start with the default settings (Deepseek)
# Uses the current directory as the working directory
deep-worker my-project

# Start with an explicit working directory
deep-worker data-analysis --dir /path/to/project

# Start while skipping the permission check (auto-confirmation)
deep-worker my-agent -sp

# Start with a different provider
deep-worker my-agent --provider openai --model gpt-4o

# Start with a config file
deep-worker my-agent --config ~/.deep_worker/config.yaml
```

**Important**: the agent starts in background (daemon) mode, the terminal is not blocked.
All logs are written to `~/.deep_worker/logs/<agent-name>.out.log` and `.err.log`.

## 🔑 Configuring API keys

API keys are stored in the project's `.env` file:

```env
# Deepseek (default provider)
DEEPSEEK_API_KEY=sk-...

# Optional: other providers
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

**Important**: the keys are loaded automatically when the agent starts.

## 🚀 Supported providers

| Command | Provider | Default model | API Key |
|---------|-----------|---------------------|---------|
| `deep-worker my-agent` | Deepseek | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `deep-worker my-agent --provider gemini` | Google Gemini | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| `deep-worker my-agent --provider openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` |

## 📖 Additional documentation

- [DEEPSEEK_PROVIDER.md](DEEPSEEK_PROVIDER.md) — details about the Deepseek provider
- [DEVELOPMENT_REQUIREMENTS.md](DEVELOPMENT_REQUIREMENTS.md) — the no-install architecture
- [README_AGENT_SEPARATION.md](README_AGENT_SEPARATION.md) — full architecture documentation

## 💡 Usage examples

### Example 1: Quick check

```bash
# Check that the commands work
dwcli --list
# Output: No running agents found

# Start an agent
deep-worker test-agent &

# Check the status
dwcli --status test-agent

# Stop it
dwcli --stop test-agent
```

### Example 2: Working with a project

```bash
# Go to the project directory
cd /path/to/my-project

# Start an agent for this project (uses the current directory)
deep-worker my-project

# The agent started in the background! The terminal is free for other commands
dwcli --list
dwcli --status my-project

# Agent logs
tail -f ~/.deep_worker/logs/my-project.out.log
```

### Example 3: Different providers

```bash
# Deepseek (default, fast and cheap)
deep-worker agent-deepseek

# Gemini (for working with Google services)
deep-worker agent-gemini --provider gemini

# OpenAI (for GPT-4)
deep-worker agent-openai --provider openai --model gpt-4o
```

## 🔧 How it works

Deep-worker uses bash wrappers (like a `cco`-style launcher in comparable CLI tools):

1. **Bash scripts** in the `deep_worker_cli` project directory:
   - `deep-worker` — starts the agent process
   - `dwcli` — manages agents (short alias)
   - `deep-worker-cli` — the same as dwcli (full name)

2. **Symbolic links** in `~/.local/bin/` point to these scripts

3. **Direct Python execution** without installation: `python3 -m deep_worker_cli.cli.*`

4. **PYTHONPATH** is configured automatically in the bash wrappers

**Result**: any code change takes effect immediately on the next command run!

## ❌ What you should NOT do

- ❌ `pip install -e .` — NOT needed!
- ❌ `pip install deep-worker-cli` — NOT needed!
- ❌ Any other pip-based installation — NOT needed!

Everything works right away, as it should on a development host.

---

**Created**: February 2026
**Status**: ✅ Production Ready
