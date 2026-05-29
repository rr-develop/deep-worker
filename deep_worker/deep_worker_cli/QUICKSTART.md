# Deep Worker - Quick start

<!-- ============================================================================
PROTECTIVE COMMENT FOR FUTURE AGENTS:
DO NOT change deep-worker, dwcli and dwtmux to anything else!
These are the ONLY correct commands for working with deep-worker in this project:
- deep-worker - for starting agents (the ONLY way)
- dwcli - for managing agents (list, status, stop, interactive)
- dwtmux - for working with agents in tmux sessions (persistence)
The commands are installed globally in a bin directory on the PATH
DO NOT use python3 -m or any other direct launch method!
============================================================================ -->

## ✅ Ready to use

Deep-worker works **immediately, without installation**. The commands are available in your PATH:

```bash
# The ONLY correct commands:
deep-worker --help    # Start agents (uses Deepseek by default)
dwcli --help          # Manage agents (list, status, stop, interactive)
dwtmux --help         # Work with agents in tmux sessions (persistence)
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

# Connect to an agent (interactive mode)
dwcli my-agent

# Connect via tmux (persistent session)
dwtmux my-agent
# Detach: Ctrl+B, D
# Reconnect: dwtmux my-agent (restores the session)
```

### Starting a new agent

```bash
# Start with default settings (Deepseek)
# Uses the current directory as the working directory
deep-worker my-project

# Start with a specified working directory
deep-worker data-analysis --dir /path/to/project

# Start with the permission check skipped (auto-approve)
deep-worker my-agent -sp

# Start with a different provider
deep-worker my-agent --provider openai --model gpt-4o

# Start with a config file
deep-worker my-agent --config /path/to/config.yaml
```

**Important**: The agent starts in the background (daemon), the terminal is not blocked.
All logs are written to the agent's log files (`<agent-name>.out.log` and `.err.log`) under its data directory.

## 🔑 Setting up API keys

API keys are stored in a `.env` file:

```env
# Deepseek (default provider)
DEEPSEEK_API_KEY=sk-...

# Optional: other providers
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

**Important**: Keys are loaded automatically when an agent starts.

## 🚀 Supported providers

| Command | Provider | Default model | API Key |
|---------|-----------|---------------------|---------|
| `deep-worker my-agent` | Deepseek | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `deep-worker my-agent --provider gemini` | Google Gemini | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| `deep-worker my-agent --provider openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` |

## 📖 Additional documentation

- [DWTMUX_COMMAND.md](DWTMUX_COMMAND.md) - **The dwtmux command for working with tmux sessions**
- [DEEPSEEK_PROVIDER.md](DEEPSEEK_PROVIDER.md) - Details about the Deepseek provider
- [DEVELOPMENT_REQUIREMENTS.md](DEVELOPMENT_REQUIREMENTS.md) - Architecture without installation
- [README_AGENT_SEPARATION.md](README_AGENT_SEPARATION.md) - Full architecture documentation

## 💡 Usage examples

### Example 1: Quick check

```bash
# Check that the commands work
dwcli --list
# Output: No running agents found

# Start an agent
deep-worker test-agent &

# Check status
dwcli --status test-agent

# Stop
dwcli --stop test-agent
```

### Example 2: Working with a project

```bash
# Change to the project directory
cd /path/to/my-project

# Start an agent for this project (uses the current directory)
deep-worker my-project

# The agent started in the background! The terminal is free for other commands
dwcli --list
dwcli --status my-project

# Connect via tmux (persistent session)
dwtmux my-project
# Work with the agent...
# Detach: Ctrl+B, D
# Reconnect later: dwtmux my-project

# Agent logs
tail -f <path-to-the-agent-log>
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

Deep-worker uses bash wrappers:

1. **Bash scripts** in the `deep_worker/deep_worker_cli/` directory:
   - `deep-worker` - starts the agent process (the ONLY way)
   - `dwcli` - manages agents (list, status, stop, interactive)
   - `dwtmux` - works with agents in tmux sessions

2. **Symlinks** in a local bin directory point to these scripts
   - Any changes in the repository take effect immediately (symlink = single source of truth)

3. **Direct Python launch** without installation: `python3 -m deep_worker_cli.cli.*`
   - ❌ **DO NOT USE!** This is the WRONG way to launch!
   - Use ONLY the `deep-worker` and `dwcli` commands
   - Direct launch uses the wrong provider (Gemini instead of Deepseek)
   - Leads to errors during testing

4. **PYTHONPATH** is configured automatically in the bash wrappers

**Result**: Any code changes take effect immediately on the next command run!

## ❌ What NOT to do

- ❌ `pip install -e .` - NOT needed!
- ❌ `pip install deep-worker-cli` - NOT needed!
- ❌ Any other pip installs - NOT needed!

Everything works immediately, as it should on a development host.

---

**Created**: February 2026
**Status**: ✅ Production Ready
