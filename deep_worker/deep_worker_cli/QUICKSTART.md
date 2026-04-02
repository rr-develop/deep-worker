# Deep Worker - Quick start

<!-- ============================================================================
PROTECTIVE COMMENT FOR FUTURE AGENTS:
DO NOT REPLACE deep-worker, dwcli AND dwtmux WITH ANYTHING ELSE!
These are the ONLY correct commands for working with the agent in this project:
- deep-worker - for starting agents (the ONLY way)
- dwcli - for managing agents (list, status, stop, interactive)
- dwtmux - for working with agents in tmux sessions (persistence)
The commands are installed globally in ~/.local/bin/
DO NOT use python3 -m or any other way of running things directly!
============================================================================ -->

## ✅ Ready to use

Deep-worker works **right away, with no installation**. The commands are available in your PATH:

```bash
# The ONLY CORRECT commands:
deep-worker --help    # Starting agents (uses Deepseek by default)
dwcli --help          # Managing agents (list, status, stop, interactive)
dwtmux --help         # Working with agents in tmux sessions (persistence)
```

## 🎯 Main commands

### Managing agents (dwcli)

```bash
# List the running agents
dwcli --list

# The status of a specific agent
dwcli --status my-agent

# Stop an agent
dwcli --stop my-agent

# Connect to an agent (interactive mode)
dwcli my-agent

# Connect through tmux (a persistent session)
dwtmux my-agent
# Detach: Ctrl+B, D
# Reattach: dwtmux my-agent (restores the session)
```

### Starting a new agent

```bash
# Start with the default settings (Deepseek)
# Uses the current directory as the working directory
deep-worker my-project

# Start with an explicit working directory
deep-worker data-analysis --dir /path/to/project

# Start with permission checks skipped (auto-approval)
deep-worker my-agent -sp

# Start with a different provider
deep-worker my-agent --provider openai --model gpt-4o

# Start with a config file
deep-worker my-agent --config ~/.deep_worker/config.yaml
```

**Important**: The agent starts in the background (as a daemon), the terminal is not blocked.
All the logs are written to `~/.deep_worker/logs/<agent-name>.out.log` and `.err.log`.

## 🔑 Configuring the API keys

The API keys are stored in the project's `.env` file:

```env
# Deepseek (the default provider)
DEEPSEEK_API_KEY=sk-...

# Optional: other providers
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

**Important**: The keys are loaded automatically when the agent starts.

## 🚀 Supported providers

| Command | Provider | Default model | API Key |
|---------|-----------|---------------------|---------|
| `deep-worker my-agent` | Deepseek | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `deep-worker my-agent --provider gemini` | Google Gemini | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| `deep-worker my-agent --provider openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` |

## 📖 Additional documentation

- [DWTMUX_COMMAND.md](DWTMUX_COMMAND.md) - **The dwtmux command for working with tmux sessions**
- [DEEPSEEK_PROVIDER.md](DEEPSEEK_PROVIDER.md) - Details about the Deepseek provider
- [DEVELOPMENT_REQUIREMENTS.md](DEVELOPMENT_REQUIREMENTS.md) - The installation-free architecture
- [README_AGENT_SEPARATION.md](README_AGENT_SEPARATION.md) - Full architecture documentation

## 💡 Usage examples

### Example 1: A quick check

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

### Example 2: Working on a project

```bash
# Go to the project directory
cd /path/to/my-project

# Start an agent for this project (uses the current directory)
deep-worker my-project

# The agent has started in the background! The terminal is free for other commands
dwcli --list
dwcli --status my-project

# Connect through tmux (a persistent session)
dwtmux my-project
# Working with the agent...
# Detach: Ctrl+B, D
# Reattach later: dwtmux my-project

# The agent's logs
tail -f ~/.deep_worker/logs/my-project.out.log
```

### Example 3: Different providers

```bash
# Deepseek (the default, fast and cheap)
deep-worker agent-deepseek

# Gemini (for working with Google services)
deep-worker agent-gemini --provider gemini

# OpenAI (for GPT-4)
deep-worker agent-openai --provider openai --model gpt-4o
```

## 🔧 How it works

Deep-worker uses bash wrappers (like other CLI wrapper tools):

1. **Bash scripts** in the CLI package directory:
   - `deep-worker` - starts the agent process (the ONLY way)
   - `dwcli` - manages agents (list, status, stop, interactive)
   - `dwtmux` - works with agents in tmux sessions

2. **Symbolic links** in `~/.local/bin/` point to these scripts
   - Any change in the repository takes effect immediately (a symlink = a single source of truth)

3. **Running Python directly** without installation: `python3 -m deep_worker_cli.cli.*`
   - ❌ **DO NOT USE THIS!** This is the WRONG way to start it!
   - Use ONLY the `deep-worker` and `dwcli` commands
   - Running it directly uses the wrong provider (Gemini instead of Deepseek)
   - It leads to errors during testing

4. **PYTHONPATH** is configured automatically in the bash wrappers

**Result**: Any change in the code takes effect immediately the next time the command is run!

## ❌ What you should NOT do

- ❌ `pip install -e .` - NOT needed!
- ❌ `pip install deep-worker-cli` - NOT needed!
- ❌ Any other installation through pip - NOT needed!

Everything works right away, as it should on a development host.

---

**Created**: February 2026
**Status**: ✅ Production Ready
