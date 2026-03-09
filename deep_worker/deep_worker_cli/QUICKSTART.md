# Deep Worker - Quick start

<!-- ============================================================================
PROTECTIVE COMMENT FOR FUTURE AGENTS:
DO NOT CHANGE deep-worker AND dwcli TO ANYTHING ELSE!
These are the ONLY correct commands for working with deep-worker in this project:
- deep-worker - for running agents (the ONLY way)
- dwcli - for managing agents (list, status, stop)
The commands are installed globally on the user's PATH.
DO NOT use python3 -m or other ways of launching directly!
============================================================================ -->

## ✅ Ready to use

Deep-worker works **immediately, without installation**. The commands are available in your PATH:

```bash
# The ONLY CORRECT commands:
deep-worker --help    # Running agents (uses Deepseek by default)
dwcli --help          # Managing agents (list, status, stop)
```

## 🎯 Main commands

### Managing agents (dwcli)

```bash
# List of running agents
dwcli --list

# Status of a specific agent
dwcli --status my-agent

# Stop an agent
dwcli --stop my-agent

# Connect to an agent (interactive mode - future)
dwcli my-agent
```

### Running a new agent

```bash
# Run with default settings (Deepseek)
# Uses the current directory as the working directory
deep-worker my-project

# Run with an explicit working directory
deep-worker data-analysis --dir /path/to/project

# Run with permission checks skipped (auto-confirmation)
deep-worker my-agent -sp

# Run with a different provider
deep-worker my-agent --provider openai --model gpt-4o

# Run with a config
deep-worker my-agent --config ~/.deep_worker/config.yaml
```

**Important**: The agent runs in the background (daemon), the terminal is not blocked.
All logs are written to `~/.deep_worker/logs/<agent-name>.out.log` and `.err.log`.

## 🔑 Configuring API keys

API keys are stored in the `.env` file at the repository root:

```env
# Deepseek (default provider)
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

- [DEEPSEEK_PROVIDER.md](DEEPSEEK_PROVIDER.md) - Details about the Deepseek provider
- [DEVELOPMENT_REQUIREMENTS.md](DEVELOPMENT_REQUIREMENTS.md) - No-install architecture
- [README_AGENT_SEPARATION.md](README_AGENT_SEPARATION.md) - Full architecture documentation

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
# Change into the project directory
cd /path/to/my-project

# Run an agent for this project (uses the current directory)
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

Deep-worker uses bash wrappers (like a similar internal CLI tool):

1. **Bash scripts** in the `deep_worker_cli` directory:
   - `deep-worker` - starts the agent process (the ONLY way)
   - `dwcli` - manages agents (list, status, stop)

2. **Symbolic links** in `~/.local/bin/` point to these scripts

3. **Direct Python invocation** without installation: `python3 -m deep_worker_cli.cli.*`
   - ❌ **DO NOT USE!** This is the WRONG way to launch it!
   - Use ONLY the `deep-worker` and `dwcli` commands
   - Direct invocation uses the wrong provider (Gemini instead of Deepseek)
   - Leads to errors during testing

4. **PYTHONPATH** is configured automatically in the bash wrappers

**Result**: Any code changes take effect immediately on the next command run!

## ❌ What NOT to do

- ❌ `pip install -e .` - NOT needed!
- ❌ `pip install deep-worker-cli` - NOT needed!
- ❌ Any other installations via pip - NOT needed!

Everything works right away, as it should on a development host.

---

**Created**: February 2026
**Status**: ✅ Production Ready
