# Deepseek Provider Configuration

## 🎯 Default Provider

**Deep-worker now uses the Deepseek provider by default!**

### Key changes

1. **Default model**: `deepseek-chat` (instead of claude-sonnet)
2. **Default provider**: `deepseek`
3. **API key**: loaded automatically from the `.env` file

### 🔑 Configuring the API key

The Deepseek API key must be present in the project's `.env` file:

```env
# Deepseek API Key
DEEPSEEK_API_KEY=<your-deepseek-api-key>
```

**✅ The security policy is followed**: keys are stored only in `.env` files (protected by `.gitignore`) and are never hardcoded in the code.

### 📝 Usage

#### 1. Running with the default provider (Deepseek)

```bash
# Automatically uses deepseek-chat with DEEPSEEK_API_KEY from .env
# The agent starts in the background, working directory = current directory
deep-worker my-agent

# Specify a different working directory
deep-worker my-agent --dir /path/to/project

# Skip the permission check (auto-confirm dangerous operations)
deep-worker my-agent -sp
```

#### 2. Using a different provider

```bash
# Gemini
deep-worker my-agent --provider gemini --model gemini-2.5-flash

# OpenAI
deep-worker my-agent --provider openai --model gpt-4o
```

#### 3. Using the `dwcli` alias

```bash
# New short alias for deep-worker-cli
dwcli --list
dwcli --status my-agent
dwcli --stop my-agent
dwcli my-agent  # interactive mode (future)
```

### 🔧 CLI commands

```bash
# List all running agents
dwcli --list

# Agent status
dwcli --status my-agent

# Stop an agent
dwcli --stop my-agent
```

### ⚙️ AgentConfig

AgentConfig now supports `provider` and `api_key`:

```python
from deep_worker_cli.agent_config_pkg import AgentConfig

# With automatic loading of the API key from .env
config = AgentConfig(
    model="deepseek-chat",
    working_dir="/path/to/work",
    provider="deepseek",  # default
)

# The API key is loaded from DEEPSEEK_API_KEY automatically

# Or specify the key explicitly (not recommended)
config = AgentConfig(
    model="deepseek-chat",
    working_dir="/path/to/work",
    provider="deepseek",
    api_key="sk-...",  # Better to use .env
)
```

### 📋 Supported providers

| Provider | Default Model | API Key Env Var | Base URL |
|----------|--------------|-----------------|----------|
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| `gemini` | `gemini-2.5-flash` | `GOOGLE_API_KEY` | - |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | - |

### 🔒 Security

**All API keys:**
- ✅ Are stored in `.env` files
- ✅ Are protected by `.gitignore`
- ✅ Are NEVER committed to Git
- ✅ Are loaded automatically with `python-dotenv`
- ❌ Are NEVER hardcoded in the code

**Compliance**: follows the project's API key security policy.

### 🧪 Testing

All tests were updated to work with the new provider:
- AgentConfig tests: provider + api_key validation
- CLI tests: `--provider` parameter
- All 843 tests pass ✅

### 📦 Quick start

The commands work immediately, without installing via pip:

```bash
# Check that the alias works
dwcli --help

# List agents
dwcli --list
```

**Important**: Deep-worker is launched directly through bash wrappers (like some comparable CLI tools), without installation via pip.

### 🎯 Examples

```bash
# 1. Run with Deepseek (default)
# Starts in the background, uses the current directory
deep-worker project-alpha

# 2. Run in a specific directory with -sp
cd /my/project && deep-worker project-beta -sp

# 3. List agents (short alias)
dwcli --list

# 4. Agent status
dwcli --status project-alpha

# 5. Stop an agent
dwcli --stop project-alpha

# 6. Run with a different provider
deep-worker project-gamma --provider gemini --model gemini-2.5-flash

# 7. View the logs
tail -f ~/.deep_worker/logs/project-alpha.out.log
```

---

**Updated**: February 2026
**Version**: 1.0
**Status**: ✅ Production Ready
