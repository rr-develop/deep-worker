# Deepseek Provider Configuration

## 🎯 Default Provider

**Deep-worker now uses the Deepseek provider by default!**

### Key changes

1. **Default model**: `deepseek-v4-flash` (instead of the legacy `deepseek-chat`)
2. **Default provider**: `deepseek`
3. **API key**: Automatically loaded from the `.env` file
4. **KV-cache isolation**: Each agent and its TaskManager use a distinct `user_id` to maximize DeepSeek API cache hits

> **Important**: The old `deepseek-chat` alias will be disabled on 24 July 2026.
> It is recommended to switch to `deepseek-v4-flash` immediately.

### 🔑 API Key Configuration

The Deepseek API key must be in the `.env` file:

```env
# Deepseek API Key
DEEPSEEK_API_KEY=sk-...
```

**✅ Security policy respected**: Keys are stored only in `.env` files (protected by `.gitignore`), never hardcoded in code.

### 📝 Usage

#### 1. Run with the default provider (Deepseek)

```bash
# Automatically uses deepseek-v4-flash with DEEPSEEK_API_KEY from .env
# The agent starts in the background; working directory = current directory
deep-worker my-agent

# Specify a different working directory
deep-worker my-agent --dir /path/to/project

# Skip the permission check (auto-approve dangerous operations)
deep-worker my-agent -sp
```

#### 2. Use a different provider

```bash
# Gemini
deep-worker my-agent --provider gemini --model gemini-2.5-flash

# OpenAI
deep-worker my-agent --provider openai --model gpt-4o
```

#### 3. Use the `dwcli` alias

```bash
# New short alias for deep-worker-cli
dwcli --list
dwcli --status my-agent
dwcli --stop my-agent
dwcli my-agent  # interactive mode (future)
```

### 🔧 CLI Commands

```bash
# List all running agents
dwcli --list

# Agent status
dwcli --status my-agent

# Stop an agent
dwcli --stop my-agent
```

### ⚙️ AgentConfig

AgentConfig now supports provider and api_key:

```python
from deep_worker_cli.agent_config_pkg import AgentConfig

# With automatic API key loading from .env
config = AgentConfig(
    model="deepseek-v4-flash",  # or None for the provider default
    working_dir="/path/to/work",
    provider="deepseek",  # default
)

# The API key is loaded automatically from DEEPSEEK_API_KEY
# Default model (if model=None): deepseek-v4-flash

# Or set the key explicitly (not recommended)
config = AgentConfig(
    model="deepseek-v4-flash",
    working_dir="/path/to/work",
    provider="deepseek",
    api_key="sk-...",  # Better to use .env
)
```

### 📋 Supported providers

| Provider | Default Model | API Key Env Var | Base URL |
|----------|--------------|-----------------|----------|
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `gemini` | `gemini-2.5-flash` | `GOOGLE_API_KEY` | - |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | - |

### ⚙️ Implementation: `ChatDeepSeek`, not `ChatOpenAI`

> **Dependency:** the provider requires the external package **`langchain-deepseek`**
> (`pip install langchain-deepseek`, installed globally). It is an external
> library, not part of the project — the CLAUDE.md `pip install` restriction
> concerns installing *the project itself*, not third-party dependencies.

`create_model(provider="deepseek")` creates
**`langchain_deepseek.ChatDeepSeek`** (NOT a bare `langchain_openai.ChatOpenAI`):

```python
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_base="https://api.deepseek.com",   # ⚠️ api_base, not base_url
    api_key=settings.deepseek_api_key,
    model_kwargs={"user": "<agent>"},       # only if user is set (see below)
)
```

**Why `ChatDeepSeek` specifically** (incident observed 2026-05-22): the default
model `deepseek-v4-flash` is a **thinking / reasoning model**. It
returns a `reasoning_content` field (the reasoning chain) alongside `content`, and
the DeepSeek API **requires `reasoning_content` to be passed back** on each
subsequent turn. A bare `ChatOpenAI` knows nothing about this non-standard field
and lost it when serialising the history → the next request was rejected with

```
400 - The `reasoning_content` in the thinking mode must be passed back to the API.
```

→ the agent fell into the ERROR state. `ChatDeepSeek` natively preserves
`reasoning_content` in `additional_kwargs` and re-sends it. **Do
not revert to `ChatOpenAI`.**

> ⚠️ `ChatDeepSeek` accepts `api_base` (not `base_url`) and does **not allow
> `model_kwargs=None`** (it iterates the dict) — so `model_kwargs`
> is passed only when there is a `user`; otherwise the key is omitted entirely.

### 🔒 KV-Cache Isolation (user_id)

**Problem**: Without a `user_id` in the DeepSeek API, different agents and their TaskManager threads share
a common cache and constantly evict each other's prefix cache. This leads to a cache miss
on every request, at a cost of $0.14/Mtok instead of $0.0028/Mtok.

**Solution**: Each thread gets its own `user_id` for cache isolation:

| Thread | user_id | thread_id |
|-------|---------|-----------|
| **Main agent** | `my_agent` | `my_agent` |
| **TaskManager (evaluator)** | `my_agent:tm` | `my_agent:tm` |

Under the hood `user_id` is passed as `model_kwargs={"user": ...}` to `ChatDeepSeek`, which
maps to the `user` field of the DeepSeek API request. The mechanism is described in
[DESIGN_CACHE_USER_ID_AND_MODEL_UPDATE.md](../DESIGN_CACHE_USER_ID_AND_MODEL_UPDATE.md).

### 🔒 Security

**All API keys:**
- ✅ Stored in `.env` files
- ✅ Protected by `.gitignore`
- ✅ NEVER committed to Git
- ✅ Automatically loaded with `python-dotenv`
- ❌ NEVER hardcoded in code

**Compliance**: documented in the project's API key security policy (not included here).

### 🧪 Testing

All tests were updated for the new provider:
- AgentConfig tests: provider + api_key validation
- CLI tests: --provider parameter, user_id passing
- Context limits: deepseek-v4-flash and deepseek-v4-pro (1M tokens)
- All 1453 tests pass ✅

### 📦 Quick start

The commands work immediately without installing via pip:

```bash
# Check that the alias works
dwcli --help

# List agents
dwcli --list
```

**Important**: Deep-worker runs directly through bash wrappers, without installing via pip.

### 🎯 Examples

```bash
# 1. Run with Deepseek (default, model deepseek-v4-flash)
# Runs in the background, uses the current directory
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

# 7. View logs
tail -f <path-to-the-agent-log>
```

### 📖 Additional documentation

- [DESIGN_CACHE_USER_ID_AND_MODEL_UPDATE.md](../DESIGN_CACHE_USER_ID_AND_MODEL_UPDATE.md) —
  design document for `user_id` cache isolation and model switching
- [QUICKSTART.md](QUICKSTART.md) — Quick start

---

**Updated**: May 2026
**Version**: 2.0
**Status**: ✅ Production Ready
