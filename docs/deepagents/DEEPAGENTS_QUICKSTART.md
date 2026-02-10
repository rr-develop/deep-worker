# Deep Agents CLI - Quick Start Guide

## What is this?

**deepagents-cli** is an interactive CLI tool that works as an AI assistant with the ability to:
- 🤖 Interactively communicate with an AI model
- 🛠️ Execute shell commands
- 📝 Work with files
- 🔄 Manage sessions (conversation threads)
- 🎯 Sub-agents (creating helper agents)

**Installed with support for the providers:**
- OpenAI (GPT-4, GPT-4O, etc.)
- Anthropic (Claude)
- Google Gemini
- VertexAI
- **Deepseek** (via the OpenAI-compatible interface) ✨

---

## Quick start (Deepseek)

### Option 1: Using the wrapper script (recommended)

```bash
# Full path
./venv/bin/deepagents-deepseek

# Or if you are inside the project
source venv/bin/activate
deepagents-deepseek
```

### Option 2: Using the standard deepagents with a model override

```bash
./venv/bin/deepagents --model deepseek-chat
```

### Option 3: Through environment variables

```bash
export OPENAI_API_KEY=sk-...  # Deepseek API Key
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_MODEL=deepseek-chat

deepagents
```

---

## Main commands

### Interactive commands (inside deepagents)

| Command | Description |
|---------|----------|
| `help` | Show help for the available commands |
| `clear` | Clear the screen and reset the conversation |
| `quit` / `exit` | Exit the CLI |
| `remember` | Re-read the conversation and update the memory |
| `tokens` | Show the tokens used |

### CLI arguments (before starting)

```bash
deepagents [OPTIONS]

Options:
  --agent NAME              Agent name (default: agent)
  --model MODEL            Which model to use (deepseek-chat, gpt-4o, etc.)
  --auto-approve           Auto-approve tools without confirmation
  --sandbox TYPE           Remote sandbox (modal, runloop, daytona)
  -r, --resume [ID]        Resume threads (-r for the last one)
  -h, --help               Help
  --version                Version
```

---

## Usage examples

### Example 1: A simple conversation with Deepseek

```bash
$ ./venv/bin/deepagents-deepseek

Deep Agents v0.0.19
Type 'help' for commands

user > Write me a Python function that calculates Fibonacci
agent > # Fibonacci function
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

user > Make it faster using memoization
agent > # Optimized Fibonacci with memoization
def fibonacci(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci(n-1, cache) + fibonacci(n-2, cache)
    return cache[n]

user > quit
```

### Example 2: Using a specific agent

```bash
deepagents --agent my_coding_bot --model deepseek-coder
```

### Example 3: Resuming a previous session

```bash
# Resume the last conversation
deepagents -r

# Resume a specific conversation by ID
deepagents -r abc123def456
```

### Example 4: Auto-approve for automation

```bash
# The agent will run commands on its own without confirmation
deepagents --auto-approve --model deepseek-chat
```

### Example 5: Working with files

```bash
$ deepagents-deepseek

user > Create a Python script that lists all files in current directory
agent > [Creates and saves script]

user > Execute the script
agent > [Runs the script and shows output]

user > Save the output to results.txt
agent > [Saves to file]

user > Show me the results.txt
agent > [Displays file contents]
```

---

## Managing threads

```bash
# List all threads (sessions)
deepagents threads list

# Delete a specific thread
deepagents threads delete <THREAD_ID>

# Resume a thread
deepagents -r <THREAD_ID>
```

---

## Managing agents

```bash
# List all agents
deepagents list

# Reset an agent to the defaults
deepagents reset --agent mybot

# Create a copy of one agent from another
deepagents reset --agent newbot --target mybot
```

---

## Configuration and Security

### Where is the configuration stored?

```
~/.deepagents/
├── agents/                 # Custom agents definitions
├── AGENTS.md              # Agent metadata and configuration
├── .env                   # Environment variables (API keys)
└── sessions.db           # Session history database
```

### Environment variables for Deepseek

```bash
# Required
OPENAI_API_KEY=<your-deepseek-api-key>
OPENAI_BASE_URL=https://api.deepseek.com

# Optional
OPENAI_MODEL=deepseek-chat           # or deepseek-coder
DEEPAGENTS_LANGSMITH_PROJECT=myproject  # LangSmith tracing
```

### Security Best Practices

⚠️ **IMPORTANT:**
1. ✅ Never commit `.env` files containing API keys
2. ✅ Store keys in `~/.deepagents/.env` (locally)
3. ✅ Use `--auto-approve` only for trusted scripts
4. ✅ Regularly review API usage and billing
5. ✅ Rotate keys every 90 days or after a leak

---

## Deepseek models

### Available models

| Model | Purpose | Speed | Cost |
|--------|-----------|-------|------|
| `deepseek-chat` | General-purpose | Fast | Low |
| `deepseek-coder` | Code generation | Very Fast | Very Low |
| `deepseek-reasoner` | Complex reasoning | Medium | Medium |

### Using different models

```bash
# Deepseek Chat (default)
deepagents --model deepseek-chat

# Deepseek Coder (for code)
deepagents --model deepseek-coder

# Deepseek Reasoner (for complex tasks)
deepagents --model deepseek-reasoner
```

---

## Custom Agents (advanced)

### Create a custom agent with Deepseek

1. **Run deepagents-cli for the first time:**
   ```bash
   deepagents --agent mybot
   ```

2. **Edit the agent configuration:**
   ```bash
   ~/.deepagents/agents/mybot.md
   ```

3. **Example configuration:**
   ```markdown
   # My Custom Bot

   ## Purpose
   Help with Python development

   ## Capabilities
   - Write Python code
   - Debug scripts
   - Optimize performance

   ## Instructions
   Always use type hints.
   Prefer modern Python (3.10+).
   ```

4. **Run the custom agent:**
   ```bash
   deepagents --agent mybot --model deepseek-chat
   ```

---

## Troubleshooting

### Problem: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Make sure the .env file has the correct permissions
ls -la ~/.deepagents/.env

# Or set the variable explicitly
export OPENAI_API_KEY=sk-...
deepagents
```

### Problem: "Connection timeout to api.deepseek.com"

**Solution:**
1. Check the internet connection
2. Check that the key is active at https://platform.deepseek.com/
3. Make sure you have enough funds on the account

### Problem: "Model not found: deepseek-chat"

**Solution:**
```bash
# Use the wrapper script, which sets the URL explicitly
./venv/bin/deepagents-deepseek

# Or set the variables manually
export OPENAI_BASE_URL=https://api.deepseek.com
deepagents --model deepseek-chat
```

### Problem: "Command 'deepagents' not found"

**Solution:**
```bash
# Make sure the venv is activated
source ./venv/bin/activate

# Or use the full path
./venv/bin/deepagents
```

---

## Integration with a project

### Use in scripts

```python
# There are several ways to integrate deepagents

# Option 1: As a subprocess
import subprocess
result = subprocess.run([
    './venv/bin/deepagents-deepseek'
], input="Your prompt here")

# Option 2: Use LangChain directly (see the other examples in the project)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-..."
)
```

### Integration with LangChain

Deep Agents uses LangGraph and LangChain under the hood. If you want to use the same thing in your own scripts:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Create a model with Deepseek
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="<your-deepseek-api-key>"
)

# Use it as usual
response = llm.invoke([
    HumanMessage(content="What is 2 + 2?")
])
print(response.content)
```

---

## Performance Tips

1. **Use deepseek-coder for code** - it is specifically trained on code
2. **Enable --auto-approve for batch operations** - faster than confirming every time
3. **Use a sandbox for dangerous commands** - for example `--sandbox runloop`
4. **Manage the agent's memory** - the `remember` command helps improve the context
5. **Resume threads** - there is no need to start from scratch every time

---

## Additional resources

- 📚 LangChain Docs: https://docs.langchain.com/oss/python/deepagents/overview
- 🚀 Deepseek API: https://api-docs.deepseek.com/
- 📖 OpenAI API Format: https://platform.openai.com/docs/api-reference/chat/create

---

## Notes

- **Version:** deepagents-cli 0.0.19
- **Deepseek API:** OpenAI-compatible
- **Default Provider:** Deepseek Chat (deepseek-chat)
- **Setup Date:** 2026-02-08
- **Status:** Production Ready ✅
