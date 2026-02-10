# Deep Agents CLI - Setup & Configuration Guide

## Architecture and configuration

### How does it work?

```
┌─────────────────────────────────────────────────────────────┐
│                    deepagents-cli                            │
│  (Interactive terminal UI with LLM provider support)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │      LangChain / LangGraph               │
        │  (Orchestration & Agent Management)    │
        └─────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │     LLM Provider Abstraction Layer      │
        │  (Supports OpenAI, Anthropic, Google)  │
        └─────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │   Deepseek API (OpenAI-compatible)      │
        │   https://api.deepseek.com              │
        └─────────────────────────────────────────┘
```

### Why Deepseek through the OpenAI API?

The Deepseek API is fully compatible with the OpenAI API:
- ✅ The same `langchain_openai` modules are used
- ✅ You only change `base_url` and `api_key`
- ✅ No special dependencies are needed
- ✅ Full compatibility with existing code

---

## Installation and Configuration

### Already installed in the project

```bash
# Check what is installed
./venv/bin/pip list | grep deepagents

# The output should be:
# deepagents                0.3.10
# deepagents-cli            0.0.19
```

### File structure

```
<project root>/
├── venv/                    # Python virtual environment
│   ├── bin/
│   │   ├── deepagents               # CLI binary
│   │   ├── deepagents-deepseek      # ✨ Custom wrapper for Deepseek
│   │   └── python, pip, etc.
│   └── lib/python3.12/site-packages/
│       ├── deepagents/              # Core agent library
│       ├── deepagents_cli/          # CLI interface
│       └── langchain*, langchain_openai, etc.
│
├── .env                              # API keys (local, not in git)
├── DEEPAGENTS_QUICKSTART.md          # ← This file
├── DEEPAGENTS_SETUP.md               # ← Detailed setup
└── setup_deepagents_deepseek.sh      # Setup automation script

~/.deepagents/                        # User config directory
├── agents/                           # Custom agent definitions
├── AGENTS.md                         # Agent metadata
├── .env                              # Environment variables
└── sessions.db                       # Session history
```

---

## Deepseek configuration

### Option 1: Environment Variables (recommended)

```bash
# Set the variables once
export OPENAI_API_KEY=<your-deepseek-api-key>
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_MODEL=deepseek-chat

# Then use it as usual
deepagents
```

### Option 2: .env file (~/.deepagents/.env)

```bash
# ~/.deepagents/.env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

On startup, deepagents-cli loads these variables automatically.

### Option 3: Project .env file

```bash
# <project root>/.env
DEEPSEEK_API_KEY=sk-...
AI_PROVIDER=deepseek
```

The `setup_deepagents_deepseek.sh` script uses this file for initialization.

### Option 4: Command line override

```bash
deepagents --model deepseek-chat
```

---

## API Key Security

### ⚠️ Critical security rules

**Never commit:**
```bash
# ❌ BAD - files containing keys
.env
.env.local
.env.production
secrets.json
credentials.json
```

**Always store locally:**
```bash
# ✅ GOOD
~/.deepagents/.env          # User-level config
<project root>/.env         # Project-level config (in .gitignore)
```

**Check the gitignore:**
```bash
# Make sure .gitignore contains:
.env
.env.*
!.env.example
*api_key*
```

### Security check

```bash
# Make sure the files are not readable by others
ls -la ~/.deepagents/.env
# The permissions should be: -rw------- or -rw-r-----

chmod 600 ~/.deepagents/.env  # If it needs fixing
```

---

## Using a project .env

### Structure of the project .env

```bash
# <project root>/.env

# =====================================
# API Keys
# =====================================

# Google Gemini (legacy, for compatibility)
GOOGLE_API_KEY=<your-google-api-key>

# Deepseek (RECOMMENDED)
DEEPSEEK_API_KEY=<your-deepseek-api-key>

# =====================================
# Provider Configuration
# =====================================

# For LangChain by default
AI_PROVIDER=deepseek

# For deepagents-cli explicitly
OPENAI_API_KEY=${DEEPSEEK_API_KEY}
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

---

## Integration between projects

### Shared configuration → this project

The **shared configuration** is already set up for Deepseek:

```bash
# <shared config>/.env
DEEPSEEK_API_KEY=sk-...
AI_PROVIDER=deepseek
```

**This project** now uses the same key:

```bash
# <project root>/.env
DEEPSEEK_API_KEY=sk-...  # Copied from the shared configuration
```

### How to synchronize the keys

```bash
# If the key was updated in the shared configuration, update it here too
source <shared config>/.env
export DEEPSEEK_API_KEY

# Update the project .env
sed -i "s/DEEPSEEK_API_KEY=.*/DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}/" \
    ./.env
```

---

## Wrapper scripts

### deepagents-deepseek

A special script that configures Deepseek automatically:

```bash
#!/bin/bash
# ~/.deepagents/bin/deepagents-deepseek

export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"

exec deepagents "$@"
```

**Usage:**
```bash
./venv/bin/deepagents-deepseek
# Or after activate:
source venv/bin/activate
deepagents-deepseek
```

---

## Models and Pricing

### Deepseek Models

```
deepseek-chat (Default)
├── Speed: ⚡⚡⚡ Fast
├── Quality: ⭐⭐⭐ Good (Like GPT-3.5)
├── Cost: 💰 Very Low (~$0.5 per 1M tokens)
└── Use for: General tasks, Q&A, coding help

deepseek-coder
├── Speed: ⚡⚡⚡ Very Fast
├── Quality: ⭐⭐⭐⭐ Excellent (Code-specific)
├── Cost: 💰 Very Low (~$0.3 per 1M tokens)
└── Use for: Code generation, debugging, optimization

deepseek-reasoner
├── Speed: ⚡⚡ Medium
├── Quality: ⭐⭐⭐⭐⭐ Excellent (Complex reasoning)
├── Cost: 💰💰 Low (~$2 per 1M tokens)
└── Use for: Complex problems, math, logic
```

### Comparison with other providers

| Provider | Model | Speed | Quality | Cost | Quota |
|----------|-------|-------|---------|------|-------|
| Deepseek | deepseek-chat | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | ∞ |
| OpenAI | gpt-4o | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | ∞ |
| Anthropic | claude-sonnet | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰 | ∞ |
| Google | gemini-pro | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Limited |

---

## Session management

### Saving conversations

```bash
# Every conversation is saved automatically
# Stored in ~/.deepagents/sessions.db

# List all sessions
deepagents threads list

# Resume the last one
deepagents -r

# Resume a specific one
deepagents -r <THREAD_ID>

# Delete a session
deepagents threads delete <THREAD_ID>
```

### Thread structure

```
Thread ID: abc123def456
├── Messages:
│   ├── User: "Write a function"
│   ├── Agent: [response]
│   ├── User: "Make it faster"
│   └── Agent: [improved version]
├── Tools Used:
│   ├── file_write
│   ├── execute_code
│   └── web_search
└── Metadata:
    ├── Created: 2026-02-08
    ├── Last Updated: 2026-02-08
    └── Model: deepseek-chat
```

---

## Tools Available

Deep Agents provides the tools:

### File Operations
```python
# Create a file
agent > Create a file test.py with: print("hello")

# Read a file
agent > Show me the contents of test.py

# Edit a file
agent > Add error handling to test.py

# Delete a file
agent > Delete test.py
```

### Execute Code
```python
# Run a command
agent > Execute: python test.py

# Run a script
agent > Create and run a script that lists files

# Shell commands
agent > Run: ls -la .
```

### Web Search
```python
# Search the internet
agent > Search for "deepseek api documentation"

# Fetch URL
agent > Get content from https://api.deepseek.com/docs
```

### Reasoning
```python
# SSub-agents for complex tasks
agent > Create a sub-agent to solve this problem

# Delegation
agent > Delegate this to another agent
```

---

## Advanced Configuration

### Custom Agents

```bash
# Create a custom agent
deepagents --agent my-researcher

# Edit the configuration
nano ~/.deepagents/agents/my-researcher.md

# File structure:
# ---
# name: my-researcher
# model: deepseek-chat
# ---
#
# You are a research assistant...
# Your job is to find and analyze information...
```

### AGENTS.md

```markdown
# My Custom Agents Configuration

## Agent: code-helper
- Model: deepseek-coder
- Tools: file_ops, execute_code
- Description: Helps with coding tasks

## Agent: researcher
- Model: deepseek-chat
- Tools: web_search, fetch_url
- Description: Research and analysis agent
```

### LangSmith Integration

```bash
# If you want to trace requests in LangSmith
export LANGSMITH_API_KEY=...
export DEEPAGENTS_LANGSMITH_PROJECT=my-project

deepagents
```

---

## Troubleshooting & FAQs

### Common problems

#### Q: How do I know whether the configuration works?

```bash
# Check that the venv is activated
which python
# It should be: <project root>/venv/bin/python

# Check that deepagents is installed
deepagents --version
# It should be: deepagents 0.0.19

# Check the API key
echo $OPENAI_API_KEY
# It should print: sk-... (if it is set)
```

#### Q: How do I choose different models for different tasks?

```bash
# For code use deepseek-coder
deepagents --model deepseek-coder

# For general tasks use deepseek-chat
deepagents --model deepseek-chat

# For complex tasks use deepseek-reasoner
deepagents --model deepseek-reasoner
```

#### Q: How do I integrate it with other tools?

```bash
# Use it as a subprocess in Python
import subprocess
result = subprocess.run([
    'deepagents-deepseek'
], input="Your prompt", capture_output=True, text=True)
print(result.stdout)

# Or use LangChain directly
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-..."
)
```

#### Q: Can several keys be used?

```bash
# Yes, for different service accounts
export DEEPSEEK_API_KEY_PROD=sk-prod...
export DEEPSEEK_API_KEY_DEV=sk-dev...

# When you need one of them, switch
export OPENAI_API_KEY=$DEEPSEEK_API_KEY_PROD
deepagents --model deepseek-chat
```

---

## Checklist

On first use, check that:

- [ ] The virtual environment is activated: `which python` shows the venv
- [ ] Deepagents is installed: `deepagents --version` prints the version
- [ ] The API key is loaded: `echo $OPENAI_API_KEY` is not empty
- [ ] The wrapper script exists: `ls -la ~/.deepagents/bin/deepagents-deepseek`
- [ ] The configuration files are in place: `ls -la ~/.deepagents/`
- [ ] .env is protected: `ls -la ~/.deepagents/.env | grep "600\|640"`
- [ ] The internet is reachable: `ping api.deepseek.com`
- [ ] The key is active: try `deepagents` and enter a simple question

---

## Summary

✅ **Installed:** deepagents-cli 0.0.19 with Deepseek support
✅ **Configured:** Wrapper script for automatic configuration
✅ **Security:** API keys are protected in .env files
✅ **Ready to Use:** Everything is ready to work with deep agents

Start with:
```bash
./venv/bin/deepagents-deepseek
```
