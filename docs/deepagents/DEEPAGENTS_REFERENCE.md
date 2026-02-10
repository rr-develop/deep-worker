# Deep Agents CLI - Reference Card

## 📋 Quick reference

### Main commands

```bash
# RUNNING
deepagents-deepseek                    # Main launch with Deepseek
deepagents --model deepseek-chat       # With an explicit model
deepagents --model deepseek-coder      # Model for code
source activate && deepagents-deepseek # After activating the venv

# MANAGEMENT
deepagents --agent mybot               # Use a specific agent
deepagents -r                          # Resume the last conversation
deepagents -r <ID>                     # Resume a specific conversation
deepagents --auto-approve              # Auto-approve tools without confirmation
deepagents --version                   # Show the version
deepagents --help                      # Help

# INFORMATION
deepagents list                        # List of agents
deepagents threads list                # List of threads (sessions)
deepagents threads delete <ID>         # Delete a thread
deepagents reset --agent mybot         # Reset an agent to the defaults
```

### Interactive commands (inside deepagents)

```
help      → Help
clear     → Clear the screen
quit/exit → Exit
remember  → Re-read and update the memory
tokens    → Show the tokens used
```

---

## 🔧 Configuration

### Environment variables

```bash
# REQUIRED for Deepseek
DEEPSEEK_API_KEY=<your-deepseek-api-key>

# OPTIONAL (if the wrapper script is used)
OPENAI_API_KEY=${DEEPSEEK_API_KEY}
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

### Configuration files

```
<project root>/.env                  # Project-level (not committed)
~/.deepagents/.env                   # User-level (local)
~/.deepagents/AGENTS.md              # Agent definitions
~/.deepagents/agents/                # Custom agents directory
```

---

## 🚀 Quick examples

### Example 1: A simple conversation

```bash
$ deepagents-deepseek
user > How do I sort a list in Python?
agent > You can use the sorted() function...
user > quit
```

### Example 2: Coding

```bash
$ deepagents --model deepseek-coder
user > Create a decorator that logs function calls
agent > [Generates code]
user > Add error handling
agent > [Enhanced version]
user > quit
```

### Example 3: Resuming a session

```bash
$ deepagents -r
# Continuing the last conversation...
```

---

## 📍 Paths and locations

```
Setup script:
  <project root>/setup_deepagents_deepseek.sh

Wrapper script:
  <project root>/venv/bin/deepagents-deepseek

CLI binary:
  <project root>/venv/bin/deepagents

Config directory:
  ~/.deepagents/

Project .env:
  <project root>/.env

Main docs:
  DEEPAGENTS_QUICKSTART.md  (Quick start)
  DEEPAGENTS_SETUP.md       (Detailed setup)
  DEEPAGENTS_REFERENCE.md   (This file)
```

---

## 🔑 API Key Management

### Storing keys

| Location | Usage | Security |
|-------|---------------|-------------|
| `~/.deepagents/.env` | Local user | ✅ Not in git |
| `<project root>/.env` | Project-specific | ✅ In .gitignore |
| Environment variables | Temporary | ✅ Not in the code |
| Files in .git | ❌ NEVER | ❌ Dangerous |

### Security check

```bash
# Make sure .env is not in git
git status | grep ".env"  # Should be empty

# Check the file permissions
ls -la ~/.deepagents/.env  # Should be 600 or 640

# Make sure the key is not in the history
grep -r "sk-" ~/.bash_history  # Should be empty (if possible)
```

---

## 🎯 Models

### Choosing a model

```bash
# General-purpose (Default)
deepagents --model deepseek-chat

# Code generation (fast)
deepagents --model deepseek-coder

# Complex reasoning (slow, but accurate)
deepagents --model deepseek-reasoner
```

### Model comparison

```
deepseek-chat
├── Speed: ⚡⚡⚡ Fast
├── Cost: 💰 Very cheap
└── Best for: Q&A, text, coding

deepseek-coder
├── Speed: ⚡⚡⚡ Very fast
├── Cost: 💰 Cheapest of all
└── Best for: Code generation

deepseek-reasoner
├── Speed: ⚡⚡ Slow
├── Cost: 💰💰 Cheap
└── Best for: Complex tasks, mathematics
```

---

## 🛠️ Integration with LangChain

### Use in Python code

```python
from langchain_openai import ChatOpenAI

# Create a model with Deepseek
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-..."
)

# Use it as usual
response = llm.invoke("Hello!")
print(response.content)
```

### Use with Chains

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

template = "You are a helpful assistant. User: {input}"
prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(input="What is Python?")
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|----------|---------|
| `command not found: deepagents` | `source venv/bin/activate` |
| `OPENAI_API_KEY not found` | `export OPENAI_API_KEY=sk-...` |
| `Connection timeout` | Check the internet connection and that the key is active |
| `Model not found` | Use the `deepagents-deepseek` wrapper |
| `Permission denied` | `chmod +x ~/.deepagents/bin/deepagents-deepseek` |

---

## 📊 Sessions and History

```bash
# List all sessions
deepagents threads list

# Information about a specific session
deepagents threads info <ID>

# Delete a session
deepagents threads delete <ID>

# Clear all sessions
rm ~/.deepagents/sessions.db  # ⚠️ Careless!
```

---

## 🎓 Usage examples

### Example: Code Review

```bash
$ deepagents --model deepseek-coder
user > Review this function for bugs
user > [Paste code]
agent > Found 3 issues...
user > Fix them
agent > [Fixed version]
```

### Example: Problem Solving

```bash
$ deepagents --model deepseek-reasoner
user > How to optimize database queries?
agent > Consider these strategies...
user > Which is best for my case?
agent > Depends on your data structure...
```

### Example: Documentation

```bash
$ deepagents
user > Create API documentation for my code
user > [Paste code]
agent > [Generates documentation]
user > Make it more concise
agent > [Updated version]
```

---

## 🔗 Links

- **Deepseek API:** https://api-docs.deepseek.com/
- **LangChain Docs:** https://python.langchain.com/docs/
- **Deep Agents Guide:** https://docs.langchain.com/oss/python/deepagents/overview
- **OpenAI API (Compatible):** https://platform.openai.com/docs/api-reference

---

## 📊 Version information

| Component | Version | Status |
|-----------|--------|--------|
| deepagents-cli | 0.0.19 | ✅ Installed |
| deepagents | 0.3.10 | ✅ Installed |
| langchain-core | 1.2.9 | ✅ Installed |
| langchain-openai | Latest | ✅ Installed |
| Python | 3.12+ | ✅ Supported |

---

## ⚙️ Aliases (optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Quick launch
alias da='deepagents-deepseek'
alias da-coder='deepagents --model deepseek-coder'
alias da-resume='deepagents -r'

# Navigation
alias cdla='cd <project root>'
alias activate-la='source <project root>/venv/bin/activate'

# Management
alias da-list='deepagents threads list'
alias da-help='deepagents --help'
```

Then you can use:
```bash
source ~/.bashrc  # or ~/.zshrc
da               # instead of deepagents-deepseek
da-coder         # instead of deepagents --model deepseek-coder
da-resume        # instead of deepagents -r
```

---

## 🔐 Security Checklist

- [ ] `.env` files are not committed to git
- [ ] `~/.deepagents/.env` has 600 permissions
- [ ] API keys are not logged
- [ ] Keys are rotated regularly (every 90 days)
- [ ] API usage is monitored
- [ ] Nobody has been given access to the .env files

---

## 📝 Notes

- **Setup Date:** 2026-02-08
- **Deepseek API:** OpenAI-compatible
- **Default Provider:** deepseek-chat
- **Production Status:** ✅ Ready

---

## 🎯 Next steps

1. Run `deepagents-deepseek` for the first time
2. Read `DEEPAGENTS_QUICKSTART.md`
3. Study the examples above
4. Create a custom agent if needed
5. Integrate it with your projects
