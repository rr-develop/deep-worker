# Deep Agents CLI + Deepseek Integration

**Status:** ✅ Complete & Production Ready
**Date:** 2026-02-08
**Version:** deepagents-cli 0.0.19

## 🎯 What is this?

This project is now fully integrated with **Deep Agents CLI** — an interactive AI assistant with **Deepseek AI** support as the main LLM provider.

## 🚀 Quick start (30 seconds)

```bash
# Just run:
./venv/bin/deepagents-deepseek

# Type your question:
user > Write me a Python function that calculates factorial
agent > [Generates code]

# Exit:
user > quit
```

**Done!** Everything else is configured automatically.

---

## 📚 Documentation

### 🎓 For newcomers - READ IN THIS ORDER

1. **[DEEPAGENTS_INSTALLATION_SUMMARY.md](./DEEPAGENTS_INSTALLATION_SUMMARY.md)** ← START HERE
   - What was installed
   - How to run it
   - Usage examples

2. **[DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md)**
   - Full quick start guide
   - Usage examples
   - Session management
   - Troubleshooting

3. **[DEEPAGENTS_INDEX.md](./DEEPAGENTS_INDEX.md)**
   - Navigation across all the documentation
   - Project structure
   - All commands in one place

### 📖 For a detailed understanding

4. **[DEEPAGENTS_SETUP.md](./DEEPAGENTS_SETUP.md)**
   - Architecture and how it works
   - Detailed configuration
   - Security and API key management
   - Advanced features
   - Integration with LangChain

5. **[DEEPAGENTS_REFERENCE.md](./DEEPAGENTS_REFERENCE.md)**
   - Reference card (cheat sheet)
   - All commands in one place
   - Aliases and shortcuts

---

## ⚡ Main commands

### Running Deep Agents

```bash
# Main launch (recommended)
deepagents-deepseek

# Or with a different model
deepagents --model deepseek-coder      # For code
deepagents --model deepseek-reasoner   # For complex tasks

# Resume the previous conversation
deepagents -r

# Choose a different agent
deepagents --agent my-agent
```

### Session management

```bash
deepagents list              # List of agents
deepagents threads list      # List of sessions (conversations)
deepagents threads delete ID # Delete a session
```

---

## 🔑 API Keys & Security

### Current configuration

```bash
# Project .env file
<project root>/.env
├── DEEPSEEK_API_KEY=<your-deepseek-api-key>
├── GOOGLE_API_KEY=<your-google-api-key>
└── AI_PROVIDER=deepseek

# User-level config
~/.deepagents/.env
├── OPENAI_API_KEY=${DEEPSEEK_API_KEY}
├── OPENAI_BASE_URL=https://api.deepseek.com
└── OPENAI_MODEL=deepseek-chat
```

### ⚠️ Security

- ✅ .env files are protected by .gitignore
- ✅ API keys are not committed to git
- ✅ All keys are stored locally
- ✅ Access permissions verified (600)

---

## 📊 Models

### Available Deepseek models

| Model | Speed | Quality | Price | Usage |
|--------|----------|----------|------|---------------|
| deepseek-chat | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | General purpose (default) |
| deepseek-coder | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | Code generation |
| deepseek-reasoner | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰 | Complex reasoning |

Deepseek prices are ~1000 times cheaper than OpenAI GPT-4!

---

## 🏗️ Project structure

```
<project root>/
├── venv/                      # Python virtual environment
│   ├── bin/
│   │   ├── deepagents                 # Main CLI
│   │   ├── deepagents-deepseek        # ✨ Wrapper script
│   │   └── [other Python tools]
│   └── lib/python3.12/site-packages/
│       ├── deepagents/                # Core library
│       ├── deepagents_cli/            # CLI interface
│       └── [LangChain modules]
│
├── .env                                # API keys (not in git)
│
├── setup_deepagents_deepseek.sh       # Setup automation script
│
├── README_DEEPAGENTS.md               # ← This file
├── DEEPAGENTS_INSTALLATION_SUMMARY.md # What was installed
├── DEEPAGENTS_INDEX.md                # Navigation & overview
├── DEEPAGENTS_QUICKSTART.md           # Quick start guide
├── DEEPAGENTS_SETUP.md                # Detailed setup
├── DEEPAGENTS_REFERENCE.md            # Reference card
│
└── [Other project files...]

~/.deepagents/                          # User config (not in repo)
├── agents/                             # Custom agents
├── AGENTS.md                           # Agent definitions
├── .env                                # User environment
└── sessions.db                         # Session history
```

---

## 🎯 Usage examples

### Example 1: General-purpose questions

```bash
$ deepagents-deepseek
user > Explain quantum computing in simple terms
agent > Quantum computing uses quantum mechanics...

user > Give me 3 real-world examples
agent > 1. Drug discovery...
         2. Financial modeling...
         3. Cryptography...

user > quit
```

### Example 2: Code generation

```bash
$ deepagents --model deepseek-coder
user > Create a Python class for managing a to-do list
agent > [Generates complete class with methods]

user > Add persistence to database
agent > [Enhanced version with SQLite]

user > Add error handling
agent > [Final version with error handling]

user > quit
```

### Example 3: Complex reasoning

```bash
$ deepagents --model deepseek-reasoner
user > How to optimize this algorithm for O(n) time complexity?
user > [Paste algorithm]
agent > [Detailed analysis and optimization steps]

user > quit
```

---

## 🔗 Integration

### With other projects in the ecosystem

**The shared configuration** is already set up for Deepseek
```bash
grep DEEPSEEK_API_KEY <shared config>/.env
# DEEPSEEK_API_KEY=sk-...
```

**The shared AI library** — supports Deepseek through LangChain
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-..."
)
```

**This project** — is now fully integrated
```bash
deepagents-deepseek  # Works right away!
```

---

## ✨ Features

- 🤖 **Interactive AI Assistant** - Conversational interface
- 🛠️ **File Operations** - Create, read, edit, delete files
- 🚀 **Code Execution** - Run commands and scripts
- 🔄 **Session Management** - Save and resume conversations
- 🎯 **Custom Agents** - Create specialized agents
- 💰 **Cost-effective** - 1000x cheaper than GPT-4
- ⚡ **Fast** - deepseek-chat is very fast
- 🔒 **Secure** - Keys protected in .env files

---

## 🆘 Troubleshooting

### Problem: "Command not found: deepagents-deepseek"

**Solution:**
```bash
# Use full path
./venv/bin/deepagents-deepseek

# Or activate venv first
source ./venv/bin/activate
deepagents-deepseek
```

### Problem: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Check if key is in .env
grep DEEPSEEK ./.env

# Set manually if needed
export OPENAI_API_KEY=sk-...
deepagents-deepseek
```

### Problem: "Connection timeout"

**Solution:**
```bash
# Check internet connection
ping api.deepseek.com

# Verify API key is active
# Visit https://platform.deepseek.com/

# Check that account has sufficient balance
```

More solutions in [DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md#troubleshooting)

---

## 📞 Support

| Type of problem | Where to find help |
|-------------|-----------------|
| Quick questions | [DEEPAGENTS_REFERENCE.md](./DEEPAGENTS_REFERENCE.md) |
| Getting started | [DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md) |
| Technical details | [DEEPAGENTS_SETUP.md](./DEEPAGENTS_SETUP.md) |
| All information | [DEEPAGENTS_INDEX.md](./DEEPAGENTS_INDEX.md) |
| Installation info | [DEEPAGENTS_INSTALLATION_SUMMARY.md](./DEEPAGENTS_INSTALLATION_SUMMARY.md) |

---

## ✅ Installation Checklist

Before using, make sure that:

- [x] deepagents-cli is installed ✅
- [x] The Deepseek API key is loaded ✅
- [x] The wrapper script is created ✅
- [x] The configuration is initialized ✅
- [x] The documentation is ready ✅
- [ ] You have run deepagents-deepseek at least once (TODO)
- [ ] You have read DEEPAGENTS_QUICKSTART.md (TODO)

---

## 📊 Installed Components

| Component | Version | Status |
|-----------|---------|--------|
| deepagents-cli | 0.0.19 | ✅ |
| deepagents | 0.3.10 | ✅ |
| langchain-core | 1.2.9 | ✅ |
| langchain-openai | latest | ✅ |
| Python | 3.12+ | ✅ |
| Deepseek API | OpenAI-compatible | ✅ |

---

## 🚀 Next Steps

1. **First time?** → Read [DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md)
2. **Want details?** → Read [DEEPAGENTS_SETUP.md](./DEEPAGENTS_SETUP.md)
3. **Need reference?** → Check [DEEPAGENTS_REFERENCE.md](./DEEPAGENTS_REFERENCE.md)
4. **Start using:** `deepagents-deepseek`

---

## 🎉 Ready to Use!

```bash
# Everything is ready, just run:
deepagents-deepseek

# Or use a different model:
deepagents --model deepseek-coder

# Or resume the last conversation:
deepagents -r
```

---

## 📋 Summary

✅ **Installed:** deepagents-cli 0.0.19
✅ **Configured:** Deepseek as the main provider
✅ **Protected:** API keys in .env (not in git)
✅ **Documented:** 5 complete guides
✅ **Ready to use:** Production ready

**Status: 🎯 Ready to Go!**

---

## 📚 Related Guides

- 🚀 [Quick Start Guide](./DEEPAGENTS_QUICKSTART.md)
- 🔧 [Setup & Architecture](./DEEPAGENTS_SETUP.md)
- 📋 [Reference Card](./DEEPAGENTS_REFERENCE.md)
- 🗺️ [Navigation Index](./DEEPAGENTS_INDEX.md)
- 📦 [Installation Summary](./DEEPAGENTS_INSTALLATION_SUMMARY.md)

---

**Last Updated:** 2026-02-08
**Version:** deepagents-cli 0.0.19
**Status:** Production Ready ✅
