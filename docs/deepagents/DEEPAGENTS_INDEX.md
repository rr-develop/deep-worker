# Deep Agents CLI - Documentation Index

## 🎯 Start here

This project is now fully configured to work with **Deep Agents CLI** with **Deepseek AI** support.

---

## 📚 Documentation

### 🚀 For a quick start
**File:** [`DEEPAGENTS_QUICKSTART.md`](./DEEPAGENTS_QUICKSTART.md)
- Basic commands
- Usage examples
- How to run it for the first time
- Session management
- Troubleshooting

→ **Read this first!**

### 🔧 For detailed configuration
**File:** [`DEEPAGENTS_SETUP.md`](./DEEPAGENTS_SETUP.md)
- Architecture and configuration
- Setup and installation
- Security & API Keys
- Advanced configuration
- Integration with LangChain

→ **Use it to understand the details**

### 📋 For reference
**File:** [`DEEPAGENTS_REFERENCE.md`](./DEEPAGENTS_REFERENCE.md)
- Short reference (cheat sheet)
- All commands in one place
- Quick examples
- Aliases and shortcuts

→ **Use it as a cheat sheet**

---

## ⚡ Quick start in 2 minutes

```bash
# 1. Activate the virtual environment
source ./venv/bin/activate

# 2. Run Deep Agents with Deepseek
deepagents-deepseek

# 3. Type your first request
user > Hello! Tell me about Deep Agents
agent > Deep Agents is...

# 4. Exit
user > quit
```

---

## 🏗️ Project structure

```
<project root>/
│
├── venv/                    # Python virtual environment
│   ├── bin/
│   │   ├── deepagents               # Main CLI binary
│   │   ├── deepagents-deepseek      # ✨ Deepseek wrapper script
│   │   └── python, pip, etc.
│   └── lib/python3.12/site-packages/
│       ├── deepagents/              # Core library
│       ├── deepagents_cli/          # CLI interface
│       └── langchain modules...
│
├── .env                              # ⚠️ API Keys (local, not in git)
│
├── setup_deepagents_deepseek.sh      # Setup automation script
│
├── DEEPAGENTS_INDEX.md               # ← This file (navigation)
├── DEEPAGENTS_QUICKSTART.md          # Quick start guide
├── DEEPAGENTS_SETUP.md               # Detailed setup & architecture
├── DEEPAGENTS_REFERENCE.md           # Reference card / cheat sheet
│
└── [Other project files...]

~/.deepagents/                         # User configuration directory (not in repo)
├── agents/                           # Custom agent definitions
├── AGENTS.md                         # Agent metadata
├── .env                              # API keys (local)
└── sessions.db                       # Session history
```

---

## 🔑 API Keys

### Current configuration

```
Project: <project root>/.env
DEEPSEEK_API_KEY=<your-deepseek-api-key>
```

### Where the key comes from

The key is taken from a shared internal configuration used by the LangChain projects in the ecosystem.

### ⚠️ Security

- ✅ `.env` files are protected by `.gitignore`
- ✅ Keys are stored locally and are not committed to git
- ✅ Access permissions verified (600)
- ✅ Used for LangChain and Deep Agents

---

## 🚀 Main commands

### Running

```bash
# Option 1: Wrapper script (recommended)
./venv/bin/deepagents-deepseek

# Option 2: After activating the venv
source ./venv/bin/activate
deepagents-deepseek

# Option 3: With an explicit model
deepagents --model deepseek-chat
```

### Session management

```bash
deepagents list              # List of agents
deepagents threads list      # List of sessions
deepagents -r               # Resume the last session
deepagents -r <ID>          # Resume a specific session
deepagents threads delete    # Delete a session
```

### Choosing a model

```bash
deepagents --model deepseek-chat       # General-purpose
deepagents --model deepseek-coder      # Code generation
deepagents --model deepseek-reasoner   # Complex reasoning
```

---

## 🎓 Usage examples

### Example 1: Coding
```bash
$ deepagents-deepseek
user > Write a Python function that reverses a string
agent > [Generates code]
user > Add error handling
agent > [Enhanced version]
```

### Example 2: Analysis
```bash
$ deepagents --model deepseek-reasoner
user > Explain how machine learning works
agent > [Detailed explanation]
```

### Example 3: Code Review
```bash
$ deepagents --model deepseek-coder
user > Review this code for bugs
user > [Paste code]
agent > [Analysis and suggestions]
```

---

## 📊 Models and Performance

| Model | Purpose | Speed | Cost |
|--------|-----------|----------|-----------|
| deepseek-chat | Everything | ⚡⚡⚡ | 💰 |
| deepseek-coder | Code | ⚡⚡⚡ | 💰 |
| deepseek-reasoner | Complex tasks | ⚡⚡ | 💰💰 |

**Deepseek pricing:**
- Input: $0.27 per 1M tokens
- Output: $1.10 per 1M tokens
- ~1000x cheaper than OpenAI GPT-4

---

## 🔗 Integration

### With the shared configuration

```bash
# The shared configuration is already set up for Deepseek
grep DEEPSEEK_API_KEY <shared config>/.env
# DEEPSEEK_API_KEY=sk-...

# This project uses the same key
grep DEEPSEEK_API_KEY ./.env
# DEEPSEEK_API_KEY=sk-...
```

### With the Shared AI Library

```python
# The shared AI library's LangChain worker supports Deepseek
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-..."
)
```

---

## ⚙️ Setup and Installation

### What is already installed

- ✅ Python 3.12+ virtual environment
- ✅ deepagents-cli 0.0.19
- ✅ deepagents 0.3.10
- ✅ langchain-core 1.2.9
- ✅ langchain-openai (latest)
- ✅ All dependencies

### What is already configured

- ✅ Deepseek API key in .env
- ✅ Wrapper script for automatic configuration
- ✅ User-level configuration in ~/.deepagents/
- ✅ Security: .env in .gitignore
- ✅ Ready for production ✓

### If you need to reinstall

```bash
# Run the setup script
bash setup_deepagents_deepseek.sh

# Or manually
mkdir -p ~/.deepagents
cp setup scripts to ~/.deepagents/bin/
source venv/bin/activate
pip install deepagents-cli
```

---

## 🆘 Troubleshooting

### "command not found: deepagents-deepseek"

```bash
# Solution: activate the venv or use the full path
source ./venv/bin/activate
deepagents-deepseek

# OR
./venv/bin/deepagents-deepseek
```

### "OPENAI_API_KEY not found"

```bash
# Solution: check .env and the environment variables
cat ./.env
echo $OPENAI_API_KEY
echo $DEEPSEEK_API_KEY

# Or set them manually
export DEEPSEEK_API_KEY=sk-...
export OPENAI_API_KEY=$DEEPSEEK_API_KEY
export OPENAI_BASE_URL=https://api.deepseek.com
```

### "Connection timeout"

```bash
# Solution: check the internet connection and API availability
ping api.deepseek.com
curl -I https://api.deepseek.com
```

More in [`DEEPAGENTS_QUICKSTART.md`](./DEEPAGENTS_QUICKSTART.md#troubleshooting)

---

## 📖 Provider documentation

- **Deepseek API:** https://api-docs.deepseek.com/
- **LangChain:** https://python.langchain.com/docs/
- **Deep Agents:** https://docs.langchain.com/oss/python/deepagents/overview
- **OpenAI API (Compatible):** https://platform.openai.com/docs/

---

## 🎯 Roadmap

- [x] Installation of deepagents-cli
- [x] Configuration of the Deepseek provider
- [x] Setup script
- [x] Wrapper for convenience
- [x] Documentation
- [ ] Custom agents (user-defined)
- [ ] Sub-agents for complex tasks
- [ ] CI/CD integration
- [ ] Learning from examples

---

## 📞 Contacts

| Question | Answer |
|--------|-------|
| How do I start? | Read [`DEEPAGENTS_QUICKSTART.md`](./DEEPAGENTS_QUICKSTART.md) |
| How does it work? | Read [`DEEPAGENTS_SETUP.md`](./DEEPAGENTS_SETUP.md) |
| Need a reference? | Use [`DEEPAGENTS_REFERENCE.md`](./DEEPAGENTS_REFERENCE.md) |
| Problems? | See Troubleshooting above |

---

## ✅ Checklist

Before using, make sure that you have:

- [ ] Read [DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md)
- [ ] Run `deepagents-deepseek` at least once
- [ ] Verified that the API key works
- [ ] Read the usage examples
- [ ] Verified that .env is not in git
- [ ] Set up aliases if needed (optional)

---

## 📊 Summary

| Component | Status | Version |
|-----------|--------|--------|
| deepagents-cli | ✅ Installed | 0.0.19 |
| deepagents | ✅ Installed | 0.3.10 |
| Deepseek API | ✅ Configured | OpenAI-compatible |
| Python venv | ✅ Ready | 3.12+ |
| Documentation | ✅ Complete | 3 guides |
| Security | ✅ Verified | .env protected |

**Status: Production Ready** ✅

---

## 🎉 Ready to use!

```bash
# Run Deep Agents with Deepseek
deepagents-deepseek

# Or use an alias on Linux/Mac
alias da='deepagents-deepseek'
da

# Or specify the model explicitly
deepagents --model deepseek-chat
```

If anything is unclear → read [DEEPAGENTS_QUICKSTART.md](./DEEPAGENTS_QUICKSTART.md)
