# Instructions for Claude and AI Assistants

## 🚨 ABSOLUTE PROHIBITIONS 🚨

### 1. NEVER INSTALL PACKAGES FROM THIS REPOSITORY

**🔥 CRITICAL: EVERYTHING MUST RUN FROM SOURCE CODE ONLY! 🔥**

**ABSOLUTELY FORBIDDEN COMMANDS:**
```bash
pip install -e .
pip install --break-system-packages -e .
pip install deep-worker
pip install deep-worker-cli
pip install deepagents
python setup.py install
python setup.py develop
```

**IF YOU INSTALLED ANYTHING - UNINSTALL IT IMMEDIATELY:**
```bash
pip uninstall --break-system-packages -y deep-worker deep-worker-cli deepagents
rm -f "$(command -v dwcli)" "$(command -v deep-worker)"
```

### 2. NEVER CREATE OR USE VIRTUAL ENVIRONMENTS

**FORBIDDEN COMMANDS:**
```bash
python -m venv .venv
python3 -m venv .venv
virtualenv .venv
```

**IF YOU SEE `.venv` DIRECTORY - DELETE IT IMMEDIATELY!**

```bash
# Check for forbidden .venv directories:
find . -name ".venv" -type d

# Delete any found:
rm -rf <path>/.venv
```

### 3. RPyC: USE THE LOCAL rpyc_async SOURCE CHECKOUT ONLY

The project uses a local `rpyc_async` source checkout for all RPyC communication.
This provides `AsyncioServer` — single event loop, no threads.

**CORRECT (production code already does this):**
```python
# In agent_process.py:
_rpyc_async_path = Path(RPYC_ASYNC_SOURCE_DIR)
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc
from rpyc.utils.async_server import AsyncioServer
```

**FORBIDDEN:**
- `pip install rpyc` — NEVER install rpyc from pip
- `import arpyc` — the arpyc library has been REMOVED from the repository
- `ThreadedServer` — NEVER use ThreadedServer, only AsyncioServer
- `BgServingThread` — NEVER use background serving threads

### 4. NEVER USE --no-verify ON GIT COMMITS

```bash
git commit --no-verify  # ❌ ABSOLUTELY FORBIDDEN
```

Pre-commit hooks enforce type checking, .venv detection, and pip installation
guards. If a commit is blocked, FIX THE ISSUE — do not bypass the check.

### 5. Why This Matters

Using installed packages instead of source code causes:
- RPyC proxy errors: `AttributeError: cannot access '__reversed__'`
- Version conflicts between installed and source packages
- Hours of debugging incompatibility issues
- Test failures that don't reflect real bugs

## Required Actions When Working on This Project

### Before Making Any Changes

1. **Check for .venv:**
   ```bash
   find . -name ".venv" -type d
   ```

2. **If any .venv found - DELETE IMMEDIATELY:**
   ```bash
   rm -rf <path>/.venv
   ```

### Testing

```bash
# Run tests WITHOUT activating any venv:
cd deep_worker/deep_worker_cli
python3 -m pytest tests/
```

## Common Mistakes to Avoid

### ❌ WRONG: Creating setup.py with package dependencies
```python
# setup.py
install_requires=[
    'rpyc',  # ❌ NEVER LIST THIS AS DEPENDENCY
]
```

### ✅ CORRECT: No setup.py, run directly from source
```bash
python3 -m deep_worker_cli.cli.main
```

### ❌ WRONG: Suggesting virtual environment for isolation
```
"Let's create a virtual environment for better dependency management..."
```

### ✅ CORRECT: Running globally with source imports
```
"This project runs globally with source imports from the local rpyc_async checkout"
```

## When User Reports RPyC Errors

If you see errors like:
- `AttributeError: cannot access '__reversed__'`
- `AttributeError: cannot access '__iter__'`
- RPyC proxy-related AttributeErrors

**FIRST ACTION: Check for .venv**

```bash
find . -name ".venv" -type d
# If found -> rm -rf immediately
```

This is likely the root cause!

## Summary for Claude

1. **NEVER** install packages from this repository (`pip install -e .`, `pip install deep-worker`, etc.)
2. **NEVER** suggest creating virtual environments
3. **NEVER** use `--no-verify` on git commits
4. **ALWAYS** run everything from source code directly
5. **ALWAYS** use the local `rpyc_async` source checkout for RPyC (AsyncioServer, single event loop)
6. **ALWAYS** check for and delete `.venv` directories when encountered
7. **IF YOU INSTALLED ANYTHING** - uninstall it immediately using the commands in section 1

Remember: The user spent hours debugging issues caused by `.venv` and installed packages. Everything MUST run from source code ONLY!
