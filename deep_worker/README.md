# Deep Worker - AI Agent Framework

## ⚠️ CRITICAL: NO VIRTUAL ENVIRONMENTS ALLOWED ⚠️

**THIS PROJECT MUST RUN GLOBALLY WITHOUT ANY VIRTUAL ENVIRONMENT (.venv)**

### Why No Virtual Environments?

1. **RPyC from source**: This project uses a local RPyC source checkout for RPyC with AsyncioServer (single event loop, no threads)
2. **Version Conflicts**: Using `pip install rpyc` or installing into `.venv` causes incompatibility issues
3. **RPyC Proxy Errors**: Installing packages creates version mismatches leading to `AttributeError: cannot access '__reversed__'` and other proxy-related errors

### FORBIDDEN ACTIONS

```bash
# ❌ NEVER DO THIS:
python -m venv .venv
pip install rpyc
pip install -e .
```

### REQUIRED SETUP

```bash
# ✅ ALWAYS DO THIS:
# 1. Run directly on host without venv
# 2. Use system Python3
# 3. RPyC is imported from the local RPyC source checkout (AsyncioServer)
```

### Detecting Virtual Environment Violations

If you see paths like:
```
deep_worker/deep_worker_cli/.venv/lib/python3.14/site-packages/rpyc
```

**THIS IS WRONG!** Delete `.venv` immediately:
```bash
rm -rf deep_worker/deep_worker_cli/.venv
rm -rf deep_worker/deepagents/.venv
```

### How RPyC is Imported

The codebase imports RPyC from the local RPyC source checkout:

```python
# CORRECT: Import from the rpyc_async source directory
_rpyc_async_path = Path(RPYC_ASYNC_SOURCE_DIR)
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc
from rpyc.utils.async_server import AsyncioServer
```

### Related Repositories

Any companion repositories checked out alongside this one MUST also run without `.venv`.

## Project Structure

```
deep_worker/
├── deep_worker_cli/     # Main CLI application
├── deepagents/          # Agent framework
└── rpyc_async/          # RPyC with AsyncioServer (imported via sys.path)
```

## Installation

NO installation required. Run directly:

<!-- ============================================================================
PROTECTIVE COMMENT FOR FUTURE AGENTS:
DO NOT REPLACE deep-worker WITH ANYTHING ELSE (deepagents, python3 -m, etc.)!
deep-worker is the ONLY correct way to run this project.
The command is installed globally on the user's PATH.
============================================================================ -->

```bash
# The ONLY CORRECT way to run it (uses Deepseek by default):
deep-worker my-agent

# ⚠️ DO NOT USE:
# ⚠️   python3 -m deep_worker_cli.main       - old entry point, may use gemini
# ⚠️   python3 -m deep_worker_cli.cli.main   - the same thing
# ⚠️   deep_worker/deep_worker_cli/deep-worker - local script, use the global command
# ⚠️
# ⚠️ See: DEEPSEEK_PROVIDER.md for details
```

## Testing

```bash
cd deep_worker/deep_worker_cli
pytest tests/
```

---

**Remember: If you see `.venv` anywhere in this project, DELETE IT IMMEDIATELY!**
