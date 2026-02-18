# Deep Worker - AI Agent Framework

## ⚠️ CRITICAL: NO VIRTUAL ENVIRONMENTS ALLOWED ⚠️

**THIS PROJECT MUST RUN GLOBALLY WITHOUT ANY VIRTUAL ENVIRONMENT (.venv)**

### Why No Virtual Environments?

1. **arpyc Dependency**: This project depends on `arpyc` from source code in the `arpyc` directory of this checkout
2. **Version Conflicts**: Using `pip install arpyc` or installing into `.venv` causes incompatibility issues
3. **RPyC Proxy Errors**: Installing packages creates version mismatches leading to `AttributeError: cannot access '__reversed__'` and other proxy-related errors

### FORBIDDEN ACTIONS

```bash
# ❌ NEVER DO THIS:
python -m venv .venv
pip install arpyc
pip install -e .
```

### REQUIRED SETUP

```bash
# ✅ ALWAYS DO THIS:
# 1. Run directly on host without venv
# 2. Use system Python3
# 3. Import arpyc from source (the arpyc directory in this checkout)
```

### Detecting Virtual Environment Violations

If you see paths like:
```
deep_worker/deep_worker_cli/.venv/lib/python3.14/site-packages/arpyc
```

**THIS IS WRONG!** Delete `.venv` immediately:
```bash
rm -rf deep_worker/deep_worker_cli/.venv
rm -rf deep_worker/deepagents/.venv
```

### How arpyc is Imported

The codebase correctly imports arpyc from source:

```python
# CORRECT: Import from source directory
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession
```

### Related Repositories

This repository and the other related checkouts it is used alongside MUST also run without `.venv`.

## Project Structure

```
deep_worker/
├── deep_worker_cli/     # Main CLI application
├── deepagents/          # Agent framework
└── ../arpyc/            # arpyc source (MUST use this!)
```

## Installation

NO installation required. Run directly:

```bash
cd deep_worker/deep_worker_cli
python3 -m deep_worker_cli.cli.main
```

## Testing

```bash
cd deep_worker/deep_worker_cli
pytest tests/
```

---

**Remember: If you see `.venv` anywhere in this project, DELETE IT IMMEDIATELY!**
