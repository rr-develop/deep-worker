# Instructions for Claude and AI Assistants

## 🚨 ABSOLUTE PROHIBITIONS 🚨

### 1. NEVER CREATE OR USE VIRTUAL ENVIRONMENTS

**FORBIDDEN COMMANDS:**
```bash
python -m venv .venv
python3 -m venv .venv
virtualenv .venv
pip install -e .
pip install arpyc
```

**IF YOU SEE `.venv` DIRECTORY - DELETE IT IMMEDIATELY!**

```bash
# Check for forbidden .venv directories:
find . -name ".venv" -type d

# Delete any found:
rm -rf <path>/.venv
```

### 2. ARPYC MUST ALWAYS BE FROM SOURCE

**CORRECT:**
```python
# Add arpyc from source to sys.path
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession
```

**WRONG:**
```bash
pip install arpyc  # ❌ NEVER DO THIS
```

### 3. Why This Matters

Using installed packages instead of source code causes:
- RPyC proxy errors: `AttributeError: cannot access '__reversed__'`
- Version conflicts between installed arpyc and source arpyc
- Hours of debugging incompatibility issues
- Test failures that don't reflect real bugs

## Required Actions When Working on This Project

### Before Making Any Changes

1. **Check for .venv:**
   ```bash
   # Run from the repository root (and any related checkouts you work with):
   find . -name ".venv" -type d
   ```

2. **If any .venv found - DELETE IMMEDIATELY:**
   ```bash
   rm -rf <path>/.venv
   ```

3. **Verify arpyc is from source:**
   ```bash
   python3 -c "import sys; import arpyc; print(arpyc.__file__)"
   # Should show the arpyc source directory in this checkout, e.g. .../arpyc/...
   # NOT: .venv/lib/python.../site-packages/arpyc/...
   ```

### When Adding arpyc Import

Always use this pattern with CRITICAL warning comment:

```python
# ============================================================================
# CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# DO NOT use 'pip install arpyc' or ANY virtual environment (.venv)!
# ALWAYS import arpyc from the arpyc source directory in this checkout.
# Using installed packages causes version conflicts and compatibility issues.
# This project MUST run globally without venv to ensure correct arpyc version.
# ============================================================================
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession
```

### Testing

```bash
# Run tests WITHOUT activating any venv:
cd deep_worker/deep_worker_cli
python3 -m pytest tests/
```

## Common Mistakes to Avoid

### ❌ WRONG: Creating setup.py with arpyc dependency
```python
# setup.py
install_requires=[
    'arpyc',  # ❌ NEVER LIST THIS AS DEPENDENCY
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
"This project runs globally. Verify arpyc is imported from the arpyc source directory"
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

1. **NEVER** suggest creating virtual environments
2. **NEVER** suggest `pip install arpyc`
3. **ALWAYS** import arpyc from the `arpyc` source directory in this checkout
4. **ALWAYS** check for and delete `.venv` directories when encountered
5. **ALWAYS** add the CRITICAL warning comment block when importing arpyc

Remember: The user spent hours debugging issues caused by `.venv`. Don't repeat this mistake!
