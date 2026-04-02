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
pip install arpyc
python setup.py install
python setup.py develop
```

**IF YOU INSTALLED ANYTHING - UNINSTALL IT IMMEDIATELY:**
```bash
pip uninstall --break-system-packages -y deep-worker deep-worker-cli arpyc
rm -f ~/.local/bin/dwcli ~/.local/bin/deep-worker
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

### 3. ARPYC MUST ALWAYS BE FROM SOURCE

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

### 4. Why This Matters

Using installed packages instead of source code causes:
- RPyC proxy errors: `AttributeError: cannot access '__reversed__'`
- Version conflicts between installed arpyc and source arpyc
- Hours of debugging incompatibility issues
- Test failures that don't reflect real bugs

## Required Actions When Working on This Project

### Before Making Any Changes

1. **Check for .venv:**
   ```bash
   # Check this repository and any related project directories:
   find . -name ".venv" -type d
   ```

2. **If any .venv found - DELETE IMMEDIATELY:**
   ```bash
   rm -rf <path>/.venv
   ```

3. **Verify arpyc is from source:**
   ```bash
   python3 -c "import sys; import arpyc; print(arpyc.__file__)"
   # Should show: the in-repo arpyc source directory, e.g. .../arpyc/...
   # NOT: .venv/lib/python.../site-packages/arpyc/...
   ```

### When Adding arpyc Import

Always use this pattern with CRITICAL warning comment:

```python
# ============================================================================
# CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# DO NOT use 'pip install arpyc' or ANY virtual environment (.venv)!
# ALWAYS import arpyc from the in-repo arpyc source directory.
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
"This project runs globally. Verify arpyc is imported from the in-repo arpyc source"
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
3. **NEVER** suggest `pip install arpyc`
4. **ALWAYS** run everything from source code directly
5. **ALWAYS** import arpyc from the in-repo `arpyc` source
6. **ALWAYS** check for and delete `.venv` directories when encountered
7. **ALWAYS** add the CRITICAL warning comment block when importing arpyc
8. **IF YOU INSTALLED ANYTHING** - uninstall it immediately using the commands in section 1

Remember: The user spent hours debugging issues caused by `.venv` and installed packages. Everything MUST run from source code ONLY!
