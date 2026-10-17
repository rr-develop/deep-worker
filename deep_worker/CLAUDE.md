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

**SCOPE OF THIS PROHIBITION (owner clarification 2026-10-13).** It governs
WORKING ON and DEVELOPING deep_worker *on this host*: here deep_worker must
keep running from source, exactly as it does now, and nothing about that may
change.

It does NOT forbid **testing other configurations** — a pip install, a
virtualenv, an isolated environment — when all of the following hold:

* everything lives in a **temporary directory** (e.g. under `/tmp`), including
  the venv, the build output and a throwaway `HOME`;
* nothing is installed into the host's system or user Python (no
  `--break-system-packages`, no `--user`, no install into the repo);
* the temporary directory is **removed afterwards**;
* the host's own from-source setup is left untouched.

This is what makes it possible to verify the shipped artifact the way a user
without sources would get it (build a wheel, install it in a throwaway venv,
run the console script). Testing the distribution is the point; adopting it as
this host's way of running deep_worker is still forbidden.

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

**SCOPE (owner clarification 2026-10-13).** Same boundary as §1: no virtualenv
for working on or developing deep_worker on this host, and never inside the
repository tree. A venv created in a **temporary directory** purely to test
another configuration (e.g. installing the built wheel the way an end user
would) is allowed, and must be deleted when the check is done. The `find .`
sweep above is the invariant that matters: the repo stays free of `.venv`.

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

### 5. CI test pipeline must stay green

The GitHub Actions `Unit tests` workflow (`.github/workflows/unittest.yml`) runs
`pytest tests/unit_tests` on Python 3.11 and 3.12 for every push to `master`/`main`.
A test that depends on a host-specific path is a red-pipeline bug the moment it
lands, because the CI runner uses `$GITHUB_WORKSPACE` on a fresh `ubuntu-latest`
image — NOT the developer's local directory layout. Required reading before touching
tests:

* [`docs/CI_CD_TESTS.md`](./docs/CI_CD_TESTS.md) — portability rules, the exact
  reproduction command, and the `gh` CLI recipes for pulling failure logs.
* [`docs/POLICIES.md`](./docs/POLICIES.md) §6 — the one-line policy statement.

If your test fails on CI but passes locally, the bug is in your test's
portability — never edit `.github/workflows/*.yml` and never add a skip to make
the pipeline green. Use the procedure in `CI_CD_TESTS.md` §5.

### 6. Why This Matters

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
