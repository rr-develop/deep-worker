# Development Requirements - Deep Worker

## 🚫 NO PIP INSTALL Policy

**CRITICAL REQUIREMENT**: Deep-worker must work **WITHOUT INSTALLATION** via `pip install`.

### Why this matters

This is an actively developed project on a development host. Any code change must take effect immediately on the next command run, without intermediate installation steps.

### Launch architecture

Deep-worker uses **the same approach as some comparable CLI tools** (e.g. a `cco`-style launcher):

1. **Bash wrappers** in the project root:
   - `deep-worker` — starts the agent process
   - `dwcli` — short alias for the client connector
   - `deep-worker-cli` — full name for the client connector

2. **Symbolic links** in `~/.local/bin/`:
   ```bash
   ~/.local/bin/deep-worker -> deep_worker/deep_worker_cli/deep-worker
   ~/.local/bin/dwcli -> deep_worker/deep_worker_cli/dwcli
   ~/.local/bin/deep-worker-cli -> deep_worker/deep_worker_cli/deep-worker-cli
   ```

3. **Direct execution of Python modules**:
   ```bash
   python3 -m deep_worker_cli.cli.agent_command    # for deep-worker
   python3 -m deep_worker_cli.cli.client_command   # for dwcli
   ```

4. **PYTHONPATH is configured automatically** in the bash wrappers:
   - The path to `deep_worker_cli` is added
   - The path to `deepagents` (dependency) is added

### How this works in practice

```bash
# 1. Edit the Python code
vim deep_worker_cli/agent.py

# 2. Run it right away — the changes take effect
dwcli --list

# There is NO intermediate "pip install -e ." step!
```

### Comparison with a similar CLI tool

| Aspect | Similar CLI tool (cco) | Deep Worker |
|--------|---------------------|-------------|
| Bash wrapper | `<tool>/cco` | `deep_worker/deep_worker_cli/deep-worker` |
| Symbolic link | `~/.local/bin/cco` | `~/.local/bin/dwcli` |
| Python entry point | `python3 -m src.<tool>.main` | `python3 -m deep_worker_cli.cli.*` |
| Installation | ❌ Not required | ❌ Not required |

### For agents and developers

**NEVER** suggest to the user:
- `pip install -e .`
- `pip install --force-reinstall`
- Any other pip-based installation options

**ALWAYS** use direct execution through the bash wrappers:
- `deep-worker my-agent`
- `dwcli --list`
- `dwcli --status my-agent`

### Verifying a correct setup

```bash
# All of these commands must work right away:
which dwcli
# Output: /home/<user>/.local/bin/dwcli

dwcli --help
# Output: usage: deep-worker-cli [-h] ...

ls -la ~/.local/bin/dwcli
# Output: lrwxrwxrwx ... dwcli -> deep_worker/deep_worker_cli/dwcli
```

### Technical details

The bash wrappers:
1. Determine `SCRIPT_DIR` (the script's directory)
2. Compute `PARENT_DIR` to access `deepagents`
3. Set `PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR/deepagents:..."`
4. Run `python3 -m deep_worker_cli.cli.*`

This guarantees:
- ✅ Modules are found without installation
- ✅ Code changes are visible immediately
- ✅ No need for virtualenv/pip install
- ✅ Works like comparable no-install CLI tools

---

**Updated**: February 2026
**Status**: ✅ Mandatory requirement
