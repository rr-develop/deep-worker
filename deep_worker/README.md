# Deep Worker — AI Agent Framework

`deep_worker` is the core agent runtime of this repository. It runs agents as
background daemon processes and exposes them to the interactive `dwcli` client.

## Runtime model

- Agents run as daemon processes on the local machine.
- All network services bind to **127.0.0.1 only** — this is a local, trusted
  environment tool (see `deep_worker_cli/SECURITY.md`).
- The daemon and client communicate over RPyC (AsyncioServer, single event
  loop, no threads).

## ⚠️ No virtual environments

This project must run with the **system Python** — **no `.venv`, no
`pip install rpyc`**.

The reason is that RPyC is imported from a local source checkout
(`rpyc-async`, an AsyncioServer-based RPyC fork). Installing RPyC from PyPI or
creating a virtual environment causes version mismatches that break the proxy
layer (e.g. `AttributeError: cannot access '__reversed__'`).

```bash
# ✅ Correct — run via the bash launchers (no venv)
# Make the CLI globally available by symlinking the launchers into ~/.local/bin:
ln -sf "$PWD/deep_worker_cli/deep-worker" ~/.local/bin/deep-worker
ln -sf "$PWD/deep_worker_cli/dwcli" ~/.local/bin/dwcli
deep-worker my_agent          # start an agent daemon
dwcli my_agent                # connect to it interactively

# ❌ Never do this
python -m venv .venv
pip install rpyc
pip install -e .
```

If you see a `.venv` path under this tree, delete it and re-run with the
system interpreter.

## Layout

| Path | Purpose |
|---|---|
| `deep_worker_cli/` | `deep-worker` daemon and `dwcli` client (the CLI package) |
| `deepagents/` | vendored copy of the upstream `deepagents-cli` source |
