# dw-hist

Host-level CLI for inspecting deep-worker agent conversation history. Standalone
from `deep-worker` — see [`docs/DESIGN_DW_HIST_INITIAL.md`](docs/DESIGN_DW_HIST_INITIAL.md)
for the design.

## Install

`dw-hist` ships as a source tree + bash launcher (same convention as
`deep-worker` / `dwcli`). To make it globally available, symlink the launcher
into `~/.local/bin/`:

```bash
# From this repository's checkout (replace <repo> with your checkout path):
ln -sf <repo>/dw_hist/dw-hist ~/.local/bin/dw-hist
```

Ensure `~/.local/bin` is on your `$PATH`. The launcher sets `PYTHONPATH`
internally — no `pip install` is needed (project policy).

## Commands (v1)

### List agents that have a built history index

```bash
dw-hist --list
dw-hist --list --format json
```

Exits `0` even when no agents have an index (prints nothing on stdout), so
shell loops are safe:

```bash
for a in $(dw-hist --list); do
    dw-hist "$a" -q "deadlock"
done
```

### Search one agent's history semantically

```bash
dw-hist <agent_name> -q "<query text>"
dw-hist <agent_name> -q "<query>" -k 10
dw-hist <agent_name> -q "<query>" --format json
```

Output mirrors `history_memory`'s tool output and includes the explicit
similarity `score` for each result. `--format json` emits a flat record list
for scripting.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success / `--list` with zero agents |
| 2 | usage error (conflicting flags, invalid `--top-k`, unknown agent) |
| 3 | agent exists but has no history index |
| 4 | search runtime error |

## What v1 deliberately does NOT do

- No reindex / clear / info commands yet (placeholders in the design).
- No agent-runtime integration — this is a read-only host operator's tool.
- No `pip install`, no `pyproject.toml` — source-imports only.
