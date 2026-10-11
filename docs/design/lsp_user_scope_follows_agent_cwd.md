# User-scope LSP bridge follows the agent's own working directory

Status: accepted (owner decision, variant B, 2026-10-09)

## 1. The defect this fixes

`deep-worker lsp enable --scope user` wrote a bridge entry with the
workspace root **baked in** as a literal, taken from the cwd of whoever
ran the command:

```json
"serena": {"command": "deep-worker",
           "args": ["lsp", "bridge", "--project-root", "/path/to/project_a"]}
```

The user scope means "for every project", so that one literal became the
navigation target for every agent on the machine. Measured with the real
`load_mcp_config` over all 268 agents that have a `working_dir`: **241**
resolved `serena` to `/path/to/project_a` instead of their own directory;
only the 27 agents under `/path/to/project_b` were correct, and only
because that project has its own project-scope config that overrides the
user layer by name.

deep_worker has no built-in navigation default — `build_bridge_mcp_entry`
is called from exactly two places, both explicit write commands (`lsp
enable`, `lsp migrate`), and the agent-startup path contains no reference
to `deep_worker_cli.lsp` at all. So the user-scope entry did not override
an automatic per-project bridge; it replaced "no navigation" with "one
foreign project for everybody", which is what made it look like the
default had been lost.

## 2. Decision

Per-scope semantics become:

| Scope | Root | Meaning |
|---|---|---|
| **user** | the agent's own working directory, resolved at agent start | the default: navigate *my* project |
| **project** | explicit `--project-root` literal | this project, for anyone working in it |
| **agent** | explicit `--project-root` literal | this one agent |
| another name (e.g. `serena_repo_graph`) | explicit literal | deliberate access to a *different* project, **in addition to** own-project navigation |

Reaching a different project is therefore an additive, separately named
server — never a silent substitution of the agent's own navigation.

## 3. How the user-scope root is expressed: `${agent:working_dir}`

The user-scope entry is written as a placeholder instead of a literal:

```json
"serena": {"command": "deep-worker",
           "args": ["lsp", "bridge", "--project-root", "${agent:working_dir}"]}
```

resolved by the config loader at agent start to that agent's workspace
root.

### Why a placeholder, and why this one

Three options were considered.

1. **Omit `--project-root` entirely** and let the bridge default to "git
   root of cwd". Rejected: it only works if the bridge subprocess
   inherits the agent's cwd, which is not guaranteed for a stdio MCP
   child, and the failure is silent — the bridge would index whatever
   directory it happened to start in. `resolve_workspace_root`'s own
   docstring already insists the root be absolute for exactly this
   reason.
2. **A new environment variable** (e.g. `DEEP_WORKER_WORKING_DIR`) with
   the existing `${env:...}` syntax. Rejected: the task explicitly asks
   to avoid new magic env vars if avoidable, and it is avoidable. It
   would also be strictly worse — an env var is process-global ambient
   state, so a stale or inherited value silently yields the wrong root,
   which is the same class of bug being fixed.
3. **`${agent:working_dir}`** — chosen. It reuses the loader's existing
   placeholder-substitution stage (`_ENV_RE` already rewrites values in
   `command`/`args`/`env`/`cwd`), so no call-site plumbing changes
   shape. The value is passed **explicitly** into `load_mcp_config`,
   keeping that module's "no global state, no import side effects"
   contract intact. The namespace prefix `agent:` is distinct from
   `env:`, so the two cannot collide, and an unresolvable placeholder
   fails **loudly** rather than expanding to empty — matching the
   existing `UnresolvedEnvError` policy, under which starting a server
   with a blank value is considered worse than failing fast.

The root is resolved with `resolve_workspace_root(working_dir)` — the
same function `lsp enable` already uses — so a user-scope entry and a
project-scope entry written in the same directory denote the same root,
and git-worktree handling stays identical.

## 4. When the agent's directory is not a project

No silent substitution. `resolve_workspace_root` falls back to the
working directory itself when there is no `.git`, so:

* the placeholder always resolves to an absolute path — never empty,
  never the config file's directory, never `/`;
* the bridge **does** start, and navigation answers honestly from that
  directory. A non-git directory is still a perfectly indexable tree;
  Serena does not require git.
* if the shared service cannot serve that root, the existing contract
  applies unchanged: the bridge fails loudly and exits non-zero (policy
  §8, no stdio fallback), so the agent sees a failed server rather than
  a tool-less session that looks healthy.

The rejected alternative was "don't start the bridge unless the
directory is a git repo". It is worse: it makes navigation silently
absent for valid non-git trees, and absence is exactly the symptom this
whole investigation started from.

## 5. Scope of the change

* `external_mcp/config.py` — resolve `${agent:working_dir}` during
  layer parsing; take the value as an explicit argument.
* `process/agent_process.py` — pass the agent's working directory to
  `load_mcp_config` (it is already computed there for `project_path`).
* `lsp/enable.py` + `cli/lsp_command.py` — `--scope user` writes the
  placeholder form; `--scope project|agent` keeps the literal.
* `lsp/migrate.py` + `lsp/service/service_cli.py` — same rule for
  `lsp migrate`, see §5.1.
* `cli/mcp_command.py` — keep `mcp list` working, see §5.2.

### 5.1 `lsp migrate --scope user` (found in review)

`lsp migrate` accepts `--scope user` as well, and it rebuilt the entry
from the legacy argv. At user scope that reproduced the original defect
exactly: a legacy `--project /path/to/some_repo` became every agent's
navigation target again. The no-`--project` legacy form was worse — its
fallback root is the config file's own directory, i.e. the user config directory
for the user layer, which is the "config drifts into the install
directory" failure the `lsp enable` code already warns about.

Both reproduced on the pre-fix code through the old public surface:

```
legacy with --project  -> --project-root /path/to/project_a        (literal persists)
legacy without         -> --project-root <dir of mcp.json>        (config dir!)
```

`migrate` therefore takes the same `follow_agent_cwd` rule as `enable`:
user scope writes the placeholder, project/agent scope keeps recovering
the explicit legacy root.

What `migrate` deliberately does *not* touch: a user-scope entry that is
already a *bridge* entry but carries a literal root (`lsp bridge
--project-root /path/to/some_repo`). `is_legacy_entry` matches only argv
that execs the backend directly (`start-mcp-server`), so `migrate`
reports "nothing to migrate" for that shape and exits 0 — correct for its
own contract, which is the legacy form, not scope policy. Such entries
exist only because `lsp enable --scope user` wrote a literal before this
change; that source is now closed, so the population cannot grow. If one
is ever found, the remedy is not `migrate` but re-running enable:

```
deep-worker lsp enable --scope user --force     # no --project-root
```

`follow_agent_cwd` is `scope == "user" and not --project-root`, so this
rewrites the literal to `${agent:working_dir}`; `--force` is required
because enable leaves an existing entry alone without it.

### 5.2 `mcp list` must keep inspecting (found in review)

`deep-worker mcp list` calls the loader with no agent context, so a
stored placeholder made it raise — and because one bad entry aborts the
whole merge, the command printed nothing at all, hiding every unrelated
server. Failing loudly is right for an agent that is about to start a
server with a wrong root; it is wrong for an inspection command.

`mcp list --scope all` already documents itself as "the effective merge
for the current cwd", so it now passes that cwd as the working
directory: an agent-scoped placeholder is shown as it would resolve for
an agent working in this directory. The rejected alternative — printing
entries unresolved — would have shown a root that no agent ever sees.

Unchanged: layer precedence (user → project → agent → CLI, higher fully
replaces a same-named entry), the entry name `serena`, the bridge
protocol, and the navigation tools' schemas (still no per-call project
parameter — one server name is one project).

## 6. Migration

The user-level `mcp.json` is rewritten to the placeholder form by the
normal command (`lsp enable --scope user --force`), with the previous
file kept beside it as a dated backup. Verification is the same
measurement used in the diagnosis: run the real `load_mcp_config` for
every agent with a `working_dir` and assert each one's `serena` root
equals its own directory, with `/path/to/project_b` unchanged.

MCP config is read at agent start, so a running agent keeps its current
bridge until it is restarted.
