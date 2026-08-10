# LSP (Language Server Protocol) for deep-worker agents

Precise code navigation for agents: go-to-definition, show-usages,
find-symbol, outline, symbol refactoring — **in a single call**, without
`grep → read_file → grep` chains.

Optional capability: acts as a semantic index, enabled per project. If not
enabled, agent behavior is unchanged.

## Contents

| Path | Purpose |
|------|---------|
| [`docs/`](docs/) | Integration design |
| [`research/`](research/) | OSS research and findings |

## How it works

The backend is [Serena](https://github.com/oraios/serena) (MIT) — an
MCP server on top of language servers (40+ languages, Python and Kotlin
included). It connects through the **existing** deep-worker External MCP:
Serena lives as a separate stdio process, and its tools arrive to the
agent as regular `BaseTool`s. We don't write our own LSP client.

```
Agent → serena__find_symbol → Serena (MCP) → pyright / kotlin-lsp → project
```

## Installation

```bash
uv tool install -p 3.13 serena-agent
```

After that, `serena`, `serena-agent`, `serena-hooks` appear in PATH.

## Enabling

```bash
deep-worker lsp enable     # for the current project (default)
deep-worker lsp status     # status + binary presence
deep-worker lsp disable    # disable
```

The scope can be set explicitly:

```bash
deep-worker lsp enable --scope user            # for all projects
deep-worker lsp enable --scope agent:my_agent  # for a single agent
deep-worker lsp enable --config /path/mcp.json --project-root /path/repo
```

The command appends a `serena` entry to the corresponding `mcp.json`
(file `0600`, directory `0700`), leaving other MCP servers untouched.
Repeated calls are idempotent; to overwrite the entry use `--force`.

**Restart the agent** for the change to take effect.

## What the agent gets

| Tool | IDE equivalent |
|------|----------------|
| `serena__find_symbol` | find symbol by name |
| `serena__find_referencing_symbols` | Show usages / Find references |
| `serena__find_declaration` | Go to definition |
| `serena__find_implementations` | Go to implementations |
| `serena__get_symbols_overview` | file outline |
| `serena__get_diagnostics_for_file` | Problems |
| `serena__rename_symbol` | Rename refactoring |
| `serena__replace_symbol_body` | replace symbol body |

Names are mangled (`.` → `__`) — that's how the agent sees them. When LSP
is enabled, a short snippet is added to the system prompt explaining when
to prefer a symbol tool over `grep`.

Serena tools are marked `auto_approve` — no HITL is requested for them
(including writing ones). Disable with:
`deep-worker lsp enable --no-auto-approve`.

## Languages

- **Python** — pyright, works out of the box.
- **Kotlin** — official `Kotlin/kotlin-lsp`, Serena downloads it itself;
  first start is noticeably slower (JVM). On Android projects (AGP) a
  Gradle init script is required — installed automatically, see below.
- The other 40+ languages — as needed, through the same unified interface.

### Kotlin + Android: Gradle init script

Without it, cross-file navigation in an AGP project is **silently empty**:
the server imports the module without a single source root, and
`find_referencing_symbols`, `find_declaration`, `find_implementations`
return `[]` for any cross-file request. Requests within a single file still
work, which makes the LSP look healthy. The server log shows `has an empty
set of source sets` and `Failed to call 'onVariants' in 'androidComponents'
extension`.

The cause is two Kotlin LSP defects: Android variants are assembled in
`beforeProject`, when the `androidComponents` extension does not exist yet;
and source dirs are read in the configuration phase, which Gradle forbids
(`Querying the mapped value of ... before task ':app:generateDebugBuildConfig'
has completed`). Full analysis — in
[DESIGN_KOTLIN_LSP_ANDROID_SOURCE_SETS.md](docs/DESIGN_KOTLIN_LSP_ANDROID_SOURCE_SETS.md).

`deep-worker lsp enable` installs the workaround:

```
$GRADLE_USER_HOME/init.d/deep-worker-kotlin-android-source-sets.gradle
```

by default — `~/.gradle/init.d/…`, permissions `0644` in a `0755` directory.

**The file is global.** Gradle reads init scripts only from
`$GRADLE_USER_HOME/init.d` (a script placed in the project directory is
**not** applied), so it ends up in all of the user's builds. Outside the
Kotlin LSP import the script is a no-op: it exits immediately if the build
has no `IdeaGradleLspPlugin` plugin. All reflection is wrapped in
`try/catch (Throwable)`: if the Kotlin LSP versions diverge, navigation
degrades to the current (broken) state, but the import does not fail.

Opt out when enabling:

```bash
deep-worker lsp enable --no-gradle-init
```

Remove:

```bash
deep-worker lsp disable          # removes only the file with our version marker
rm ~/.gradle/init.d/deep-worker-kotlin-android-source-sets.gradle   # manually
```

A manually edited file is not overwritten by `enable` without `--force`, and
`disable` does not remove it (the version marker in the first line no longer
matches). The current state is shown by `deep-worker lsp status`.

#### Kotlin LSP version

Verified on `kotlin_server-262.9593.0` (bundled JBR 25). Serena downloads
an older build, and `kotlin_lsp_version` in `.serena/project.yml` does not
work for new versions: SolidLSP only supports `.zip`, whereas linux-x64
starting with `v262.4739.0` is published as `.tar.gz`. Therefore the new
server is specified as an unpacked directory in the target project's
`.serena/project.yml`:

```yaml
ls_specific_settings:
  kotlin:
    ls_path: /path/to/kotlin_server-262.9593.0/bin/intellij-server
```

`.serena/project.yml` belongs to the target project; `deep-worker` does not
manage it — it only writes `mcp.json`.

## Highlights

- **The cost is paid at agent startup, not at the first call.** The
  language server starts when the MCP starts (~3.6 s on a Python project —
  measured in §11 of the design), after which the calls themselves are
  fast (~1 s). Kotlin/JVM is more expensive at startup. This is not a hang.
- **`.serena/` at the project root.** Serena itself creates `project.yml`,
  a cache and its own `.gitignore` there. It's the tool's internal
  directory; we don't commit it to git.
- **Complements, doesn't replace.** Semantic search answers "roughly where
  auth is", LSP answers "where `AuthMiddleware.authenticate` is declared
  and who calls it". `grep` remains for text, configs and logs.

## Tests

```bash
# unit
python3 -m pytest tests/unit_tests/lsp/ -q

# integration (real Serena + the real deep-worker binary, no mocks)
python3 -m pytest tests/integration_tests/lsp/ -q -m integration
```

The integration suite includes `test_lsp_cli_wiring.py` — it launches the
**real** `deep-worker lsp status` through the launcher. It exists because
the parser unit tests once missed a wiring defect: the command was
registered in a dead entry point `main.py` and did not work. Details and
the rule for the future — §4.6 of the design.

In strict mode (`DEEP_WORKER_REQUIRE_INTEGRATION=1`) a missing `serena`
binary fails the test instead of skipping it.

## Documents

- [Integration design](docs/DESIGN_LSP_MCP_INTEGRATION.md)
- [Kotlin/Android source sets](docs/DESIGN_KOTLIN_LSP_ANDROID_SOURCE_SETS.md)
- [OSS research](research/RESEARCH_LSP_MCP.md)
