# DESIGN: LSP navigation for agents via a unified MCP (Serena)

**Date:** 2026-07-30
**Branch:** `dev`
**Status:** DESIGN — APPROVED WITH CHANGES (internal reviewer); M1–M3 incorporated,
supplemented with empirical measurements (§11)
**Basis:** [`lsp/research/RESEARCH_LSP_MCP.md`](../research/RESEARCH_LSP_MCP.md)
+ user decisions on 6 open questions (§1.2).

---

## 1. Statement of the problem

### 1.1 Goal

Give deep-worker agents **IDE-style code navigation in a single call**:
go-to-definition, show-usages, find-symbol, document outline, and when
needed — symbol refactoring (rename, replace body).

Key success criterion: **the agent does not do grep archaeology**.
Instead of the `grep → read_file → grep → read_file` chain — a single
`find_symbol` / `find_referencing_symbols` with a precise symbol answer.

This does **not** replace semantic search and grep — semantic answers
"roughly where auth is", LSP answers "where `AuthMiddleware.authenticate`
is declared and who exactly calls it".

### 1.2 User decisions (recorded, not revisited)

| # | Question | Decision |
|---|----------|----------|
| 1 | Default backend | **Serena** |
| 2 | Write tools (rename/replace body) | **auto_approve** |
| 3 | Kotlin LS | **official** `Kotlin/kotlin-lsp` (unless strong counter-arguments) |
| 4 | Own wrapper over OSS | **do not** build, if Serena tool names are acceptable |
| 5 | Number of LS per project | depends on the project's languages, usually 1–2 |
| 6 | CI / integration tests | **in scope** |

**Optionality** — a mandatory requirement: LSP is enabled per-project /
per-agent, like a semantic index. A project without LSP enabled must work
exactly as it does now.

### 1.3 Out of scope

- Own MCP server / own LSP client (decision #4 — we don't write one).
- Changing `ProxyMCPServer` of any provider.
- Changing the External MCP architecture (reuse as is).
- JetBrains-backend Serena (paid).

---

## 2. Key architectural observation

**The connection mechanism already exists and does not need to change.**

deep-worker implements External MCP
(`deep_worker_cli/external_mcp/`, see `DESIGN_EXTERNAL_MCP_TOOLS.md`):

- layered `mcp.json` config: user → project → per-agent → CLI;
- `ExternalMCPClientPool` starts stdio/HTTP MCP servers for the whole
  `AgentProcess` lifecycle, hands `BaseTool` adapters to
  `create_cli_agent(..., external_mcp_tools=...)`;
- per-server `auto_approve` → `auto_approved_tool_names` → HITL bypass;
- `MCPToolNameMangler` — name collisions between servers fail loud;
- a failing server does not bring down the pool (graceful degradation);
- teardown via `AsyncExitStack` (child processes are killed).

The Serena server process is an external stdio process, not a thread inside
the agent, so the "no new threads" policy is satisfied automatically.
Transport — JSON-RPC over stdio, event-driven, **no polling**.

**Consequence:** Serena connects by an entry in `mcp.json`. The main work
of this design is not a "new framework", but:

1. **enable ergonomics** (one command/one flag instead of hand-written JSON);
2. **correct workspace root** (worktree-aware);
3. **a prompt fragment** so the agent actually calls LSP instead of grep;
4. **tests** (unit + integration without mocks);
5. **documentation** of the optional capability.

---

## 3. Architecture

```
AgentProcess (working_dir = worktree root)
   │
   ├─ _start_external_mcp_pool()          ← exists, NOT changed
   │     └─ load_mcp_config(user, project, agent, cli)
   │           └─ ServerSpec(name="serena", type="stdio", …)
   │
   └─ ExternalMCPClientPool                ← exists, NOT changed
         └─ stdio subprocess:
              serena start-mcp-server --context ide
                     --project <workspace-root> --transport stdio
                 │  JSON-RPC (LSP)
                 ├─ pyright-langserver --stdio        (Python)
                 └─ kotlin-lsp                        (Kotlin, optional)
```

One Serena process per agent serves **all** of the project's languages
(decision #5: 1–2 languages — Serena itself starts LS for the languages it
detected in the project; see §7.1 on the start moment).

### 3.1 Why Serena, not a thin bridge

The defining factor — **agent-first addressing by symbol name**.

`mcpls` / `isaacphi` require `file + line + character`. To get those, the
agent first needs grep/read — exactly the chain we are eliminating. Serena
accepts `find_symbol("AuthMiddleware/authenticate")` and solves the
position-search problem itself.

Plus: 40+ languages (Python + Kotlin out of the box), MIT, active
development, the `ide` context already disables the duplicate
file/shell tools.

---

## 4. Components (what we actually write)

New package: `deep_worker_cli/lsp/`

### 4.1 `lsp/serena_config.py` — ServerSpec entry generator

A pure function, no side effects, no I/O at import time:

```python
def build_serena_mcp_entry(
    *,
    workspace_root: Path,
    context: str = "ide",
    auto_approve: bool = True,
    serena_command: str | None = None,
) -> dict[str, Any]:
    """Return the JSON object of the mcp.json entry for Serena."""
```

Result (the format — exactly what the existing `config.py` understands):

```json
{
  "type": "stdio",
  "command": "serena",
  "args": ["start-mcp-server", "--context", "ide",
           "--transport", "stdio", "--project", "<workspace-root>"],
  "auto_approve": true
}
```

**Rationale for the arguments:**

- `--context ide` — the upstream-recommended context for external agent
  harnesses: disables duplicate `read_file`/`list_dir`/
  `execute_shell_command`/`search_for_pattern` (we already have those),
  leaves the symbol tools. Reduces context bloat and agent confusion.
- `--project <root>` (not `--project-from-cwd`) — the cwd of the MCP
  subprocess is inherited from the agent process and is not guaranteed; an
  explicit path is more reliable and worktree-correct (§4.3).
- `auto_approve: true` — decision #2.

### 4.2 `lsp/enable.py` — enabling the option

Writer function for the per-agent/per-project `mcp.json` layer:

```python
def enable_lsp_for_project(project_root: Path, *, force: bool = False) -> Path
def enable_lsp_for_agent(agent_name: str, *, workspace_root: Path, force: bool = False) -> Path
```

Semantics:

- reads the existing `mcp.json` (if any), **merges** the `mcpServers.serena`
  key, leaves other entries untouched;
- `force=False` and a `serena` entry already exists → do not overwrite
  (idempotent, friendly to manual edits);
- atomic file write (tmp + `os.replace`);
- **permissions:** file `0o600`, `.deep_worker/` directory `0o700` — the
  config may contain `${env:…}` references to secrets. No 777/666
  (project policy).

CLI wrapper: `deep-worker lsp enable [--project <path>|--agent <name>]`
— a thin call to the functions above.

### 4.3 Workspace root (worktree-aware)

Root resolution rule:

1. explicit `--project <path>`, if given;
2. otherwise `find_project_root(working_dir)` (looks for `.git`);
3. otherwise `working_dir` itself.

**Important for our environment:** the checkout is a git *worktree*,
where `.git` is a **file**, not a directory. The existing
`find_project_root` uses `git_dir.exists()`, which is also True for a
file → the worktree root is resolved correctly, and Serena receives
the worktree root, not the parent checkout. This is exactly the
behavior we need (decision #5: pool on `AgentProcess`/cwd).

The design does **not** change `find_project_root`; a test locks this
behavior in as a contract (regression protection).

### 4.4 Prompt fragment (without it the feature does not "take off")

Observation from Serena's upstream documentation: clients with strong
built-in tools systematically **ignore** MCP tools — Serena ships
reminder hooks for this. We have no hook mechanism, so we use what we
have: **a system prompt fragment**, added only when LSP tools are
actually present.

`lsp/prompt.py`:

```python
def lsp_prompt_fragment(tool_names: Sequence[str]) -> str: ...
```

Content (concise, ~15 lines): when to call `find_symbol` /
`find_referencing_symbols` instead of `grep`; that `grep` remains for
non-code and textual patterns.

About cold start in the prompt — **carefully**: per the measurement (§7.1)
the first symbol call is cheap (~1 s); what is expensive is the **agent
start**. So we do not promise the agent "the first call will be slow"
(that would provoke a false interpretation of a normal delay as a hang
out of nowhere), but say it shorter: a one-time warm-up delay of the
language server is possible — do not treat it as a hang and do not
repeat the request.

The fragment is injected **only** if `external_mcp_tools` contains
serena tools. No LSP → the prompt does not change by a single byte
(optionality).

**M2 — only mangled names in the prompt.** The agent does NOT see raw
MCP names, but those passed through `MCPToolNameMangler`: canonical
`serena.find_symbol` → `serena__find_symbol`. The prompt must name the
tools **exactly** so, otherwise the agent will call a non-existent name.
Therefore `lsp_prompt_fragment()` takes the **actual** `tool.name` of the
adapters (already mangled) and builds the text from them, not from
hardcoded literals. If an expected tool is not in the list, its line is
not emitted (we don't promise the agent what is not there).

**Actual tool surface** (measured Serena 1.6.1, `--context ide`,
22 tools; the relevant navigation ones):

`find_symbol`, `find_referencing_symbols`, `find_declaration`,
`find_implementations`, `get_symbols_overview`,
`get_diagnostics_for_file`, `rename_symbol`, `replace_symbol_body`,
`insert_after_symbol`, `insert_before_symbol`, `safe_delete_symbol`.

Note: `--context ide` did **not** remove `search_for_pattern`,
`replace_content`, `replace_in_files` and the memory tools (they
duplicate our built-ins). Not a blocker for v1, but in the prompt we
explicitly do NOT recommend them — we steer the agent to symbol tools.
Targeted disabling of the extra (via Serena `tool_exclusions`/mode) is a
follow-up candidate, not a subject of this feature.

### 4.5 Stable tool names

Decision #4 — we don't build our own wrapper. Serena's names are semantic
and self-documenting (`find_symbol`, `find_referencing_symbols`,
`get_symbols_overview`, `rename_symbol`, `replace_symbol_body`).

After mangling the agent will see `serena__find_symbol` and so on
(dot → `__`), valid for all providers and readable. We don't add a
separate renaming layer — it would add a failure point without benefit.

---

### 4.6 CLI wiring: where the subcommands are registered

**A rake we already stepped on (2026-07-30).** The `lsp` command was
originally registered in `deep_worker_cli/main.py` following the pattern of
the neighbouring `mcp` command. The tests passed (they called
`setup_lsp_parser`/`handle_lsp_command` directly), but the real command
failed:

```
$ deep-worker --help | grep lsp      # empty
$ deep-worker lsp status             # error: unrecognized arguments: status
```

Reason: **`main.py` is a dead entry point.** The real launch chain:

```
deep-worker (bash launcher)
  └─ python3 -m deep_worker_cli.cli.agent_command   ← the actual entry point
       └─ main() → create_parser()
```

`main.py` marks itself in its own docstring as "❌ python3 -m
deep_worker_cli.main", and `config.py` calls it the "OLD entry point".
Registering subcommands there does not produce a working command.

**Why we cannot simply add `add_subparsers` to `create_parser()`.**
The main parser has a positional `name` (`nargs='?'`). Adding subparsers
breaks **all** normal launches — verified experimentally:

```
deep-worker myagent  →  error: argument command: invalid choice: 'myagent'
```

**Decision taken.** Route before `create_parser()` in
`agent_command.main()`:

- `_dispatch_subcommand(sys.argv[1:])` compares the first token WHOLLY
  against the names of the known subcommands (the agent name
  `lspagent`/`mcpagent` is not intercepted); tokens starting with `-` are
  ignored;
- handled → returns the exit code, the main parser is not created;
- not a subcommand → returns `None`, the normal agent launch path is
  unaffected;
- the positional `name` and the main parser are **not touched at all**;
- in `--help` the command is visible via `epilog` (subparsers cannot be
  added).

**Regression protection** — integration tests on the REAL binary
(`tests/integration_tests/lsp/test_lsp_cli_wiring.py`), including the lock
`test_normal_agent_launch_still_parses`: a normal launch must not break.
Parser unit tests are fundamentally insufficient here — they are exactly
what missed the original defect.

**Rule for the future:** new CLI subcommands are registered in
`cli/agent_command.py`, not in `main.py`, and must be accompanied by a
subprocess test on the real launcher.

**The neighbouring commands suffered from the same defect** (fixed in a
separate commit at the user's request): `mcp`, `skills`, `threads` — all
were registered only in `main.py`, i.e. `deep-worker mcp list` failed with
`unrecognized arguments: list`. Discovered during verification: a cursory
`deep-worker mcp --help` returned exit 0 and therefore looked working — in
fact argparse treated `mcp` as an AGENT NAME and printed the general help.
The only reliable check is a real subcommand (`mcp list`), not `--help`.

Final routing scheme in `agent_command.main()`:

| Mechanism | Commands | Why |
|---|---|---|
| `_build_subcommand_registry()` | `lsp`, `mcp` | common contract: `setup_*_parser(subparsers)` + `handler(args) -> int` |
| `_dispatch_skills_subcommand` | `skills` | `setup_skills_parser` requires keyword-only `make_help_action`; the handler returns `None` |
| `_dispatch_threads_subcommand` | `threads` | no own `setup_*_parser` — the subcommands are described inline, mapped onto the existing `list_threads_command` / `remove_thread_command` |

All of them are listed in `epilog` — otherwise they are not visible in
`--help`. Coverage:
`tests/integration_tests/cli/test_dead_entrypoint_subcommands.py` (10 tests
on the real binary, including the lock of the normal launch and a check that
agent names like `mcpagent`/`lspagent` are not intercepted).

---

## 5. Language servers

| Priority | Language | LS | Installation | Note |
|---|---|---|---|---|
| 1 | Python | **pyright** | already installed in the environment (`pyright-langserver` in PATH) | Serena default |
| 2 | Kotlin | **Kotlin/kotlin-lsp** (official) | managed download by Serena | JVM, heavy cold start |
| 3 | others | as needed | — | Serena supports 40+ |

Decision #3 — official `Kotlin/kotlin-lsp`. No "strong" counter-arguments
were found: it is the one Serena uses for Kotlin by default, it is
developed more actively than fwcd. The cost is the JVM weight, acceptable
because the Kotlin LS starts only if Serena detected `.kt` files in the
project — on a pure Python project the JVM does not start at all. But if
the project has Kotlin, the cost is paid at **agent start** (§7.1), not
at first use.

---

## 6. Testing

### 6.1 Unit (mocks allowed)

| Test | Checks |
|---|---|
| `test_build_serena_mcp_entry_*` | entry shape, args, absolute `--project`, auto_approve |
| `test_entry_parses_via_existing_config_loader` | the generated entry **actually** passes `_parse_one`/`load_mcp_config` → `ServerSpec` (contract between our generator and the existing loader) |
| `test_enable_is_idempotent` | repeated call does not duplicate/overwrite |
| `test_enable_preserves_other_servers` | other `mcp.json` entries are not lost |
| `test_enable_file_permissions` | file `0o600`, directory `0o700` (anti-regression against 666/777) |
| `test_prompt_fragment_absent_without_lsp_tools` | prompt optionality |
| `test_worktree_root_resolution` | `.git`-as-file → worktree root |

### 6.2 Integration (WITHOUT mocks) — decision #6

`lsp/tests/test_serena_integration.py`, marked and
**skip-if-unavailable** (no `serena` in PATH → skip, not fail):

1. fixture repository: a temporary directory with `pkg/mod.py` containing
   a class `Target` and its call from another file;
2. start a **real** Serena stdio through the **existing**
   `ExternalMCPClientPool` (not a hand-rolled client — we test the real
   execution path);
3. call `find_symbol("Target")` → expect a non-empty result with the
   correct `relative_path`;
4. call `find_referencing_symbols` → expect the calling file;
5. teardown: `pool.stop()` — make sure no child process remains.

**Hard time budget** (the "integration tests must not hang" policy,
feedback 2026-07-28, + reviewer M1): the whole test ≤ 60 s, event-driven
waiting via `asyncio.wait_for`, guaranteed teardown in `finally`.

Concrete requirements:

- `asyncio.wait_for(ExternalMCPClientPool.create(...), timeout=...)` —
  **the pool start itself is also under timeout** (otherwise a hung
  Serena would hang CI, see §7.1);
- every `call_tool` — under its own `wait_for`;
- on timeout — **fail loud** with a clear message, not skip;
- `finally: await pool.stop()` + check that no child process remains;
- **skip-if-unavailable**: no `serena` in PATH → `pytest.skip`
  (an environment without the tool must not "redden" the build); with
  the strict-mode flag set (`REQUIRE_INTEGRATION`) — on the contrary,
  hard fail.

Measured budget (§11): ~3.6 s start + ~1 s first call → 60 s is
sufficient with a large margin.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Agent start blocking (M1)** | see §7.1 — measured, acceptable for v1 |
| Cold start of LS (especially Kotlin/JVM) | the cost is paid at **agent start**, not at first call (§7.1, measured); integration test only on the Python fixture; Kotlin — a separate measurement and, if needed, a separate design |
| Serena not installed | the pool logs a warning and continues without it (existing graceful degradation); `enable` prints the `uv tool install -p 3.13 serena-agent` instruction |
| Agent ignores LSP tools | prompt fragment §4.4; measure on real tasks |
| Write tools corrupt code | `auto_approve` — a deliberate decision #2; safety net — git (work on `dev`, everything under review/commit control) |
| Tool name collision | existing `MCPToolNameMangler` fail loud |
| RAM with several LS | 1–2 languages per project (decision #5); Serena starts LS only for languages actually found in the project |
| Upstream drift of Serena CLI | args locked in one place (§4.1) + integration test catches breakage |
| Serena writes `.serena/` to the project root | §7.2 — documented, `.gitignore` |

### 7.1 Cold start and agent start blocking (M1)

**Mechanics (verified against the code).** `ExternalMCPClientPool._start_all`
executes `await session.initialize()` and `await session.list_tools()`
**without a time budget**, and `AgentProcess` **waits** for the pool to be
created before the graph is ready. So a hung MCP server blocks the agent
start. The pool's graceful degradation triggers on **exception**, but not
on **a hang** — this is an existing property of External MCP, not
introduced by us.

**The lazy-LS assumption was checked by measurement — the assumption did
NOT hold in its original form.** Running real Serena 1.6.1 on a Python
fixture (`/tmp/lsp_probe`, `--context ide --transport stdio --project <fixture>`):

| Stage | Time |
|---|---|
| `initialize()` | **3.60 s** |
| `list_tools()` (cumulative) | **3.61 s** |
| first `find_symbol` | **0.96 s** |

From Serena's logs: the language server starts **within** the MCP start
(`init_project_services` → `Language server initialization completed in
1.007 seconds`), i.e. **not** fully lazy — the LS comes up at startup,
not at first use. The tool catalog is returned quickly, and the first
symbol call is already cheap (cache warmed).

**Conclusion for v1:** ~3.6 s added to agent start on a Python project —
acceptable; we don't change the pool architecture (M1 closed as a
"documented risk with a measured value"). **Kotlin/JVM is expected to be
more expensive** — if a measurement on a real Kotlin project shows an
unacceptable delay, that is a basis for a separate design (lazy
start/budget in the pool), not for an edit within this feature.

**Mitigation in tests (M1, §6.2):** `ExternalMCPClientPool.create` and all
tool calls are wrapped in `asyncio.wait_for` with a total budget of ≤60 s;
on timeout — fail loud, teardown in `finally`, no hanging.

### 7.2 Serena side artifacts in the project root

Serena creates `<project>/.serena/` in the root: `project.yml`,
`project.local.yml`, `cache/`, `memories/`, `.gitignore`
(self-ignoring). Actual permissions: directory `755`, `project.yml` `600`.
This is upstream behavior, we don't control it; our own
`mcp.json` is created with `0600`/`0700` (§4.2).

Action: document in `lsp/README.md` that `.serena/` is the tool's
internal directory (we don't commit it to git; Serena puts its own
`.gitignore` there).

| Policy | Compliance |
|---|---|
| Polling prohibited | JSON-RPC stdio event-driven; no busy-wait |
| No new threads | Serena — an external **process**, not a thread; the pool already exists |
| File permissions (not 777/666) | `mcp.json` `0o600`, `.deep_worker/` `0o700` (§4.2) |
| Don't change architecture/contracts | External MCP is reused **as is**; new code — additive `lsp/` package |
| Non-blocking approaches | calls through the existing async adapter + `BoundedToolCallMiddleware` |

---

## 9. Implementation plan (TDD)

1. RED: unit tests §6.1 for `serena_config.py` / `enable.py` / `prompt.py`.
2. GREEN: implement the three modules.
3. Inject the prompt fragment at the agent assembly point (additive,
   conditional on the presence of serena tools).
4. CLI `deep-worker lsp enable`.
5. Integration test §6.2 against real Serena.
6. Documentation: `lsp/README.md` (how to enable), a note in the
   agent documentation about the new optional capability.
7. Code review (internal reviewer) → commit → fast-forward into `master`.

---

## 10. Closed decisions (former open questions) — M3

Reviewer answers, recorded as decisions:

| # | Question | **Decision** |
|---|----------|--------------|
| 1 | CLI command placement | Group **`deep-worker lsp enable\|status`**. Convenience flag at agent creation — optionally later, **not** in v1. |
| 2 | Is `lsp enable` needed | **Yes, keep it.** Ergonomics is the product surface; "generator + docs" is not enough for "an option like semantic index". |
| 3 | Where the prompt fragment goes | **Into the system prompt** when tools are present (not only `AGENTS.md`): `AGENTS.md` loads even without LSP and clutters non-LSP projects. Visibility in Full Dialog — matches POLICIES §6. Inject **after** the pool's tools are known — at the same point where external tools enter `create_cli_agent`. |

---

## 11. Empirical verification (performed before implementation)

To avoid designing on assumptions, real Serena was installed and run
before writing any code.

**Environment:** `uv 0.10.0`; `uv tool install -p 3.13 serena-agent` →
**serena-agent 1.6.1**; the binary in PATH — **`serena`** (plus
`serena-agent`, `serena-hooks`). This closes the reviewer's non-blocking
remark about the command name: `build_serena_mcp_entry` uses
`"command": "serena"`, and `lsp status` checks exactly that.

**What was confirmed:**

- `serena start-mcp-server --context ide --transport stdio
  --project <root>` works as intended (§4.1);
- `find_symbol("Target")` returned a precise result:
  `{"name_path": "Target", "kind": "Class", "relative_path": "pkg/mod.py",
  "body_location": {...}}` — i.e. the target UX "one call → precise
  answer" is achievable;
- the MCP pipeline is fully compatible with our `ExternalMCPClientPool`
  (the same `mcp` SDK, stdio, `initialize`/`list_tools`/`call_tool`);
- correct teardown: the LS receives `shutdown`/`exit`, the process
  terminates (`exit code -15`), no orphans remain.

**What did NOT hold and was accounted for in the design:**

- the LS starts **at MCP start**, not at the first request → §7.1
  (measured 3.6 s, risk documented);
- `--context ide` does **not** disable `search_for_pattern`/
  `replace_content`/memory tools → §4.4 (the prompt steers to symbol
  tools; disabling is a follow-up);
- Serena writes `.serena/` to the project root → §7.2.
