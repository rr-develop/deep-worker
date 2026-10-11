# Tool selection: turning deep-worker tools off, and back on one by one

## 1. The need

Measuring whether a tool earns its place requires an agent that has *only*
that tool. Today there is no way to say so: the tool list is assembled
unconditionally, and `AgentConfig.tools` — which reads like the knob for
exactly this — is dead. It is declared at
`agent_config_pkg/agent_config.py` and referenced nowhere; setting it
changes nothing. A field that looks like a control but silently does
nothing is worse than no field, so this design gives it meaning.

The requirement, in the owner's words: default behaviour must not change;
turning *everything* off must be simple (a star); and after turning
everything off it must be possible to name the survivors.

## 2. The knob

One config field and one CLI flag, carrying the same tri-state value.

```yaml
tools: []                      # absent / empty  → every tool, as today
tools: ["-*"]                  # no deep-worker tools at all
tools: ["-*", "read_file"]     # only read_file
tools: ["read_file", "grep"]   # the same, without the star
tools: ["-execute"]            # everything except execute
tools: ["*", "-execute"]       # the same, said out loud
```

- **Empty or absent** — the default, and it must stay byte-identical to
  current behaviour. This is the case that protects every running agent.
- **`-*`** — deny all. On its own it leaves the agent with no deep-worker
  tools.
- **`-*` plus names** — deny all, then re-enable exactly those names. This
  is the measurement mode.
- **A bare list of names** — the same allow-list without the star. Getting
  *more* tools than you listed would be the surprising reading, and this is
  the shape a YAML `tools:` list already had while the field did nothing,
  so those configs now mean what they always appeared to mean.
- **`-name`** — subtractive: keep everything except the named tools. Useful
  for the mirror-image question ("is the agent worse *without* X?").
- **`*`** — every tool, which is already the baseline. Redundant on its own
  or beside `-name`, and permitted so that "all except X" can be written
  the way it is spoken.

`*` means EVERY tool. Glob, `.gitignore`, and every ACL syntax a reader
arrives with agree on that, so the negation is spelled `-*` — the same
`-name` rule that removes one tool, applied to all of them.

### The spelling changed, and the old one is an error

`*` originally meant deny-all, so `["*", "read_file"]` meant "only
read_file". That reading was backwards: a star that says "all" while
meaning "none" is a trap the reader walks into once per config.

Under the current grammar the same spec would mean "every tool, plus
read_file" — it would WIDEN a restricted agent. Since specs live in
`session_metadata.json` and survive a restart, re-interpreting them
silently would hand tools back to an agent that was deliberately confined,
with nothing on screen to say so.

Two legacy spellings do this, and **both are launch errors** carrying the
rewrite:

| Legacy spec | Used to mean | Would now mean |
|---|---|---|
| `["*"]` | no tools at all | **every tool** — the whole registry |
| `["*", "read_file"]` | only read_file | every tool, plus read_file |

The bare `["*"]` is the larger widening of the two — nothing to everything —
and it is the form earlier revisions of this document taught, so configs in
the wild carry it. Refusing it costs the grammar nothing: "every tool" is
already what an empty spec says, so a lone star is only ever redundant.

`["*", "-execute"]` is unaffected: `*` is the baseline being subtracted
from, so the meaning is the same with or without it.

Two combinations are rejected rather than resolved by picking a winner:
`-*` beside `-name` (redundant — `-*` already denies everything), and
a bare allow-list beside `-name` (two different selections stated at once).
Unknown names are also rejected: a typo that silently yields an agent with
fewer tools than intended would corrupt the very measurement this feature
exists to support. Validation spans the assembled tool list *plus* the
names middleware contribute, asked of the middleware themselves rather
than kept in a constant that would drift.

### `none`, because argparse eats a leading dash

`argparse` treats any value beginning with `-` as the next option, so
`--tools -*` is rejected before the parser hands anything to this feature;
only `--tools=-*` survives. Requiring the `=` form for *deny everything* —
the single most common spec — is a trap, so **`none` is an accepted alias
for `-*`**. It needs no `=`, no quotes, and no care about shell globbing.

```bash
deep-worker probe --tools none                  # no deep-worker tools
deep-worker probe --tools none,project_memory   # only project_memory
deep-worker probe --tools='-execute'            # everything except execute
```

## 5. Changing the selection without restating it

`--tools` is absolute: it replaces whatever the agent had. Adding one
tool to an agent restricted to five therefore means retyping all six, and
getting that list wrong silently narrows the agent — the same failure the
unknown-name check exists to prevent.

`--tools-add` and `--tools-remove` (alias `--tools-rm`) are relative to
the selection already in force. Both are comma-separated and repeatable.

```bash
deep-worker -r probe --tools-add grep              # keep everything, gain grep
deep-worker -r probe --tools-rm execute,write_file # keep everything else
deep-worker -r probe --tools-add 'history_*'       # a whole family
```

### Patterns, and why the specific one wins

A name containing `*`, `?` or `[` is a glob (`fnmatch`), so families of
tools can be named at once. The interesting case is an exception to a
family:

```bash
deep-worker -r probe --tools-rm 'history_*' --tools-add history_at
```

Both rules match `history_at`, so one has to win, and the rule that names
the tool *more precisely* does. Specificity is the length of a pattern's
literal (non-wildcard) part, with a literal name ranked above any pattern
that ties with it — so an exact name always beats `history_*`, which in
turn beats `*`. This is the same "most specific rule wins" principle CSS,
`.gitignore` and routing tables all use, and it is what makes the
exception above read the way it is spoken.

Two rules that are *equally* specific and disagree — `--tools-add 'lsp_*'`
beside `--tools-rm 'lsp_*'` — have no non-arbitrary winner, so they are a
launch error rather than a silent choice. Letting flag order decide would
be worse than it looks: the spec is re-parsed from `session_metadata.json`
on every restart, where order is an implementation detail.

A tie does not require the same pattern twice, which is the part worth
stating plainly. `serena__*` and `*d_symbol` carry eight literal
characters each and both match `serena__find_symbol`; grouping by the
pattern string cannot see that, because whether two patterns overlap is a
question about which tools exist. So the check runs twice, at two
different moments:

* **At launch, against the real inventory** — equal specificity, opposite
  directions, and at least one tool both match is an error naming the
  tool they fight over (`validate_against_inventory`).
* **In `permits()`, for any tie that still gets through** — a spec read
  back from `session_metadata.json` is parsed with no inventory in hand,
  so the resolution cannot merely be asserted away. It resolves to
  **deny**. Of the two directions only deny is safe to pick blind:
  guessing wrong withholds a tool rather than handing an agent one it was
  deliberately confined away from.

Without that tie-break the outcome moved with position, and because
`merge_increments` appends, the *newer* rule was the one that lost:
`--tools-rm 'ab*'` followed by a restart with `--tools-add '*ab'` left
`abab` denied, so the add silently did nothing — the exact class of quiet
miss that makes an unmatched pattern an error in the first place.

### A pattern that matches nothing is an error too

The unknown-name rule extends to patterns. `--tools-rm 'lsp_*'` when no
tool starts with `lsp_` would otherwise be a no-op that reads as a
restriction — the same silent miss a typo'd literal causes, which is
precisely what §2 refuses to allow.

The validated set is the deep-worker tools, so it does NOT include tools
contributed by external MCP servers (`serena__*`, `repo_graph__*`, …):
those are attached by the provider after the selection is checked. Naming
one is therefore rejected as unknown. This is the same boundary §4 draws
for the ACP providers — the selection governs deep-worker tools, and the
warning at startup says so.

Both flags reach the AgentConfig the process is *built* from as well as
the launch_config it is *restarted* from. Resolving them for only one of
the two is not a cosmetic split: while `launch_config` alone carried the
increments, `deep-worker probe --tools-rm 'lsp_*'` reached `idle` with a
pattern matching nothing, because the unknown-name check runs against the
AgentConfig spec and never saw the rule.

### How they survive a restart

The increments are folded into the stored spec as `+name` / `~name`
entries beside the base selection, so `session_metadata.json` holds one
self-contained spec that is re-parsed on every start. Storing them as a
separate layer would re-apply them forever: `--tools-add grep` once would
keep re-adding `grep` after a later `--tools-rm grep`, because both layers
would still be on file. Re-stating the same pattern replaces the earlier
rule rather than stacking a second one, which is what makes a repeated
restart idempotent.

Combining `--tools` with an increment in one command is allowed and reads
left to right: `--tools` sets the base, the increments apply on top.

A spec that *starts* with `-name` still needs the `=` form for the same
argparse reason; the help text says so. Config files are unaffected — YAML
and JSON carry the list directly, so `["-*"]` and `["none"]` are equally
fine there.

CLI: `--tools 'none,read_file,grep'`. The flag is stored in `launch_config`
so a restart reproduces the same restricted agent.

## 3. Where the filter goes

`create_cli_agent` assembles tools in one place (`agent.py`, the
`_dedupe_tools_by_name_priority` line), but that line is not the whole
truth: middleware contribute their own tools afterwards
(`project_memory.py`, `subagents.py`, `filesystem.py`,
`history_access.py`, `history_memory.py`, …). A filter applied only to the
assembled list would leave middleware tools untouched and produce a false
"all tools off".

So the filter is applied **at the boundary where the model sees tools**,
not where the lists are built — one place that no middleware can bypass.

### 3.1 Withholding a schema is not enforcement

The first implementation stopped at the filter above, and that was not
enough. `wrap_model_call` governs what the model is *offered*; it does
not govern what LangGraph will *run*. `ToolNode` resolves an incoming
call against the graph's own registry — the full tool list passed to
`create_deep_agent` — so a call for a tool that was never offered still
reaches the real implementation.

Models do issue such calls. Verified live on 2026-08-30: an agent
launched with `--tools '*,project_memory'` logged

```
tool selection (only project_memory): 1 of 23 tools offered to the model
```

and then successfully ran `ls` twice and dispatched `read_file`. The
filter was working exactly as designed; the design was incomplete.

So the selection is applied a **second time, at dispatch**.
`ToolSelectionMiddleware.wrap_tool_call` / `awrap_tool_call` check the
call's name against the same `permits()` predicate and, when it is
denied, return a `ToolMessage(status="error")` *instead of* calling the
handler. The tool body never executes. The refusal names the tool, says
the restriction is deliberate, and explicitly forbids both retrying and
routing around it — an agent told only "no" tends to go looking for
another way to the same capability.

A nameless call is passed through rather than refused: it cannot be
matched against the selection, `ToolNode` rejects it on its own, and
inventing a refusal here would misreport why it failed.

This is why the enforcement tests drive a **real compiled agent with a
real `ToolNode`**, and assert that the tool's side effect did not happen
— not merely that an error came back. A middleware exercised in isolation
cannot see this class of bug: the isolated filter was correct throughout
the entire period the live agent was bypassing it.

### 3.1a A denied call must never reach the approval prompt

The dispatch barrier refuses the call, but it refuses it *late*. With
`interrupt_on` configured for a tool the selection denies, the observed
behaviour was an approval interrupt: the user was asked to authorise a
tool the agent is not allowed to have, and only after saying yes did the
barrier refuse it. Asking permission for something already forbidden is
not a safety prompt but a nonsensical question — and a user who approves
is entitled to expect the call to happen.

The first hypothesis was that this is a middleware-ordering problem, and
it is not. `HumanInTheLoopMiddleware` does its work in an **`after_model`
graph node**, not in `wrap_tool_call`. That node runs as soon as the model
returns — strictly before the whole `wrap_tool_call` chain the barrier
lives in — so no placement of the selection inside that chain can get in
front of it. Verified on the compiled graph:

```
model -> HumanInTheLoopMiddleware.after_model -> ToolSelectionMiddleware.after_model
```

So the selection grew a **third cut, in its own `after_model`**:
`_prune_tool_calls` removes denied calls from the last `AIMessage` and
answers each with the same refusal text the barrier would have produced.
HITL then never sees them. Rewriting the `AIMessage` via `model_copy`
keeps its id, so the checkpoint gets a replacement rather than a second
copy.

This makes the middleware's position order-sensitive in **two opposite
directions**, which is why the placement in `create_deep_agent` looks
backwards at first glance:

* `wrap_tool_call` wrappers nest **in list order** — earlier is further out;
* `after_model` nodes chain in **reverse list order** — later runs first.

The selection therefore goes **last**, after `HumanInTheLoopMiddleware`,
in all three stacks that build one (main, general-purpose subagent, custom
subagent spec). The test that pins this asserts no interrupt was raised
*and* that the tool body never ran; reverting either the ordering or the
`after_model` cut fails it.

### 3.2 The restriction must survive delegation

A subagent is a **separate graph** with its own middleware stack — and
its own `FilesystemMiddleware`, which contributes `ls`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep` all over again. Middleware
installed on the main agent does not reach it.

So `--tools '*,task'` produced a main agent that correctly refused every
file tool, and a subagent that ran them. Reproduced before the fix: with
the denied tool's body instrumented, the main graph refused the call and
the subagent executed the identical one. This is the worst possible place
for the hole — delegation is exactly where an agent goes once its own
tools are gone.

`create_deep_agent` therefore propagates into **every** subagent graph it
builds — the general-purpose one and each custom spec — any middleware
that declares `applies_to_subagents = True`. The same instance is reused,
so the selection and its log line stay singular. It is appended **last**
in each stack, after that stack's own filesystem/skills middleware, for
the same ordering reason as §3.

The opt-in is a plain attribute rather than an `isinstance` check because
`deepagents` sits *below* `deep_worker_cli` and must not import from it.

`CompiledSubAgent` is the one shape that cannot be covered: its graph was
compiled by the caller, with its middleware already fixed. Rather than
fail silently, `create_deep_agent` logs a warning naming the subagent when
a selection is in force. No deep-worker agent can reach that branch —
`_merge_common_tools_into_subagents` rejects compiled subagents outright —
but `create_deep_agent` is public API, so the limit is stated rather than
assumed away.

### 3.3 The prompt must agree with the schemas

Removing a tool's schema is not sufficient. Each tool-owning middleware
also appends its own advertisement to the system prompt — "You have
access to an `execute` tool", "This project has a semantic search
index", the skills catalogue, the `task` spawner section. Left in place,
those sentences tell the model it holds tools it cannot call, which
corrupts the measurement this feature exists to produce.

`wrap_model_call` wrappers nest outer-to-inner, so the selection
middleware — appended last — is handed the request *after* every other
middleware has contributed its text. Because `append_to_system_message`
stores each contribution as a **separate content block**, those blocks
can be identified by their headings and dropped. A section is removed
only when *none* of the tools it advertises survived; when at least one
survives the section stays verbatim, since that tool still needs its
instructions. Text no middleware owns — the agent's own prompt, the
owner's instructions — is never touched.

Matching by heading is the weak point: a heading that drifts would make
its section un-droppable, restoring the exact inconsistency this solves.
`test_every_section_marker_matches_the_real_prompt_constant` pins every
marker against the real constant it comes from, so drift fails the suite
rather than silently degrading the prompt.

Suppression is verified by running the genuine middleware chain and
asserting on the `ModelRequest` the model layer receives — not on a
hand-built stub, which could agree with the filter and still disagree
with production.

### 3.4 The static prompt needs its own cut

The middleware can only prune what passes through it. The largest single
source of tool advertising never does: `default_agent_prompt.md` and the
sections `get_system_prompt` assembles are baked into the prompt *before*
any middleware runs. Left alone, a `*,project_memory` agent still read
full instructions for `read_file`, `execute`, `send_message_to_agent`,
`restart_self`, `web_search` and `write_todos` — an invitation to call
them, and (before §3.1) an effective one.

Four cuts, one rule — *a whole section whose entire subject is a removed
tool goes; nothing else is touched*:

| Source | Mechanism |
|---|---|
| `default_agent_prompt.md` | `strip_disabled_prompt_sections` matches headings, removing each section up to the next heading of equal-or-shallower depth |
| Sections built in `get_system_prompt` (skills, web search, todos, path-handling rules) | rendered conditionally |
| Identity section (`send_message_to_agent` protocol) | replaced by a statement that messaging is disabled |
| Per-provider invocation notes (`proxy_tool_hints`) | command-tool forms and the CRITICAL block filtered by the same predicate |

Two deliberate non-cuts. The **working directory** stays whatever the
selection: it is where the agent is, and tools that never touch the
filesystem are still scoped to it — only the path-handling *rules* depend
on file tools surviving. And grok's note keeps naming `read_file`/`grep`
as that CLI's **own** built-ins: the sentence warns the model not to
mistake them for ours, our selection cannot switch them off (§4), so the
warning stays true.

The identity section is rewritten rather than deleted because peer
messages still *arrive* — the inbox is not a tool. An agent that cannot
reply must be told so, not left to discover it mid-turn.

The default path is pinned **byte-for-byte** against the pre-feature text
for every provider category. Rewrapping a line would silently change the
prompt of every existing agent, so the unrestricted rendering keeps its
original line breaks even where the restricted one reads better.

Heading matching carries the same drift risk as §3.3, and gets the same
tripwire: `test_base_prompt_sections_have_not_drifted` fails the suite if
a heading in the markdown stops matching the table.

## 4. Native tools: what is and is not achievable

The owner asked that denying everything also deny the *native* tools of
stateful providers. Investigation shows this is achievable for some
providers and architecturally impossible for others. Reporting the limit
is part of the deliverable.

| Provider | Native tools | Channel | Achievable |
|---|---|---|---|
| `claude-code-sdk` | Claude Code built-ins | `ClaudeAgentOptions.disallowed_tools` → `--disallowedTools` | **Yes** — already always-on; a hand-maintained deny-list |
| `codex` | Codex natives | `--disable <feature>`, `--config web_search="disabled"` | **Partly** — 7 natives have no flag |
| `claude-code` (ACP) | 15 built-ins | none | **No** |
| `grok` (ACP) | grok built-ins | none | **No** |

The ACP verdict is not an oversight on our side; it is a property of the
protocol. `session/new` carries exactly `{cwd, mcpServers}`
(`acp/acp_client.py`) — there is no tool-configuration field to send. The
passthrough module states the same conclusion in its own words: ACP CLIs
"run their local built-ins … LOCALLY: they cannot be told not to … So
prevention is impossible". The repo already responded to this by building
observe+passthrough instead of prevention.

Codex's residue is enumerated in `codex_provider/chat_model.py`:
`apply_patch`, `update_plan`, `view_image`, `request_user_input`,
`list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`.

### The honest consequence

On ACP providers a measurement run is **not hermetic**: the agent can
still reach for `Bash`/`Read`/`Edit` regardless of this setting. Silently
letting the user believe otherwise would poison exactly the experiment
they described. Therefore, when tools are restricted on a provider whose
natives cannot be suppressed, deep-worker **says so** — once, at startup,
naming the provider, the tools it cannot block, and the provider to use
instead for a hermetic run.

That notice is printed to stderr **before `daemonize()`**, on both the
normal launch path and the `--restart` path. This placement is the whole
point: agents run detached, so anything emitted after the fork lands in
a per-agent log file the user has no reason to open, and the disclosure
would exist only in principle. The logger warning inside
`create_cli_agent` is kept as a record in that log, but it is not the
disclosure. An unrestricted launch — the default — prints nothing.

The system prompt additionally carries the restriction as an
instruction, explicitly labelled a preference rather than enforcement,
matching what the ACP layer itself says it can guarantee.

The prompt is not a substitute for enforcement and is not presented as
one. For a hermetic experiment the usable providers are `claude-code-sdk`
(full) and `codex` (minus the 7 listed natives).

Note the scope boundary against §3.1: the dispatch barrier is absolute
for **deep-worker** tools on **every** provider — those calls travel
through our middleware and are refused before execution. It does not and
cannot reach a provider CLI's own built-ins, which never enter our graph
at all. That, and only that, is what the caveat above discloses.

## 5. What this design deliberately does not do

- It does not invent an ACP tool-suppression channel. There is none to
  invent; a system-prompt request is what the protocol allows, and it is
  reported as such rather than dressed up as enforcement.
- It does not silently keep an "essential core" of tools alive under `-*`.
  `-*` means none. Infrastructure tools (`send_message_to_agent`,
  `restart_self`) are subject to it like any other — an agent under
  measurement should be measurable, and hiding exceptions inside the
  implementation would be its own kind of lie. Turning them off has real
  consequences (no inter-agent replies), which the startup notice states.
