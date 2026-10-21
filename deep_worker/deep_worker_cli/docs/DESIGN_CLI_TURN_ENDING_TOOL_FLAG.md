# Design: `--turn-ending-tool` CLI flag (channel (b), part 1)

**Status:** rev 8 (2026-08-13) — SHIPPED as `aa70703` (`dev`, FF to `master`). DESIGN + CODE SIGN OFF: reviewer. LIVE-ACCEPTED end-to-end by the first consumer, who reported zero behavioural deviations from the contract (§8 rev 7). rev 7 fixed two documentation defects that acceptance exposed; rev 8 tightens the acceptance evidence so it claims only what it observes (§8 rev 8). Doc revisions awaiting DESIGN SIGN OFF; not committed.
**Branch:** shipped on `dev` as `aa70703`, fast-forwarded to `master` (both at `aa70703`).
**Follows:** `docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md` §3.7 (deferred channel (b) — CLI variant).
**Precedent for restart round-trip:** `--mcp-config` / `--mcp-server` in `deep_worker_cli/cli/agent_command.py` (already survives `--restart` end-to-end).

## 1. Problem

`create_cli_agent(..., turn_ending_tools=[...])` exists (shipped 021c5f8) but agents launched by the stock CLI (`deep-worker <name> …`) have no launcher code of their own, so they cannot pass anything to that parameter. The only real consumer today — an extractor agent (`rg_extractor`) — is exactly such an agent (started via `asyncio.create_subprocess_exec` on the `deep-worker` binary by the consumer's own launcher code). A Python-launcher workaround was ruled out because it would fork the entry point: `create_cli_agent()` does NOT read `mcp.json`; external MCP tools reach the graph via `AgentProcess._start_external_mcp_pool()` (`agent_process.py:1467`), which the launcher does not run.

Owner has authorized building the CLI channel now.

## 2. Scope

- **In scope:**
  1. New repeatable flag `--turn-ending-tool <name>[:<field>]` on `deep-worker <name>`.
  2. Restart round-trip: the flag survives `--restart` via `merge_restart_config()` and `build_launch_config()` — modeled 1-for-1 on `--mcp-config` / `--mcp-server`.
  3. Startup diagnostics: two machine-greppable log lines per registered spec (`turn_ending_tool_registered` on success, `turn_ending_tool_not_found` on mismatch), emitted AFTER the final tool inventory is known.
  4. Documentation: README section for the flag, with the working `rg_extractor` example.
- **Out of scope:**
  1. Alternate value syntax (`=<expected>`, dotted paths, arbitrary predicates from CLI). Two forms only.
  2. Echo from a CLI-registered tool. CLI channel always uses `echo_field=None` (jump-only, no injected AIMessage).
  3. Other channels from §3.7 (per-agent config file `turn_ending_tools.json`, per-server hint in `mcp.json`) — not requested by any consumer.
  4. Removing `send_message_to_agent` from the built-in registry — it stays a hard-coded turn-ending tool.
  5. `AgentStatus.ERROR` on startup for a not-found name — see §3.4.

## 3. Design

### 3.1 The flag

```
--turn-ending-tool <name>[:<field>]
```

- `action="append"`, `default=[]`, repeatable. Same shape as `--mcp-config`.
- Values are opaque strings until parsed (see §3.2); argparse does not split them.
- `dest="turn_ending_tools"` on the argparse namespace. The name matches `create_cli_agent`'s existing keyword-only parameter, so there is exactly one vocabulary word for this concept end-to-end.
- Flag is added to BOTH parsers: the main one (`agent_command.py:959+`) and the SUPPRESS-defaulted detector inside `_explicit_cli_flags()` (`agent_command.py:218-266`). The detector wiring is needed for one edge case only — explicitly passing an empty override (which is not currently expressible for list flags anyway; adding it keeps the pattern uniform with future work).

Working example (tool name verified from the live `rg_extractor` agent's `mcp.json`, then at its legacy location; the agent config now lives in the agent's per-agent config directory. Provider/model are the consumer's real ones):

```
deep-worker rg_extractor --provider deepseek --model deepseek-chat \
    --dir /path/to/workdir -m -sp \
    --turn-ending-tool code_graph__submit_fragments:document_closed
```

⚠️ Earlier revisions of this document printed `--provider deepworker_a2a --model gpt-5-mini` here. That is NOT an accepted `--provider` value (`argparse` choices are `deepseek`, `gemini`, `openai`, `claude-code-sdk`, `codex`, `grok`), so the command could not be run verbatim. Caught by the consumer during live acceptance, 2026-08-13. Lesson: a documented command is only "verified" if it was actually executed.

The registered name is the **graph-side** name (post-mangling). For an external MCP server named `<server>` exposing tool `<tool>`, the graph name is `mangle(f"{server}.{tool}")` (see `deep_worker_cli/external_mcp/mangler.py:40` — replaces `.` and `/` with `__`). For the code-graph MCP server + `submit_fragments` that is `code_graph__submit_fragments`. Users MUST NOT pass the provider-visible form (`mcp__…`); the middleware matches on `ToolMessage.name`, which carries the graph name, not the invocation form.

### 3.2 Value parsing

Two forms exactly:

| CLI value | Parsed spec | Predicate |
| --- | --- | --- |
| `<name>` | `TurnEndingTool(tool_name=<name>)` | `bool(d.get("success"))` — default |
| `<name>:<field>` | `TurnEndingTool(tool_name=<name>, is_success=<λ>)` | `d.get("<field>") is True` |

**Grammar rules (normative).** The following are validated the same way in every place a spec string is accepted (CLI, --restart re-parse, direct `AgentProcess(...)` ctor). Any violation raises `TurnEndingToolFlagError(argparse.ArgumentTypeError)`:
- `<name>` MUST be non-empty and MUST NOT contain `:`.
- `<field>` (if the `:` is present) MUST be non-empty. `foo:` is invalid.
- `<field>` MUST NOT contain `.`. Dotted paths are out of scope; nested contracts are a later change if they ever become real.
- **No whitespace anywhere in the spec string.** Not around `<name>`, not around `<field>`, not inside either. `" foo"`, `"foo "`, `"foo: bar"`, `"foo :bar"`, `"a b"` are all invalid. Motivation: whitespace-tolerant grammars silently paper over quoting bugs in shell pipelines (`"$var"` expanding with trailing space) and would let two visually different specs deduplicate to the same middleware key — the whole point of `DuplicateTurnEndingToolError` fails then. Test [R1] `test_whitespace_rejected` pins this for both positions.
- At most ONE `:`. `"foo:bar:baz"` is invalid.

**Predicate semantics** (unchanged from rev 1, restated for completeness):
- The `:field` predicate uses `.get()`, not `[]` — key absence and value `false` behave IDENTICALLY (turn continues). Per owner: "a missing key and `false` must behave identically — the turn does not end". This is normative, not incidental: the underlying tool contract on the `rg_extractor` side is "the key appears ONLY when the document is closed", so any other result — absent key, `false`, unexpected type — must never end the turn.
- `is True` (identity, not truthiness): only the boolean literal `True` ends the turn. `1` does not, `"true"` does not, `["yes"]` does not. Reviewer [R1] specifically endorsed `is True`.
- Built with an explicit default argument to avoid Python's late binding: `is_success=lambda d, _f=<field>: d.get(_f) is True`. Test guards this specifically.

**Layered validation (fixes [R1] gap 1 and [R2] collision path).** There are TWO validation levels — single-string grammar and full-list constraints. Both live in `middleware/tool_completion.py` and are called from every entry point (argparse `type=`, --restart re-parse, `AgentProcess.__init__`).

**Level 1 — single-string grammar.** One function, three thin wrappers so `argparse` keeps raw strings on the namespace and `_parse_turn_ending_tool_specs()` produces `TurnEndingTool` objects.

```python
# deep_worker_cli/middleware/tool_completion.py

def _validate_turn_ending_spec_str(raw: str) -> tuple[str, str | None]:
    """Grammar validator. Returns (name, field-or-None) or raises
    TurnEndingToolFlagError. The single source of truth for the grammar."""
    ...  # applies all rules above

def _canonical_turn_ending_spec_str(raw: str) -> str:
    """argparse `type=` handler. Validates via _validate_turn_ending_spec_str
    and returns the CANONICAL raw string (which today == the input, since
    whitespace is rejected and there is nothing to normalize). Guarantees:
    the argparse namespace attribute `turn_ending_tools` is always
    `list[str]`, JSON-serializable, ready for build_launch_config()."""
    _validate_turn_ending_spec_str(raw)  # raises on invalid
    return raw

def _parse_turn_ending_tool_specs(raw_specs: Iterable[str]) -> list[TurnEndingTool]:
    """Full parser. Called at every entry-point boundary (see §3.5) to
    turn validated raw strings into TurnEndingTool objects using the
    default-arg late-binding fix. Also runs the FULL-LIST validator
    below so cross-entry collisions raise at the same boundary as
    single-string grammar errors."""
    parsed: list[TurnEndingTool] = []
    for raw in raw_specs:
        name, field = _validate_turn_ending_spec_str(raw)  # Level 1
        if field is None:
            parsed.append(TurnEndingTool(tool_name=name))
        else:
            parsed.append(TurnEndingTool(
                tool_name=name,
                is_success=lambda d, _f=field: d.get(_f) is True,
            ))
    _validate_turn_ending_specs_full_list(parsed)  # Level 2
    return parsed
```

**Level 2 — full-list constraints ([R2] gap).** Cross-entry problems cannot be caught by a per-value `type=` handler. They are enforced by `_validate_turn_ending_specs_full_list(list[TurnEndingTool])` called at the end of `_parse_turn_ending_tool_specs()`, so every entry point that parses a full list runs it automatically.

Two constraints, both raising `DuplicateTurnEndingToolError` (the existing exception from the parent design — reused, not re-invented; message prefixed `--turn-ending-tool: ` so CLI users see it clearly):

1. **Duplicate CLI names.** Two entries whose `tool_name` are equal collide, regardless of whether they carry different fields. `foo` and `foo:done` are both `tool_name="foo"` after Level 1 parsing, so they collide. Enforced by dict-buildup keyed on `spec.tool_name`; a repeat raises immediately with a message naming the tool and both raw declarations for user clarity.

2. **CLI redeclares the built-in.** Any spec whose `tool_name == deep_worker_cli.middleware.send_message_completion.TOOL_NAME` (== `"send_message_to_agent"`) is rejected with an actionable message: users cannot override or shadow the built-in from CLI. This currently fires later inside `SendMessageCompletionMiddleware.__init__`, which is called from `create_cli_agent`, which runs **inside** the swallowing init `try` at `agent_process.py:1169-1559` — the exact defect [R2] identified.

```python
def _validate_turn_ending_specs_full_list(specs: list[TurnEndingTool]) -> None:
    """Reject cross-entry collisions BEFORE any resource allocation.

    Runs at every list-parsing entry point (see §3.5). Complements the
    per-string _validate_turn_ending_spec_str: that one can't see
    across entries; this one can.
    """
    from deep_worker_cli.middleware.send_message_completion import TOOL_NAME as _BUILT_IN

    seen: dict[str, str] = {}  # tool_name -> "first raw declaration"
    for spec in specs:
        # Level 2 constraint 2: no CLI declaration may target the built-in.
        if spec.tool_name == _BUILT_IN:
            raise DuplicateTurnEndingToolError(
                f"--turn-ending-tool: {_BUILT_IN!r} is a built-in "
                f"turn-ending tool and cannot be redeclared from the "
                f"CLI (it is always registered)"
            )
        # Level 2 constraint 1: no two CLI entries may share tool_name.
        if spec.tool_name in seen:
            raise DuplicateTurnEndingToolError(
                f"--turn-ending-tool: {spec.tool_name!r} is declared "
                f"more than once (first via {seen[spec.tool_name]!r}, "
                f"then via a repeat); a bare form and a :field form "
                f"of the same name are still duplicates — pick one"
            )
        seen[spec.tool_name] = spec.tool_name  # first-decl marker
```

The exception class is `DuplicateTurnEndingToolError` — same one `SendMessageCompletionMiddleware` uses today, so downstream code that catches it (currently none, but the type stays stable) is unaffected. Messages are CLI-facing when raised from these paths, but the class does not need a subclass: the raiser location and message body carry the context.

Note: `SendMessageCompletionMiddleware.__init__` still enforces the same rules as its LAST line of defense (belt-and-braces — direct programmatic construction of the middleware without going through `AgentProcess`). Nothing there changes; only its role changes from primary enforcer to backstop. Confirmed via inspection: `tool_completion.py:164-179` builds `_by_name` dict and raises `DuplicateTurnEndingToolError` on duplicate, and `SendMessageCompletionMiddleware([built_in, *additional])` uses that same code path — so a caller that reaches the middleware with the built-in name still fails, just at a later layer. In production this second failure is unreachable because Level 2 validation always runs first.

---

**Consequences pinned by tests:**

- `parse_args(["--turn-ending-tool", "foo:bar"]).turn_ending_tools == ["foo:bar"]` (list of strings, not spec objects).
- `json.dumps(build_launch_config(namespace))` succeeds — the launch config is JSON-safe end-to-end.
- Grammar errors at any entry point raise the SAME `TurnEndingToolFlagError` with the SAME message body.
- Cross-entry collisions raise the SAME `DuplicateTurnEndingToolError` regardless of entry point, at the SAME boundary (before AgentProcess construction, or inside `AgentProcess.__init__` for programmatic callers — never in `.start()` or in `create_cli_agent`).

Parse errors and collisions surface exactly where the input enters the system: at the CLI boundary during `parse_args` (grammar) and immediately after (cross-entry via full-list validation in `run_agent()`), or at the `AgentProcess.__init__` boundary (see §3.5) — never deep inside init. That's fail-fast [R1] gap 2 AND [R2] collision path.

### 3.3 Restart round-trip

Modeled 1-for-1 on `--mcp-config`. Three touch points in `agent_command.py`:

1. **`build_launch_config()` (`agent_command.py:701-734`)** — new key at the same nesting level as the existing MCP keys:
   ```python
   "turn_ending_tools": list(getattr(args, "turn_ending_tools", None) or []),
   ```
   Values are stored as the raw CLI strings (`"code_graph__submit_fragments:document_closed"`), not as parsed `TurnEndingTool` objects. This is on purpose: the format is stable, human-readable in `session_metadata.json`, and re-parsed on each start — mirroring what `--mcp-server` already does with its inline JSON strings.

2. **`merge_restart_config()` (`agent_command.py:329+`, after the existing MCP block):**
   ```python
   effective.turn_ending_tools = list(saved.get("turn_ending_tools") or [])
   ...
   turn_ending_tools = list(getattr(args, "turn_ending_tools", None) or [])
   if _is_override("turn_ending_tools", turn_ending_tools, []):
       effective.turn_ending_tools = turn_ending_tools
   ```
   The override rule matches `--mcp-config`: a non-empty CLI value replaces the saved list wholesale (there is no set-add semantics). Passing `--turn-ending-tool` on `--restart` REPLACES; passing nothing INHERITS from `session_metadata.json`.

3. **Consumer (`agent_command.py:1795+`, the `AgentProcess(...)` call):**
   ```python
   turn_ending_tool_specs=list(getattr(args, "turn_ending_tools", None) or []),
   ```
   List of RAW STRINGS, passed straight through to `AgentProcess.__init__`.

Why raw strings all the way to `AgentProcess`, not parsed objects at the argparse boundary:
- `argparse` `type=` handlers cannot see the full inventory yet; they can catch malformed strings but nothing more.
- Persistence is trivially JSON-safe when the values are strings.
- Restart replays the strings verbatim through the same parse path used on first launch, so first-launch and restart cannot diverge by construction.
- Parsing in one place (`AgentProcess`, right before `create_cli_agent`) means the inventory check (§3.4) sees exactly what the middleware will see.

Executable round-trip test (see §5): saved `session_metadata.json` with `turn_ending_tools: [...]` → `_load_launch_config()` → `merge_restart_config()` → `build_launch_config()` → asserts the reconstructed launch_config carries the same list.

### 3.4 Diagnostics

Two log events, one per registered spec. Emitted by `AgentProcess`, at the callsite immediately after `_extract_final_tool_metadata()` returns — that is the earliest point where the graph's final tool set is known (see `agent_process.py:1533`, existing precedent for the same "final inventory" concern).

Success event (INFO):
```
turn_ending_tool_registered name=<mangled>
```

Miss event (WARNING):
```
turn_ending_tool_not_found name=<mangled> candidates=[<n1>, <n2>, ...]
```

Rules:
- **Both events are logged unconditionally**, even in the healthy case. Rationale from the consumer, verbatim: "the absence of a WARNING is indistinguishable from 'the flag never reached this code at all'". So a launcher grepping the log can prove positively that its flag reached AgentProcess.
- **Machine-greppable format:** `key=value` pairs, no rich text, no quoting inside `name=`. `candidates=[...]` uses Python list `repr` for names, which is stable enough for grep-based tooling. Test pins the exact format.
- **WARNING, not hard fail.** Per the consumer — its MCP endpoint is brought up by the ingest process and the agent may start before the listener is ready; hard fail turns "optimization didn't apply this run" into "run didn't start", which is strictly worse. The specs still get registered with `ToolCompletionMiddleware` regardless — the miss is diagnostic, not blocking. If the tool actually shows up later (dynamic MCP), the middleware will match by name normally.
- **`candidates` list contents ([R1] gap 3: deterministic ranking):** the top-K (K=8) names from the final inventory, ranked by `(-similarity_score, name)` — so higher similarity comes first, and ties break by ascending name. Similarity is `difflib.SequenceMatcher(None, missing, candidate).ratio()`; empirically it puts real matches of a trailing-`s`-dropped typo above unrelated names. Ranking is a total order, so the same inventory always produces the same `candidates` list — the test can pin exact contents, not "contains". If the inventory is empty, `candidates=[]`. Test [R1] `test_candidates_deterministic_tie_break` covers same-score cases explicitly.
- **Inventory names come from `_extract_final_tool_metadata()`'s return value.** No independent inventory source, no fake helper — the test [R1] `test_diagnostics_uses_metadata_from_extract_helper_return` runs the REAL extraction helper over a fake compiled graph and asserts the diagnostic classifies against exactly those names.
- **Entry shape is normalized, because that return value is NOT uniformly `ToolMetadata`.** `_extract_final_tool_metadata()` returns plain `dict`s (`{"name", "description", "input_schema"}`) for every provider except `claude-code-sdk`, and `ToolMetadata`-shaped objects only from the claude-code-sdk `bind_tools` probe. Reading `.name` alone would therefore yield an empty name — a FALSE MISS for a tool that IS present — on every non-claude provider, including `deepseek`, the one the first consumer runs. The diagnostic MUST read each entry through `_tool_metadata_name()`, which takes the `"name"` key when the entry is a `dict` and falls back to the `.name` attribute otherwise. Test `test_metadata_may_be_dicts` pins the dict branch.
- **Emitted after inventory finalization and before provider prewarm.** No early emit before `_extract_final_tool_metadata()` — that would give false positives when external MCP hasn't connected yet — and no late emit after `_prewarm_client_manager()`, which would describe a state the prewarm already consumed. The single emission point sits between them. Because no test in the suite runs `start()` end-to-end (it daemonises, opens the registry, an RPyC server and a live model — the same constraint `test_provider_capability_guards_in_start.py` documents), this ordering is pinned structurally: `test_start_orders_extract_then_diagnostics_then_prewarm` parses `AgentProcess.start()`'s AST, asserts exactly one extraction assignment / one diagnostic call / one prewarm call, and compares their real `lineno` values.

### 3.5 Wiring into `AgentProcess` — fail-fast in `__init__` (fixes [R1] gap 2)

**Where validation happens.** Every path that produces a spec list validates it at the earliest boundary WHERE THE SPEC ENTERS ITS CONSTRUCTOR — never in `.start()`, which sits inside the swallowing init `try` (`agent_process.py:1169-1559`) that flips to `AgentStatus.ERROR` and continues registration.

Three entry points, all fail-fast BEFORE any resource allocation. **Grammar (Level 1) and cross-entry constraints (Level 2)** are enforced at each entry point via `_parse_turn_ending_tool_specs()`, which runs BOTH.

1. **CLI first-launch (`argparse type=_canonical_turn_ending_spec_str`).** Grammar-rejects malformed VALUES before argparse returns. Cross-entry collisions cannot be caught here — argparse `type=` sees one value at a time. Those are caught immediately after in step 2, on the same code path.

2. **CLI first-launch AND --restart, full-list check in `run_agent()`.** After `parse_args` (first-launch) or after `merge_restart_config` (--restart), and BEFORE `AgentProcess(...)`:
   ```python
   # In run_agent(), applied to the final effective list from either path:
   raw_specs = list(getattr(args, "turn_ending_tools", None) or [])
   parsed_specs = _parse_turn_ending_tool_specs(raw_specs)
   #   ^ raises TurnEndingToolFlagError on grammar,
   #     raises DuplicateTurnEndingToolError on cross-entry collision
   #     (dup CLI names, OR CLI name == built-in send_message_to_agent)
   # AgentProcess construction happens next, receiving raw_specs;
   # the parsed_specs list is discarded here — ctor re-parses. See §3.5 note.
   ```
   Fires for CLI-passed duplicates on first launch (`--turn-ending-tool foo --turn-ending-tool foo`), for a CLI declaration of the built-in (`--turn-ending-tool send_message_to_agent`), and for hand-edited `session_metadata.json` on --restart.

   The raise lands in `run_agent()`'s PRE-EXISTING top-level `except Exception` handler (unchanged by this work), which prints `Error: <message>` to stderr and returns exit code `1`. It does NOT propagate to the caller. That is sufficient and is the contract this design pins, because the property that matters is WHERE the failure happens, not how it is reported: it happens before `AgentProcess(...)` is constructed, so no agent is registered, no port bound, no checkpointer opened, and — critically — no `AgentStatus.ERROR` ghost process is left behind, which is exactly what the old late-detection path produced. Tests assert all three facts together: exit code `1`, the offending spec present in stderr, and an `AgentProcess.__init__` spy recording ZERO calls.

3. **Programmatic `AgentProcess(turn_ending_tool_specs=[...])`.** Validation is done SYNCHRONOUSLY in `__init__`, running the SAME `_parse_turn_ending_tool_specs()` (which runs both Level 1 and Level 2). Not later:
   ```python
   def __init__(
       self,
       ...,
       mcp_config_paths: list[Path] | None = None,
       mcp_server_inline: list[str] | None = None,
       turn_ending_tool_specs: list[str] | None = None,
       ...
   ):
       ...
       # Validate at boundary — cannot be swallowed by the init try.
       # We store the PARSED specs so .start() never re-encounters bad input.
       specs_raw = list(turn_ending_tool_specs or [])
       self._turn_ending_tools: list[TurnEndingTool] = (
           _parse_turn_ending_tool_specs(specs_raw)  # raises on invalid
       )
       # Retain raw for introspection/logging only.
       self._turn_ending_tool_specs_raw = specs_raw
   ```
   `_parse_turn_ending_tool_specs` runs the same `_validate_turn_ending_spec_str` grammar — one source of truth. Any test constructing `AgentProcess(turn_ending_tool_specs=["bad:spec:trap"])` gets an exception BEFORE checkpointer/MCP-pool/RPyC/registry allocation.

At the `create_cli_agent(...)` callsite (`agent_process.py:1470`), the already-parsed list is forwarded — no re-parsing:
```python
self.agent, self.backend = create_cli_agent(
    ...,
    turn_ending_tools=self._turn_ending_tools,
)
```

Immediately after `_extract_final_tool_metadata()` (§3.4), a second helper walks `self._turn_ending_tools` and emits the two log events, resolving each inventory entry's name through `_tool_metadata_name()` — the dict-or-object normalization §3.4 makes normative. It MUST NOT read `.name` directly: the inventory entries are plain `dict`s on every non-claude provider, so a direct attribute read would report a false miss for a tool that is present.

**Why this is a real fail-fast (not repeat of previous defect):**
- CLI first-launch: argparse handler raises grammar errors → no AgentProcess object exists.
- CLI/--restart full-list check in `run_agent()`: cross-entry collisions raise BEFORE `AgentProcess(...)` — at the same CLI layer as `merge_restart_config` itself, OUTSIDE the init try.
- Programmatic ctor: raises in `__init__` before any I/O; the caller sees a clean exception, no ERROR-mode ghost process. Test [R2] `test_ctor_duplicate_specs_raises_before_resource_allocation` and `test_ctor_builtin_name_raises_before_resource_allocation` prove this by asserting the exception AND that no checkpointer/registry side effects occurred (checked via monkeypatch spies on the ctor-time allocation points).

**Why parse twice (run_agent + ctor)?** In the CLI path, `run_agent()` runs `_parse_turn_ending_tool_specs()` for validation only (result discarded), then hands raw strings to `AgentProcess.__init__()`, which runs it again and stores the parsed list. The duplication is intentional — two independent enforcement points guarantee the constraint holds in both entry paths. If a future refactor of `run_agent()` skips the validation, the ctor still catches it; if a future refactor of `AgentProcess` moves parsing to `.start()`, `run_agent()` still catches it for CLI callers. `_parse_turn_ending_tool_specs()` is pure and cheap; there is no measurable cost.

### 3.6 Backward compatibility

- Existing agents with no `--turn-ending-tool` and no saved `turn_ending_tools` key: entirely unchanged. The kwarg defaults to `None`/`[]` everywhere.
- Existing `session_metadata.json` files without the key: `saved.get("turn_ending_tools") or []` → `[]`, no error, no migration required.
- `send_message_to_agent` remains a hard-coded turn-ending tool via `SendMessageCompletionMiddleware`'s built-in spec; CLI additions are layered ON TOP, never replace it. Users cannot silently override the built-in — `DuplicateTurnEndingToolError` still fires (from the earlier commit).
- The `create_cli_agent(turn_ending_tools=...)` Python API is unchanged. This work only adds a second delivery channel.

### 3.7 Public surface

New CLI flag `--turn-ending-tool`. That is the only public addition. No new middleware exports (`middleware/__all__` is unchanged). `_parse_turn_ending_tool_specs` is module-private (leading underscore); the middleware package continues to export exactly the three names §3.6 of the parent design authorizes.

README section documents the flag with the `rg_extractor` example and one sentence about mangled names.

## 4. Testing (TDD)

New unit tests in `tests/unit_tests/`. Legacy tests must stay green with zero edits (the 2911/0/5 baseline plus these).

**`tests/unit_tests/middleware/test_turn_ending_tool_spec_parsing.py`:**
- `test_bare_name` — `"foo"` → `TurnEndingTool("foo")`, default predicate is truthy on `{"success": 1}` and falsy on `{"success": 0}`.
- `test_name_and_field` — `"foo:bar"` → predicate is True on `{"bar": True}`, False on `{"bar": False}`, `{"bar": 1}`, `{"bar": "true"}`, `{}`, `{"bar": None}`.
- `test_predicate_uses_get_not_bracket` — a dict without the key doesn't raise KeyError.
- `test_late_binding_across_multiple_specs` — parse `["a:x", "b:y"]`; assert that spec-a's predicate reads `x` and spec-b's reads `y` on `{"x": True, "y": True}` and cross-matrices, guarding the closure late-binding trap.
- `test_empty_name_rejected` — `":x"` raises.
- `test_empty_field_rejected` — `"foo:"` raises.
- `test_dotted_field_rejected` — `"foo:a.b"` raises with a message pointing at nested fields being out of scope.
- `test_extra_colon_rejected` — `"foo:bar:baz"` raises.
- `test_whitespace_rejected` — `" foo "` and `"foo: bar"` raise.
- `test_multiple_specs_returned_in_order` — parsing `["a", "b:x"]` yields specs in given order.

**`tests/unit_tests/cli/test_agent_command_turn_ending_tool_flag.py`:**
- `test_argparse_namespace_carries_raw_strings` **[R1 gap 1]** — `parser.parse_args(["a", "--turn-ending-tool", "foo:bar"]).turn_ending_tools == ["foo:bar"]`. Type asserts each element `isinstance(x, str)`. Guards the "raw strings on namespace" contract.
- `test_launch_config_is_json_serializable` **[R1 gap 1]** — `json.dumps(build_launch_config(ns_with_specs))` succeeds and round-trips: `json.loads(json.dumps(built))["turn_ending_tools"] == ns.turn_ending_tools`.
- `test_flag_appears_in_build_launch_config` — args namespace with `turn_ending_tools=["foo:bar"]` → `build_launch_config` returns dict containing that exact list under `"turn_ending_tools"`.
- `test_omitted_flag_produces_empty_list_in_launch_config` — namespace without the attr → `[]` in output.
- `test_merge_restart_inherits_saved_when_no_cli_override` — saved `{"turn_ending_tools": ["foo:bar"]}` + CLI without the flag → effective namespace carries `["foo:bar"]`.
- `test_merge_restart_cli_override_replaces_saved` — saved `["foo:bar"]` + CLI `["baz"]` → effective is `["baz"]` (not appended).
- `test_merge_restart_empty_cli_does_not_override` — saved `["foo:bar"]` + CLI `[]` → saved wins (matches `--mcp-config` behavior).
- **`test_restart_round_trip_preserves_specs`** — executable round-trip the reviewer required for the deferred channel:
  ```python
  original = argparse.Namespace(turn_ending_tools=["code_graph__submit_fragments:document_closed"], ...)
  saved = build_launch_config(original)
  effective = merge_restart_config(saved, argparse.Namespace(...))  # no CLI override
  rebuilt = build_launch_config(effective)
  assert rebuilt["turn_ending_tools"] == original.turn_ending_tools
  ```
  Executes real code; no `inspect.getsource`.
- **`test_restart_with_invalid_saved_spec_errors_before_agentprocess`** **[R1 gap 2]** — session_metadata.json contains `{"turn_ending_tools": ["bad:spec:trap"]}` (hand-edited). Call `run_agent(args_with_restart)`. Assert `TurnEndingToolFlagError` is raised; a monkeypatched `AgentProcess.__init__` spy records ZERO calls. Proves the error surfaces at the CLI/restart boundary, not deep in init.
- **`test_first_launch_duplicate_cli_names_errors_before_agentprocess`** **[R2]** — argparse succeeds on `--turn-ending-tool foo --turn-ending-tool foo:done` (per-value grammar OK). Call `run_agent(args)`. Assert `DuplicateTurnEndingToolError` is raised with message naming `"foo"`; AgentProcess spy shows 0 calls. Covers the bare-vs-`:field` duplicate case explicitly.
- **`test_first_launch_builtin_name_errors_before_agentprocess`** **[R2]** — `--turn-ending-tool send_message_to_agent`. `run_agent(args)` raises `DuplicateTurnEndingToolError` with message naming `"send_message_to_agent"` and indicating it is a built-in; AgentProcess spy shows 0 calls.
- **`test_restart_with_duplicate_saved_specs_errors_before_agentprocess`** **[R2]** — session_metadata.json has `{"turn_ending_tools": ["foo", "foo:done"]}` (hand-edited or persisted from a hypothetical past bug). `run_agent(--restart)` raises `DuplicateTurnEndingToolError`; AgentProcess spy shows 0 calls.
- **`test_restart_with_builtin_saved_spec_errors_before_agentprocess`** **[R2]** — session_metadata.json has `{"turn_ending_tools": ["send_message_to_agent"]}`. Same as above but with the built-in-collision message.
- **`test_multiple_distinct_specs_pass_full_list_validation`** **[R2 positive case]** — `--turn-ending-tool foo --turn-ending-tool bar:x` — the repeated flag with DIFFERENT names must succeed all the way through `run_agent()` and reach `AgentProcess.__init__` with the two-element list. Distinguishes "duplicates rejected" from "repeatable flag broken".

**`tests/unit_tests/process/test_agent_process_turn_ending_specs_wiring.py`:**
- `test_ctor_accepts_turn_ending_tool_specs` — `AgentProcess(name=..., turn_ending_tool_specs=["foo"], ...)` stores parsed `TurnEndingTool` list; default `None` → `[]`.
- `test_ctor_stores_raw_alongside_parsed` — `_turn_ending_tool_specs_raw` matches the input list of strings; `_turn_ending_tools` is the parsed list.
- **`test_ctor_invalid_spec_raises_before_resource_allocation`** **[R1 gap 2]** — construct `AgentProcess(turn_ending_tool_specs=["bad:spec:trap"], ...)`. Assert `TurnEndingToolFlagError` raised. Monkeypatch spies on `_load_and_consume_restart_notice`, `_start_external_mcp_pool`, checkpointer factory, and any registry-writing call — all show ZERO invocations. Proves fail-fast in `__init__`, not `.start()`.
- **`test_ctor_duplicate_specs_raises_before_resource_allocation`** **[R2]** — `AgentProcess(turn_ending_tool_specs=["foo", "foo:x"], ...)`. Assert `DuplicateTurnEndingToolError` raised naming `"foo"`; same allocation spies show 0 calls. Also runs the bare-and-`:field`-collide-under-same-name case.
- **`test_ctor_builtin_name_raises_before_resource_allocation`** **[R2]** — `AgentProcess(turn_ending_tool_specs=["send_message_to_agent"], ...)`. `DuplicateTurnEndingToolError` naming `"send_message_to_agent"` and mentioning built-in; allocation spies 0.
- **`test_ctor_builtin_name_with_field_still_rejected`** **[R2]** — `AgentProcess(turn_ending_tool_specs=["send_message_to_agent:success"], ...)`. Same rejection: the field does not "un-collide" from the built-in.
- `test_ctor_multiple_invalid_specs_reports_first` — first invalid spec's grammar error is raised; subsequent parsing not attempted.
- **`test_ctor_multiple_distinct_specs_survive`** **[R2 positive case]** — `AgentProcess(turn_ending_tool_specs=["foo", "bar:done"], ...)` succeeds; `self._turn_ending_tools` is a two-element list with correct names.

**`tests/unit_tests/process/test_agent_process_turn_ending_diagnostics.py`:**
- **`test_diagnostics_uses_metadata_from_extract_helper_return`** **[R1 gap 3]** — the behavioral seam test. No fake helper and no hand-built inventory: run the REAL `AgentProcess._extract_final_tool_metadata` over a fake compiled graph (a `tools` node whose `tools_by_name` holds real `StructuredTool`s), then feed its return value to the diagnostic. Because the helper is real, the entries arrive in their production shape — `dict`s on this provider — which is exactly what makes the `_tool_metadata_name()` normalization of §3.4 observable rather than assumed. Assert:
  1. the extraction returned a non-empty inventory (the seam was hit);
  2. every `turn_ending_tool_registered` line names a tool present in that inventory;
  3. every `turn_ending_tool_not_found` line names a tool NOT in it;
  4. `candidates=[...]` on each miss line is a subset of it.
  Inventory names are read the same way the product does, via the `"name"` key. This proves diagnostics classify against the SAME metadata the rest of startup uses — not a test-local inventory.
- `test_registered_and_not_found_events_format` — with the seam captured as above and specs `["code_graph__submit_fragments:document_closed", "typo__submit"]`, `caplog` shows:
  - one INFO line matching `^turn_ending_tool_registered name=code_graph__submit_fragments$`,
  - one WARNING line matching `^turn_ending_tool_not_found name=typo__submit candidates=\[.*\]$` with `code_graph__submit_fragments` first in the candidates list.
- `test_no_events_when_no_specs` — empty specs list, no `turn_ending_tool_*` lines at all.
- **`test_start_orders_extract_then_diagnostics_then_prewarm`** **[R1 gap 3, R3]** — the PRODUCTION ordering guard. Parses `AgentProcess.start()`'s AST; asserts exactly one `_extract_final_tool_metadata` assignment, one `_log_turn_ending_tool_diagnostics` call and one `_prewarm_client_manager` call, then compares their real `lineno` values (`extract < diagnostics < prewarm`). Fails if a refactor moves the emit before the inventory or after the prewarm. Structural because no test in the suite runs `start()` end-to-end (see `test_provider_capability_guards_in_start.py` for the same, pre-existing constraint).
- `test_diagnostics_run_between_inventory_and_prewarm` — the behavioural companion: replays the three REAL methods through monkeypatched seams and asserts the observed call order plus that the diagnostic receives the extracted inventory. On its own it could not detect a production reordering (the test chooses the order), which is why the AST guard above is the normative one.
- `test_start_calls_diagnostics_with_the_extracted_inventory` — argument provenance: the diagnostic's argument must be the `Name` bound to `_extract_final_tool_metadata`'s return value, not an inline expression or a second independent lookup.
- `test_metadata_may_be_dicts` — the inventory entries are `dict`s on non-claude providers; the diagnostic must resolve names through `_tool_metadata_name()` and report `registered`, not a false `not_found`.
- `test_candidates_ranked_by_suffix_similarity` — inventory `["code_graph__submit_fragments", "other__unrelated"]`, spec `["code_graph__submit_fragment"]` (missing trailing s) → `candidates[0] == "code_graph__submit_fragments"`.
- **`test_candidates_deterministic_tie_break`** **[R1 gap 3]** — inventory `["abc", "acb", "bca"]`, spec `["axx"]` — construct so at least two candidates share the same similarity score. Assert `candidates` uses `(-score, name)` tie-break: same-score candidates appear in ascending alphabetical order. Same run repeated 5 times must produce identical output.
- `test_empty_inventory_yields_empty_candidates` — inventory `[]`, spec `["foo"]` → `candidates=[]` in the WARNING line.

**Integration guard (`tests/unit_tests/cli/test_agent_command_flag_visibility.py`):**
- `test_flag_visible_in_argparse_help` — `--help` output contains `--turn-ending-tool`; guards accidental removal.
- `test_flag_registered_in_detector` — `_explicit_cli_flags(["a", "--turn-ending-tool", "x"])` contains `"turn_ending_tools"`; guards forgetting the detector.

Total new tests: **73 as implemented** (rev 3 estimated ~37; the grammar and collision matrices expanded during TDD, and reviewer round 3 added the production-ordering guard). Full suite: 2984 passed / 0 failed / 5 skipped, against a 2911/0/5 baseline.

## 5. Risk / rollback

- **Risk 1: parser divergence.** THREE callers (argparse `type=`, --restart re-parse, `AgentProcess.__init__`) must all use the same grammar. Mitigation: single validator `_validate_turn_ending_spec_str(raw) -> (name, field|None)`; the argparse `type=` handler and `_parse_turn_ending_tool_specs()` both delegate to it. Grammar is thus tested in one place; layer-specific tests only prove wiring. Test `test_all_three_boundaries_use_same_grammar` compares error messages across boundaries for the same invalid input.
- **Risk 2: late binding across specs.** Python closures over the loop variable trap. Mitigation: predicate builder uses explicit default arg (`lambda d, _f=field: d.get(_f) is True`); dedicated test.
- **Risk 3: WARNING flood on healthy start.** Only one `registered` line per spec, one `not_found` line per miss. No polling, no retries. `rg_extractor` today would emit exactly one INFO line per start (single spec, successfully matched).
- **Risk 4: restart replacing user's saved specs by accident.** `--turn-ending-tool` on `--restart` REPLACES saved list, matching `--mcp-config` semantics. Documented in the README and in `merge_restart_config` inline comment; tested.
- **Risk 5: `session_metadata.json` hand-edit with invalid string.** The re-parse on next start raises loudly with the offending string in the message — this is a first-class error path, not swallowed.
- **Rollback:** revert the single feature commit. `turn_ending_tools=` in `session_metadata.json` on rolled-back nodes is ignored (`saved.get(..., [])` returns `[]` — no error).

## 6. Owner questions (none open)

Owner has resolved:
- Flag form (§3.2): two forms only, no `=value`.
- Echo for CLI: always `None` (§3.1).
- Diagnostics: WARNING with symmetric INFO on success (§3.4).
- Order of work: CLI flag first, then `rg_extractor` dispatcher rework, then joint live integration.

## 7. Implementation order (TDD)

1. RED: grammar tests (`test_turn_ending_tool_spec_parsing.py`) — fails, `_validate_turn_ending_spec_str` / `_parse_turn_ending_tool_specs` absent.
2. GREEN: `_validate_turn_ending_spec_str()`, `_canonical_turn_ending_spec_str()`, `_parse_turn_ending_tool_specs()`, `TurnEndingToolFlagError` in `deep_worker_cli/middleware/tool_completion.py`. Middleware package `__all__` unchanged.
3. RED: agent_command tests (`test_agent_command_turn_ending_tool_flag.py`, `test_agent_command_flag_visibility.py`) — fail, flag not defined.
4. GREEN: add flag to argparse main parser (with `type=_canonical_turn_ending_spec_str`) + detector; add persistence in `build_launch_config` and `merge_restart_config`; add --restart re-parse call in `run_agent()` BEFORE `AgentProcess(...)`; add pass-through at the `AgentProcess(...)` call site.
5. RED: AgentProcess wiring + fail-fast tests — fail, `turn_ending_tool_specs` kwarg absent.
6. GREEN: `AgentProcess.__init__` kwarg with SYNCHRONOUS parse (raises on invalid, spies show no resource allocation on failure); attribute stores parsed list; `.start()` no longer parses.
7. RED: diagnostics tests including the metadata-seam test and ordering guard.
8. GREEN: diagnostics helper called immediately after `_extract_final_tool_metadata()` returns; consumes that return value; deterministic tie-break (score desc, name asc).
9. README section with the `rg_extractor` example (`code_graph__submit_fragments:document_closed`).
10. Full suite `tests/unit_tests` — baseline **2911 passed / 0 failed / 5 skipped preserved** (021c5f8 baseline), plus ~37 new tests. Never pass `-p` (argparse prefix collision yields 26 false failures).
11. CODE REVIEW REQUEST → CODE SIGN OFF (hard gate) → single commit to `dev`. No FF to `master` without explicit user permission.

## 8. Corrections log

### rev 8 (2026-08-13) — reviewer round 5 (evidence precision)

Both findings are about MY claims overreaching the evidence, not about the evidence itself. Recorded because the failure mode — a live log that looks like it proves more than it observes — is easy to repeat.

| # | Reviewer finding | Where fixed |
| - | ---------------- | ----------- |
| 15 | Rev 7 said the live timeline showed "the §3.4 ordering rule holds in production". It cannot: the consumer's log observes `pool ready` and `registered`, but neither `_extract_final_tool_metadata` nor `_prewarm_client_manager` emits anything, so the log cannot place the diagnostic relative to those two calls. | §8 rev 7 now separates what the timeline DOES prove (the diagnostic did not run before external-MCP readiness — the false-`not_found` failure mode) from what it does NOT (the full `extract < diagnostics < prewarm` chain, which remains pinned only by `test_start_orders_extract_then_diagnostics_then_prewarm` reading the three call sites from `start()`'s AST). Stated explicitly that the live log and the AST guard cover different halves and neither replaces the other. |
| 16 | Rev 7 said "filler turns went to zero". The table measures `completion<=3`, a PROXY: turn contents were never classified, so short ≠ filler. The runs also differ in total turns (29 / 28 / 13), so it is not a like-for-like comparison. | §8 rev 7 now states the measurement literally — "turns with `completion<=3` fell from 9–10 to 0" — and adds both limits explicitly: the proxy caveat, and the differing turn counts. The conclusion is downgraded to "consistent with the filler pattern having been eliminated", with the wording "consistent-with, not demonstrates". |

### rev 7 (2026-08-13) — live acceptance by the consumer

Shipped as `aa70703` (`dev`, fast-forwarded to `master`) and exercised end-to-end against a live `the code-graph server` MCP endpoint. **Accepted with zero behavioural deviations.** Verbatim evidence from the consumer:

```
16:45:27 INFO  external MCP server 'the code-graph server': 2 tools loaded
16:45:27 INFO  external MCP pool ready for rg_extractor: 2 adapter tool(s)
16:45:30 INFO  turn_ending_tool_registered name=code_graph__submit_fragments

16:49:37 WARNING turn_ending_tool_not_found name=code_graph__submit_fragments candidates=['restart_agent', 'read_image', 'send_message_to_agent', 'edit_file', 'read_file', 'write_file', 'write_todos', 'restart_self']
```

What this timeline does and does not prove. The `registered` marker at 16:45:30 lands AFTER "external MCP pool ready" at 16:45:27, which establishes ONE thing: the diagnostic did not run before external-MCP readiness, so the healthy case produced no false `not_found` while the pool was still connecting. That is the failure mode §3.4 warns about, and it is the reason both events are emitted at a single late point.

It does NOT establish the full §3.4 ordering `extract < diagnostics < prewarm`: neither `_extract_final_tool_metadata` nor `_prewarm_client_manager` logs anything the consumer could observe, so the timeline cannot place the diagnostic relative to them. The full ordering remains pinned by `test_start_orders_extract_then_diagnostics_then_prewarm`, which reads the three real call sites out of `AgentProcess.start()`'s AST and compares their `lineno` values. Live log and AST guard cover different halves of the claim; neither replaces the other.

The miss case left the agent at `status=idle`, `error_message=None`, as §3.4 requires.

Measured effect (the reason the flag exists), completion tokens per turn from `token_stats.jsonl`:

| Run | turns | turns with `completion<=3` |
| --- | --- | --- |
| before (`tok_default43b`) | 29 | 10 |
| before (`tok_default43`) | 28 | 9 |
| after (`tok_tet_live`) | 13 | **0** |

Stated literally: **turns with `completion<=3` fell from 9–10 to 0.** That is the measurement, and it is what the numbers support.

Two limits on reading more into it. First, `completion<=3` is a PROXY for the filler pattern, not proof that each such turn was filler — individual turn contents were not classified, so "0 short turns" is not the same claim as "0 filler turns". Second, the runs differ in total turn count (29 / 28 / 13), so the comparison is not like-for-like at the run level; the absolute count of short turns is the meaningful figure, not a ratio. The result is CONSISTENT with the filler pattern having been eliminated — which is what the flag was built to do — but the evidence is consistent-with, not demonstrates. Ingest result: `files=101 symbols=597 semantics=441 rejected=1 incomplete=1 silent=1`, 9 documents closed via `document_closed`.

Documentation defects found by that acceptance and fixed in this revision:

| # | Finding | Where fixed |
| - | ------- | ----------- |
| 13 | The example command used `--provider deepworker_a2a --model gpt-5-mini`, which argparse rejects (`invalid choice`). The documented command could not be executed verbatim. The value had been copied from an early consumer message and never run. | §3.1 example corrected to `--provider deepseek --model deepseek-chat` with a note recording the defect; the same command fixed in `README.md`; §3.4's "the one the first consumer runs" corrected from `deepworker_a2a` to `deepseek`. |
| 14 | "Nested field paths are rejected" read as though a dot were forbidden anywhere in a spec. In fact a dot is legal in the tool NAME (`a.b:c` parses) and rejected only in the FIELD (`x:a.b`). | `README.md` now states the asymmetry explicitly, together with the no-whitespace rule and its rationale. |

Not a defect, recorded for provenance: the consumer's launcher builds its own `deep-worker` command, so a start without the flag persists `turn_ending_tools: []` and overwrites the saved list. That is the §3.3 replace-wholesale semantics working as specified; the flag has to be added to that launcher, which is the consumer's own change.

### rev 6 (2026-08-13) — reviewer round 4 (documentation only)

| # | Reviewer finding | Where fixed |
| - | ---------------- | ----------- |
| 12 | §3.5's closing paragraph still said the diagnostics emit "using the metadata's `.name` attributes", contradicting the §3.4 rule rewritten in rev 5 and recreating the very false-miss claim that rev 5 removed. | §3.5 paragraph rewritten: entries are resolved through `_tool_metadata_name()`, with an explicit MUST NOT on reading `.name` directly and the reason (entries are plain `dict`s on every non-claude provider). Swept the whole document for the same stale phrasing and found one more: §4's description of `test_diagnostics_uses_metadata_from_extract_helper_return` still described a monkeypatched spy reading `[m.name for m in ...]`, whereas the implemented test runs the REAL extraction helper and reads the `"name"` key. That bullet is now rewritten to match, and to state WHY running the real helper matters — it is what makes the dict shape, and therefore the normalization, observable instead of assumed. Remaining `.name` mentions in the document are deliberate: §3.4's statement of the fallback branch, and the rev 4/rev 5 corrections rows quoting the superseded text. |

No product-code, test-code or behavioural change: documentation only.

### rev 5 (2026-08-13) — reviewer round 3 (code review)

| # | Reviewer finding | Where fixed |
| - | ---------------- | ----------- |
| 9 | `test_diagnostics_run_between_inventory_and_prewarm` replays the three calls in an order the TEST chooses, so it stays green if production `start()` moves diagnostics after prewarm. The AST test only proved argument provenance, not ordering. | New `test_start_orders_extract_then_diagnostics_then_prewarm`: locates the extraction assignment, the diagnostic call and the `_prewarm_client_manager` call in `AgentProcess.start()`'s AST, asserts exactly one of each, and compares their real `lineno` values. Mutation-verified: moving the diagnostic call below the prewarm block turns exactly this test RED (1 failed / 10 passed) while the behavioural test stays green — precisely the gap the reviewer identified. §3.4 and §4 updated to make this the normative ordering guard and the behavioural test its companion. |
| 10 | §3.4 normative text still described the inventory as `list[ToolMetadata]` read via `.name`, contradicted only later in the corrections log. | §3.4 rewritten: the dict-vs-object split is now stated as a normative rule with its failure mode (false miss on `deepworker_a2a`), and `_tool_metadata_name()` is named as the required accessor. |
| 11 | §3.5 normative text still said the exception "propagates out of `run_agent()`". | §3.5 step 2 rewritten: the pre-existing top-level handler prints `Error: <message>` and returns exit code `1`; the normative property is the BOUNDARY (before `AgentProcess(...)`, no registration, no ERROR-mode ghost), with the three facts the tests assert spelled out. |

No product-code change was requested and none was made; the diff is test + documentation only.

### rev 4 (2026-08-13) — implementation findings

Recorded when the deviations were first discovered. Findings 6–8 below are now folded into the normative sections themselves (see rev 5); the rows are kept for provenance.

Three points where the implementation had to deviate from what rev 3 asserted. All are narrower than the design contract, none change it.

| # | Design said | Reality | Resolution |
| - | ----------- | ------- | ---------- |
| 6 | §3.5 step 2: "the exception propagates out of `run_agent()`, is printed by the CLI top-level, and the process exits". | `run_agent()` has a pre-existing top-level `except Exception` (`agent_command.py`, unchanged by this work) that prints `Error: <msg>` to stderr and returns `1`. Nothing propagates out. | Contract unchanged where it matters: the failure still happens BEFORE `AgentProcess(...)`, so no ERROR-mode ghost agent is registered. Tests assert exit code `1`, the offending spec in stderr, AND a `AgentProcess.__init__` spy with zero calls. The design sentence was wrong about the mechanism, not about the boundary. |
| 7 | §3.4: "the diagnostic helper takes `list[ToolMetadata]` … and reads `.name` on each". | `_extract_final_tool_metadata()` returns plain `dict`s (`{"name", "description", "input_schema"}`) for every non-`claude-code-sdk` provider, and `ToolMetadata` objects only from the claude-code-sdk `bind_tools` probe. `.name` alone would silently see empty names on every other provider — i.e. report a false miss for a tool that IS present. | `_tool_metadata_name()` accepts both shapes (dict key first, attribute fallback). Test `test_metadata_may_be_dicts` and `test_diagnostics_uses_metadata_from_extract_helper_return` (which runs the REAL extraction helper over a fake compiled graph) pin it. |
| 8 | §4: `test_events_after_inventory_finalized_and_before_prewarm` pins log-record ORDER from a real `start()` run. | Reaching that region of `start()` needs a daemonised process, registry, RPyC server and a live model. No existing test in the suite runs `start()` end-to-end; `test_provider_capability_guards_in_start.py` documents the same constraint and uses structural guards instead. | Replaced by three tests — see rev 5 finding 9 for the ordering guard that actually pins the contract. |

Also worth recording: the tie-break and top-K rules were mutation-checked. Replacing `(-score, name)` with `-score` alone, and `K=8` with `K=20`, each turned exactly the intended test red — the assertions pin behaviour, not incidental formatting.

### rev 3 (2026-08-13) — reviewer round 2

| # | Reviewer finding | Where fixed |
| - | ---------------- | ----------- |
| 5 | Duplicate CLI names and CLI declaration of the built-in `send_message_to_agent` were caught only by `SendMessageCompletionMiddleware.__init__` — inside `create_cli_agent`, inside the swallowing init `try`. | §3.2 new `_validate_turn_ending_specs_full_list()` (Level 2) inside `_parse_turn_ending_tool_specs()`; §3.5 explicit full-list check in `run_agent()` before `AgentProcess(...)`, AND synchronous parse in `AgentProcess.__init__` (still calls the same function). §4 new tests: `test_first_launch_duplicate_cli_names_errors_before_agentprocess`, `test_first_launch_builtin_name_errors_before_agentprocess`, `test_restart_with_duplicate_saved_specs_errors_before_agentprocess`, `test_restart_with_builtin_saved_spec_errors_before_agentprocess`, `test_ctor_duplicate_specs_raises_before_resource_allocation`, `test_ctor_builtin_name_raises_before_resource_allocation`, `test_ctor_builtin_name_with_field_still_rejected`. Positive case: `test_multiple_distinct_specs_pass_full_list_validation`, `test_ctor_multiple_distinct_specs_survive`. Bare-vs-`:field` normalization: covered in `test_ctor_duplicate_specs_raises_before_resource_allocation` (input `["foo", "foo:x"]`) and `test_first_launch_duplicate_cli_names_errors_before_agentprocess`. |

### rev 2 (2026-08-13) — reviewer round 1

| # | Reviewer finding | Where fixed |
| - | ---------------- | ----------- |
| 1 | `type=` handler returning `TurnEndingTool` broke the raw-string persistence contract. | §3.2 two-layer parsing; `_canonical_turn_ending_spec_str` returns str, `_parse_turn_ending_tool_specs` returns objects. New tests `test_argparse_namespace_carries_raw_strings`, `test_launch_config_is_json_serializable`. |
| 2 | Parsing "right before `create_cli_agent`" was inside the swallowing init `try`. | §3.5 rewritten: validation happens synchronously in `AgentProcess.__init__` AND at the CLI/--restart boundary in `run_agent()`, both OUTSIDE the init try. New tests `test_ctor_invalid_spec_raises_before_resource_allocation`, `test_restart_with_invalid_saved_spec_errors_before_agentprocess`. |
| 3 | Diagnostics test would have observed a fake inventory helper, not the real seam. | §4 new test `test_diagnostics_uses_metadata_from_extract_helper_return` intercepts `_extract_final_tool_metadata`'s return value and asserts diagnostics classify against exactly that data. §3.4 pins deterministic tie-break `(-score, name)`. |
| 4 | Whitespace prohibition tested but not stated normatively. | §3.2 "Grammar rules" now states the no-whitespace rule as normative with motivation (silent dedup after shell quoting bug). |

