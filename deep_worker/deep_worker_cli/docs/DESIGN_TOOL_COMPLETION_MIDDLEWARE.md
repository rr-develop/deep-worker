# Design: `ToolCompletionMiddleware` — a class of turn-ending tools

**Status:** OWNER-APPROVED (2026-08-12) — pending reviewer design sign-off.

**Owner decisions folded in (§6 resolved):**
1. Channel (a) Python API is **mandatory** and is what this release ships. Channel (b) launch-config JSON is **deferred — NOT in v1**: code review found that stock `--restart` drops the key and that validation inside the initialization catch cannot fail fast (§3.7). (a) was built to be independently complete precisely so (b) could slip without blocking, which is what happened.
2. `turn_ending_tools` is **public API** — documented in the README as a Python API, not an internal extension point. Launch-config schema documentation is **out of scope for this release** and lands with channel (b) (§3.7).
3. Naming settled: parameter `turn_ending_tools`, spec class `TurnEndingTool`.

**Branch:** `dev` — no fast-forward to master until the feature is complete + signed off.
**Related:** `docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md` (existing precedent), `deep_worker_cli/middleware/send_message_completion.py`, `deep_worker_cli/agent.py:988–997`.

## 1. Problem

`send_message_to_agent` is one member of a class of tools whose successful invocation SHOULD end the current agent turn (drive the agent to `idle`), without a second model call producing a filler `AIMessage`. The framework already implements this for `send_message_to_agent` via `SendMessageCompletionMiddleware`, but:

- `TOOL_NAME = "send_message_to_agent"` is a module-level constant (hard-code).
- Registration in `agent.py:988–997` is a hard-coded, unconditional `agent_middleware.append(SendMessageCompletionMiddleware())`.
- There is no supported way to declare *another* tool as turn-ending — every additional case would require another middleware class + another edit to `agent.py`.

The owner asked for a proper feature: a single class of turn-ending tools, extensible per-agent, without name-based magic.

## 2. Non-goals (owner-stated constraints)

- **No name-based semantics for MCP or user tools.** A tool does NOT become turn-ending because it happens to be named `submit_something`. Semantics come from **explicit registration by the agent's launcher**, not from string matching. This rules out any "known suffix / naming convention" scheme (a previously rejected variant).
- **No hard-coded list of turn-ending tools.** The registration channel must be data-driven.
- **No behavior change for existing agents.** `send_message_to_agent` must keep working exactly as today (behaviorally identical, verified by the existing `test_send_message_completion*` tests).

## 3. Design

> **Revision 2 (2026-08-12)** — incorporates the five normative corrections requested by the reviewer. Changes vs revision 1 are marked **[R2]**.

### 3.1 The public declaration: `TurnEndingTool`

A frozen dataclass — the contract says immutable, so the type is immutable **[R2: was a mutable `__slots__` class]**. Validation happens at construction, so a malformed declaration fails at wiring time rather than silently never matching.

```python
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _default_is_success(data: dict[str, Any]) -> bool:
    return bool(data.get("success"))


@dataclass(frozen=True, slots=True)
class TurnEndingTool:
    """Declaration that one specific tool ends the agent's turn when it
    succeeds.

    Attributes:
        tool_name: exact tool name as it appears on ``ToolMessage.name``.
            Compared with ``==``. No prefix, suffix, or fuzzy matching:
            a tool is turn-ending only because it was registered here.
        is_success: predicate over the parsed tool-result dict. Returns
            True when this call should end the turn. Defaults to
            "the ``success`` key is truthy". If the payload cannot be
            parsed into a dict, the turn never ends regardless.
        echo_field: name of the field whose non-empty string value is
            emitted as the injected ``AIMessage.content``. ``None`` means
            "end the turn without emitting any message" (jump-only).
    """

    tool_name: str
    is_success: Callable[[dict[str, Any]], bool] = field(default=_default_is_success)
    echo_field: str | None = "delivered_content"

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("TurnEndingTool.tool_name must be a non-empty, non-whitespace string")
        if not callable(self.is_success):
            raise TypeError("TurnEndingTool.is_success must be callable")
        if self.echo_field is not None:
            if not isinstance(self.echo_field, str) or not self.echo_field.strip():
                raise ValueError(
                    "TurnEndingTool.echo_field must be None or a non-empty, non-whitespace string"
                )
```

**[R2]** `is_success` defaults via a module-level function, and the middleware tests `is_success is None` nowhere — the field is never `None`, so no truthiness check on a callable is possible. Passing `is_success=None` explicitly raises `TypeError` from `__post_init__`.

`TurnEndingTool` is **public API** and is exported from the documented package surface (§3.6).

### 3.2 `ToolCompletionMiddleware`

```python
def _trailing_tool_message_run_length(messages: Sequence[Any]) -> int:
    """Number of contiguous ToolMessages at the end of ``messages``.

    The run boundary is the first preceding message that is NOT a
    ToolMessage — in practice the AIMessage carrying the tool call(s)
    that produced this run. Returns 0 when the tail is not a ToolMessage.
    """
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        count += 1
    return count


class DuplicateTurnEndingToolError(ValueError):
    """Raised when the same tool name is declared turn-ending twice."""


class ToolCompletionMiddleware(AgentMiddleware):
    """Ends the agent turn when a registered tool reports success."""

    tools: list = []  # registers no tools of its own

    def __init__(self, specs: Iterable[TurnEndingTool] = ()) -> None:
        by_name: dict[str, TurnEndingTool] = {}
        for spec in specs:
            if not isinstance(spec, TurnEndingTool):
                raise TypeError(f"turn-ending tool declarations must be TurnEndingTool, got {type(spec).__name__}")
            if spec.tool_name in by_name:
                raise DuplicateTurnEndingToolError(
                    f"tool {spec.tool_name!r} is declared turn-ending more than once; "
                    f"remove the duplicate declaration"
                )
            by_name[spec.tool_name] = spec
        self._by_name = by_name

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime=None) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        if not isinstance(messages[-1], ToolMessage):
            return None

        # [R2] Only a SOLE trailing tool result may end the turn. The
        # trailing run is delimited by the first preceding message that is
        # not a ToolMessage — i.e. the AIMessage that issued the tool
        # call(s). A run longer than 1 means the model made parallel tool
        # calls: ending the turn there could hide a failed sibling result
        # from the model, so we fall through and let it see every result.
        if _trailing_tool_message_run_length(messages) != 1:
            return None

        last = messages[-1]
        spec = self._by_name.get(getattr(last, "name", None))
        if spec is None:
            return None
        data = _parse_tool_result(last.content)
        if data is None:
            return None
        try:
            if not spec.is_success(data):
                return None
        except Exception:
            logger.warning(
                "turn-ending predicate for %r raised; not ending the turn",
                spec.tool_name, exc_info=True,
            )
            return None

        if spec.echo_field is None:
            return {"jump_to": "end"}                     # [R2] jump-only, no synthetic message

        value = data.get(spec.echo_field, "")
        if not isinstance(value, str) or not value.strip():
            return None
        echo = AIMessage(content=value)
        ensure_message_created_at_after(messages, echo)
        return {"messages": [echo], "jump_to": "end"}
```

**[R2] No empty `AIMessage` is ever manufactured.** Verified empirically against the installed LangGraph/LangChain:

- `langchain/agents/middleware/types.py:886` and `:985` document `return {"jump_to": "end"}` with no `messages` key as the canonical form — jump-only is supported API, not a hack.
- Live check (fake tool-calling model, `before_model` returning bare `{"jump_to": "end"}`): the graph terminated with final state `HumanMessage → AIMessage(content='', tool_calls=[...]) → ToolMessage`, and the queued second model response was never consumed. Jump-only ends the turn with **zero** additional model calls and leaves valid state.
- Injecting an empty `AIMessage` would in any case be pointless: `process/agent_process.py:3414-3416` walks messages backwards and explicitly `continue`s past AIMessages whose text is empty ("Skip AIMessages with empty text (tool-call-only messages)"). An empty echo is invisible to `get_last_response` while still polluting history and token budget.

Consequence to state plainly: with `echo_field=None` the turn ends with a `ToolMessage` as the last message and `get_last_response()` reports the last *earlier* non-empty AIMessage (or `None`). That is the correct semantic for a tool whose result is not conversational output. Callers wanting a visible answer set `echo_field`.

**Multiple / parallel tool calls [R2, tightened in R3].** When the model emits parallel tool calls, LangGraph appends one `ToolMessage` per call and `before_model` runs once, after all of them. Inspecting only `messages[-1]` would be unsafe: with `[failed sibling, successful registered result]` the turn would end and the sibling's failure would never reach the model.

Normative rule: **the turn may end only when the contiguous trailing run of `ToolMessage`s has length exactly 1.** The run's boundary is the first preceding message that is *not* a `ToolMessage` — the model's tool-call `AIMessage`. This is implemented by `_trailing_tool_message_run_length()` above and enforced in `before_model` *before* any spec lookup, so it holds regardless of which position the registered tool occupies in the batch. Any parallel batch — registered result first, last, or in the middle — falls through to the model, which then sees every result. Both ordering cases are tested (§4).

`_parse_tool_result` moves to this module verbatim (JSON first, then `ast.literal_eval`) and is re-exported from `send_message_completion` for import compatibility.

### 3.3 `SendMessageCompletionMiddleware` stays a concrete, appended class

**[R2] Corrects a contradiction in revision 1**, which claimed both that `agent.py` appends a plain `ToolCompletionMiddleware` and that the stack still contains a `SendMessageCompletionMiddleware`. Only the latter is compatible with existing tests and with `_is_outgoing_inter_agent_echo` consumers. Resolution: the subclass carries the extra specs.

```python
# send_message_completion.py — after refactor
from deep_worker_cli.middleware.tool_completion import (
    ToolCompletionMiddleware,
    TurnEndingTool,
    _parse_tool_result,          # re-exported: existing importers keep working
)

TOOL_NAME = "send_message_to_agent"          # unchanged, still the single source of truth


class SendMessageCompletionMiddleware(ToolCompletionMiddleware):
    """Turn-ending middleware for send_message_to_agent, plus any
    additional turn-ending tools declared by the launcher."""

    def __init__(self, additional: Iterable[TurnEndingTool] = ()) -> None:
        built_in = TurnEndingTool(tool_name=TOOL_NAME)   # defaults: success / delivered_content
        super().__init__([built_in, *additional])
```

Preserved without change: the class name, `TOOL_NAME`, `_parse_tool_result`, the module path, the `middleware/__init__.py` exports, the stack position, and every `isinstance` check. `agent.py:988–997` keeps appending `SendMessageCompletionMiddleware(...)`, now with the launcher's specs passed in.

Known dependents re-verified as unaffected (they key off the tool name string or the class name, both preserved): `client/agent_client.py:113`, `process/agent_process.py:417/3973/4097`, `rpyc_service/agent_service.py:614`, `tools.py:971`, `middleware/inter_agent_reply.py:120`, `providers/claude_code_sdk_provider/delivery_resolve_middleware.py:44` (duplicates the string by value — left alone deliberately; deduplicating it is a separate cross-component change), `chat_model.py:358`.

### 3.4 Conflicts are configuration errors, not silent no-ops

**[R2] Replaces revision 1's "first registration wins, duplicates silently ignored".** `turn_ending_tools` is public configuration; a silently dropped entry hides typos and config layering.

- Two user specs with the same `tool_name` → `DuplicateTurnEndingToolError` (a `ValueError`) naming the tool.
- A user spec for `send_message_to_agent` → same error. The built-in wins **semantically**, but the collision is reported rather than absorbed, so a launcher cannot believe it redefined delivery semantics.
- Both errors surface at agent construction, i.e. at startup, with an actionable message.

### 3.5 Registration channel (a) — Python API (mandatory, v1)

`create_cli_agent` (`agent.py:805`) gains one keyword-only parameter:

```python
def create_cli_agent(
    model, assistant_id, *, ...,
    turn_ending_tools: Iterable[TurnEndingTool] | None = None,
) -> tuple[Pregel, CompositeBackend]:
```

It is forwarded into `SendMessageCompletionMiddleware(additional=turn_ending_tools or ())` at the unchanged append site. All five existing call sites (`main.py:486`, `non_interactive.py:631`, `agent_process.py:1470`, `task_manager.py:364`, plus tests) pass nothing and get behaviorally identical results.

```python
agent, backend = create_cli_agent(
    model=..., assistant_id="extractor-42",
    tools=[submit_fragments_tool],
    turn_ending_tools=[
        TurnEndingTool(
            tool_name="submit_fragments",
            is_success=lambda d: d.get("document_closed") is True,
            echo_field=None,        # no conversational echo — just end the turn
        ),
    ],
)
```

### 3.6 Public surface

`TurnEndingTool`, `ToolCompletionMiddleware`, and `DuplicateTurnEndingToolError` are exported from `deep_worker_cli.middleware` (added to `middleware/__init__.py.__all__`) so the documented import is `from deep_worker_cli.middleware import TurnEndingTool`.

### 3.7 Registration channel (b) — launch-config JSON — **DEFERRED, not shipped in v1**

> **[R4] Status: deferred to a follow-up change.** An implementation was written and then withdrawn during code review, which found two contract gaps that cannot be closed inside this change's blast radius:
>
> 1. **Restart does not preserve the value.** Stock restart reconstructs metadata through `merge_restart_config()` → `build_launch_config()` (`cli/agent_command.py:329,644`). `build_launch_config` returns a fixed allowlist dict that has no `turn_ending_tools` key, and `merge_restart_config` never copies one onto the namespace, so a value hand-written into `session_metadata.json` is silently dropped on `--restart`. Verified by reading both functions. Honouring the documented "re-read on every start, including `--restart`" guarantee means extending the restart round-trip — a separate, higher-risk subsystem that this change should not touch on its way past.
> 2. **Malformed config would not have failed startup as specified.** The parse sat inside the agent-initialisation `try` at `agent_process.py:1169`, whose handler (line 1559) sets `self.agent = None`, flips the agent to `AgentStatus.ERROR` and *continues* bringing the process up. That is "degrade and continue", not the fail-fast this section requires. Correcting it means moving validation to an early boundary ahead of checkpointer/MCP allocation — again outside this change's scope.
>
> The owner's ruling explicitly allowed deferring (b) so that (a) does not end up half-done, and the reviewer's test-shaped objection was decisive: the `inspect.getsource` wiring test that shipped with it asserted the *presence* of the plumbing while both defects lived in its *behaviour*. A test that cannot observe the bug it is meant to guard is worse than no test, because it advertises coverage that does not exist.
>
> **When (b) is implemented it must land with:** an executable round-trip test (saved launch_config → `merge_restart_config` → `build_launch_config` → new `AgentProcess` → `create_cli_agent` receives equivalent specs), and a test asserting that malformed config raises the indexed, actionable error *and* that `create_cli_agent` is never called / no resources are registered. Not `inspect.getsource`.
>
> The remainder of this section is the specification for that follow-up work.

For agents started by the stock CLI, which have no launcher code of their own. `AgentProcess` would read an optional array from launch-config / `session_metadata.json`:

```json
{
  "turn_ending_tools": [
    {"tool_name": "submit_fragments", "success_key": "document_closed", "echo_field": null}
  ]
}
```

- `success_key: "X"` desugars to `is_success = lambda d: bool(d.get("X"))`. Omitted → the default `success` predicate. **Arbitrary Python predicates are never accepted from JSON** (that would be RCE); launchers needing them use channel (a).
- **[R2] Strict validation, fail-fast.** The array must be a list of objects; `tool_name` is required and must be a non-empty string; `success_key` and `echo_field` must be string-or-absent / string-or-null; **unknown keys are rejected**. Any violation raises an actionable configuration error that **fails agent startup** — revision 1's "warning + ignore" is too silent for a public, behavior-changing setting: an agent that silently didn't get its turn-ending tool looks like a hang.
- **[R4] Persistence/restart (future requirement, NOT implemented).** The specs **must be** re-read from launch-config on every start, including `--restart` and TaskManager relaunch, so a restarted agent has identical turn-ending behavior. This **must be tested** with the executable round-trip described in the status note above — the first attempt failed exactly here, because `build_launch_config()` drops the key.

Channel (b) is layered on top of channel (a) and, per the owner's ruling, may be deferred to a follow-up commit if it proves problematic; channel (a) is complete and commit-ready on its own — which is exactly what happened, see the status note above.

### 3.8 Ordering in the middleware stack

Unchanged: `SendMessageCompletionMiddleware` is appended at exactly its current position — after `DanglingToolCallMiddleware`, before `InterAgentReplyMiddleware` (`test_inter_agent_reply_integration.py` asserts this order). Docs referencing the ordering (`DESIGN_MESSAGE_CREATED_TIMESTAMPS.md:73`, `DESIGN_REPLY_BACK_FOR_INTER_AGENT_MESSAGING.md:§3.5`) need no update.

## 4. Testing (TDD)

New `tests/unit_tests/middleware/test_tool_completion.py`:

- `test_empty_registry_never_ends` — no specs → no turn-end for any ToolMessage.
- `test_matches_by_exact_name_only` — register `foo`; `ToolMessage(name="bar")` → no-op.
- **[R2] `test_convention_looking_name_never_ends`** — unregistered `submit_fragments`, `mcp__x__submit`, `send_message_to_agent_v2` all produce no turn-end. Direct proof that no name-based magic exists.
- `test_is_success_false_no_end`, `test_raising_predicate_no_end_and_warns`.
- `test_echo_field_missing_no_end`, `test_echo_field_empty_string_no_end`, `test_echo_field_non_string_no_end`.
- **[R2] `test_echo_field_none_jumps_without_message`** — returns exactly `{"jump_to": "end"}`; asserts no `messages` key.
- `test_ignores_non_toolmessage_tail`, `test_parses_json_and_repr_forms`, `test_hook_config_can_jump_end`.
- **[R2] `test_duplicate_user_specs_raise`** and **`test_user_spec_colliding_with_builtin_raises`** — both `DuplicateTurnEndingToolError`, message names the tool.
- **[R2] `test_turn_ending_tool_validation`** — empty/whitespace `tool_name`, non-callable `is_success` (incl. `None`), empty/whitespace `echo_field` → raise; frozen instance rejects mutation.
- **[R2/R3] Parallel tool calls — both ordering cases required:**
  - `test_parallel_registered_result_last_does_not_end` — `[AIMessage(tool_calls=2), failed sibling ToolMessage, successful registered ToolMessage]` → `None`. This is the case the naive `messages[-1]` implementation got wrong.
  - `test_parallel_registered_result_first_does_not_end` — `[AIMessage(tool_calls=2), successful registered ToolMessage, sibling ToolMessage]` → `None`.
  - `test_parallel_all_successful_still_does_not_end` — run length 2 with two successes → `None` (the rule is run-length based, not failure-detection based).
  - `test_single_trailing_tool_message_ends` — run length 1 → ends, proving the guard is not over-broad.
  - `test_trailing_run_boundary_is_first_non_toolmessage` — direct unit test of `_trailing_tool_message_run_length` over `[Human, AI, Tool, Tool]` → 2, `[Human, AI, Tool, AI]` → 0, `[]` → 0, and a history containing an *earlier* ToolMessage run separated by an AIMessage → only the trailing run counts.

**[R2] Graph-level** `test_tool_completion_graph.py`:
- A custom registered tool ends the turn with **zero second model call** (assert the fake model's queued next response is unconsumed), for both `echo_field` set and `echo_field=None`; resulting state is valid and re-invocable.

Wiring `test_tool_completion_wiring.py`:
- `create_cli_agent(..., turn_ending_tools=[TurnEndingTool("foo")])` → stack contains a `SendMessageCompletionMiddleware` whose registry has both `send_message_to_agent` and `foo`.
- `turn_ending_tools=None` → wiring identical to today.
- Colliding user spec → startup error.

**[R4] Deferred with channel (b):** `test_launch_config_turn_ending_tools.py` is NOT part of v1. When (b) lands it must ship an executable restart round-trip test and a fail-fast test proving `create_cli_agent` is never reached — not an `inspect.getsource` presence check (§3.7).

Regression: every existing `test_send_message_completion*`, `test_inter_agent_reply_integration.py`, and `test_delivery_resolve_wiring.py` passes **with zero edits**.

## 5. Risk / rollback

- **Regression of `send_message_to_agent`.** Mitigated: concrete class, name, constant, exports, and stack position all preserved; existing tests unedited.
- **User predicate raises.** Caught → treated as "not successful" + WARNING. Never stalls the graph.
- **Jump-only leaves a ToolMessage tail.** Empirically validated above; `get_last_response` already tolerates it (skips empty AI content by design). Documented as the intended semantic of `echo_field=None`.
- **Launch-config JSON injection.** N/A in v1 — channel (b) is deferred (§3.7). When it lands: only `{tool_name, success_key, echo_field}` accepted, unknown keys rejected, no eval.
- **Stricter errors could break a live config.** No shipped config currently sets `turn_ending_tools` (the key is new), so fail-fast cannot regress an existing deployment.
- **Rollback:** revert the single commit; no data migration, no state format change.

## 6. Owner decisions (resolved — no open questions)

1. **Channels — RESOLVED.** (a) Python API — **mandatory**, shipped in v1. (b) launch-config JSON — **deferred to a follow-up change**, not in v1: the owner's ruling permitted deferral, and code review found two contract gaps that cannot be closed within this change's blast radius (§3.7).
2. **Visibility.** `turn_ending_tools` is **public API**. Documentation obligations for this change:
   - `deep_worker_cli/README.md` — a short section documenting `create_cli_agent(..., turn_ending_tools=[...])` and `TurnEndingTool(tool_name, *, is_success, echo_field)`, with the `submit_fragments` example.
   - launch-config schema documentation for the `turn_ending_tools` JSON array — **deferred with channel (b)** (§3.7); not documented as available in v1.
   - This design doc remains the normative reference.
   - `default_agent_prompt.md` is NOT touched: the mechanism is a launcher-side capability, not something the model needs to be told about.
3. **Naming.** Parameter `turn_ending_tools`; spec class `TurnEndingTool`; middleware `ToolCompletionMiddleware` (unchanged — it is the mechanism, not the declaration).

## 7. Implementation order (TDD)

1. RED: `tests/unit_tests/middleware/test_tool_completion.py` per §4 — fails (module absent).
2. GREEN: `middleware/tool_completion.py` with `TurnEndingTool` (frozen, validated), `ToolCompletionMiddleware` (duplicate-rejecting, jump-only on `echo_field=None`, conservative on parallel tool calls), `DuplicateTurnEndingToolError`.
3. Refactor `send_message_completion.py` to a subclass accepting `additional` specs; re-export `_parse_tool_result`. Full existing suite stays green with **zero test edits**.
4. Graph-level test (§4) proving no second model call, for both echo modes.
5. `create_cli_agent(..., turn_ending_tools=...)` + `test_tool_completion_wiring.py` + exports in `middleware/__init__.py` — **channel (a) complete and commit-ready here**.
6. ~~Channel (b): `AgentProcess` launch-config plumbing~~ — **[R4] deferred to a follow-up change** (§3.7): restart round-trip does not preserve the key, and validation inside the init `try` cannot fail fast.
7. Docs: `README.md` Python-API section. (Launch-config schema docs are deferred with channel (b) — §3.7.)
8. Full suite `tests/unit_tests` — baseline 2843 passed / 0 failed / 5 skipped preserved, plus the new tests. Never pass `-p` (argparse prefix collision yields 26 false failures).
