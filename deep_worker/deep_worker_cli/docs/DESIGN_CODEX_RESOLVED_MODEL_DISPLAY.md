# Design: Codex Resolved Model Display

Status: Implemented

## Problem

When a deep-worker agent is launched with `--provider codex` and no explicit `--model`, the Codex CLI chooses its own current default model. Deep-worker intentionally does not hardcode that default and does not pass `--model` to Codex.

That is correct for execution, but incorrect for display: `deep_worker_web` currently receives `None`/empty metadata and shows the model as `unknown`.

The fix must discover the model that Codex actually used in this session and surface it to the web UI without pinning a default model in deep-worker.

## Constraints

- Do not hardcode Codex's default model.
- Do not pass `--model` when the user did not specify one.
- Do not reintroduce the legacy poison value `"unknown"` into launch config, restart config, or Codex API calls.
- No polling.
- Prefer event-driven/non-blocking flow.
- Do not add threads.
- Avoid adding OpenAI Agents SDK as a runtime dependency unless the Codex CLI path cannot expose the needed value.

## Source of Truth

Codex CLI writes a session JSONL file under `CODEX_HOME/sessions` (or `~/.codex/sessions`). For live sessions on this host, the file contains:

```json
{"type":"turn_context","payload":{"model":"gpt-5.5", ...}}
```

The filename includes the Codex session UUID emitted to our JSON stream as `thread.started.thread_id`.

This is better than asking deep-worker's provider config because it is the actual model selected by Codex for that session. It also avoids a new runtime dependency on the OpenAI Agents SDK. The existing provider README explicitly states that `codex_provider` has zero runtime dependency on that SDK.

## Proposed Flow

1. Codex starts normally with `ExecArgs.model=None` when no model was specified.
2. `CodexChatModel._consume_turn()` receives `ThreadStartedEvent` from the Codex JSON stream.
3. When `state.codex_session_uuid` is first set, `CodexChatModel` resolves the display model by reading the matching Codex session JSONL and extracting the first `turn_context.payload.model` string.
4. The model is stored on `CodexChatModel` as display-only state, not copied into `_model_name` and not used for subsequent `ExecArgs.model` unless the user explicitly configured a model.
5. `AgentProcess` registers a callback on Codex models after `create_model()`. The callback updates `SessionManager.launch_config["model"]` and `AgentConfig.model`/display metadata once the real model is known.
6. `SessionManager` persists the updated launch config and emits a metadata/status event through its existing notifier path.
7. `deep_worker_web` consumes the event and updates the cached `AgentInfo.model`, then broadcasts an `agent_metadata` message to frontend clients.
8. On agent restart, before a new turn arrives, `AgentProcess` also checks the persisted `codex_session_state.json` session UUID and resolves the model from the existing Codex session JSONL. This repairs reconnect/restart display without polling.
9. `exposed_get_metadata()` returns the resolved model for later web reconnects and `dwcli --status` style consumers.

## Event-Driven Details

There is no sleep/retry loop. The only trigger is Codex's `ThreadStartedEvent`, which means Codex has created a concrete session. The resolver then performs one synchronous filesystem lookup/read against the matching session file.

If the session file or `turn_context.payload.model` is not present at that moment, the resolver returns `None` and no display update is emitted. On later process restart, the persisted Codex session UUID gives one more deterministic startup-time recovery point. We do not poll for the value later.

Matching session file:

- base dir: `Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"`
- file glob: `**/*<thread_id>.jsonl`
- parse line by line until the first object with `type == "turn_context"` and string `payload.model`

## Metadata Contract

Add display-only resolved model support without changing the execution contract:

- `AgentConfig.resolved_model` remains execution-oriented and may return `None` for Codex default.
- `AgentConfig.display_model` may return the dynamically learned Codex model once set.
- `launch_config["model"]` is NOT mutated for an agent launched with `config.model is None`. Display continuity is provided in-memory via `set_dynamic_display_model`; the durable execution config stays `None` so the next restart resolves the provider's own default. Only an explicit operator `--model M` pins execution. (Narrowed 2026-06-29 by `DESIGN_FIX_CROSS_PROVIDER_MODEL_POISON.md` — the original wording "passing this known value as `--model` is acceptable" is what allowed a cross-provider stale model to be re-pinned in the ca1_init incident; see that design for the resolver-guard mechanism.)
- Never write `"unknown"` as a model.

## API Changes

### `codex_provider`

Add:

```python
def resolve_model_from_session_log(thread_id: str, *, codex_home: Path | None = None) -> str | None
```

Add to `CodexChatModel`:

```python
def set_resolved_model_callback(self, callback: Callable[[str], Awaitable[None] | None]) -> None
@property
def resolved_model_name(self) -> str | None
```

The callback is invoked at most once per model value.

### `deep_worker_cli`

Add to `SessionManager`:

```python
async def update_launch_config_model(self, model: str) -> None
```

This updates the persisted metadata atomically using the existing `_persist()` path and then notifies subscribers with a payload containing `provider` and `model`.

`AgentProcess` registers the callback after model creation when the model exposes `set_resolved_model_callback`. For Codex agents restarted with `model=None`, it also resolves a saved `codex_session_state.json` UUID from the existing Codex session log at startup and feeds the same metadata update path.

`exposed_get_metadata()` uses:

1. `config.model` if explicit;
2. dynamic display model if known;
3. `config.resolved_model` for providers with static/env defaults;
4. `None` if genuinely unknown.

### `deep_worker_web`

Extend subscriber routing for metadata/status payloads so an event containing `model` and/or `provider` updates `AgentInfo` and broadcasts `agent_metadata`.

## TDD Plan

1. Add failing unit tests for Codex session-log parsing:
   - extracts `turn_context.payload.model`
   - returns `None` when no matching session file exists
   - ignores malformed/non-matching lines
2. Add failing unit tests for `CodexChatModel` callback:
   - receives model after `ThreadStartedEvent`
   - callback invoked once for same value
   - `_model_name` remains `None` when no explicit model was passed
3. Add failing unit tests for `SessionManager.update_launch_config_model()`:
   - persists model into existing `launch_config`
   - does not write `"unknown"`
   - emits notifier payload with `model`
4. Add failing unit tests for `exposed_get_metadata()` returning dynamic model.
5. Add web unit test: subscriber payload with `type=status_change`, `provider`, and `model` updates cache and emits `agent_metadata`.
6. Add no-mock integration test for the practical path using a temporary Codex session JSONL and the real parser/helper. This avoids a live Codex API call but uses real filesystem layout and JSONL parsing.

## Non-Goals

- Do not call OpenAI APIs to infer model defaults.
- Do not add OpenAI Agents SDK as a runtime dependency in this feature.
- Do not modify Codex CLI config or `~/.codex/config.toml`.
- Do not change the model used by a running Codex session after it starts.
