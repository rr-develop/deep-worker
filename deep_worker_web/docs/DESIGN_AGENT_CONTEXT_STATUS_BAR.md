# DESIGN: Show agent context size in the web UI status bar (event-driven; no polling)

## User requirement

> After every message in the web UI, the status line (which lists
> provider, model, uptime) must also display the current context size for
> the agent (and for its task manager, when a task manager is used).
> Display it according to the provider's data; if there is none, fall back
> explicitly to our own estimate marked "est.".
> Source: the last message received from the server for that agent
> and for its task manager (when applicable).

## What "current context size" means

Two semantically different kinds of data, both stored in
the agent's data directory, in `token_stats.jsonl`:

1. **`usage_semantics = context_snapshot`** — the provider itself reported
   a gauge: `context_used`, `context_size`. This is ground truth from the provider
   (grok / claude-code-sdk / codex via their own state-machine event).
   Format: `<used>/<size>` (for example, `42.1k/200k`).
2. **`usage_semantics = tokens`** — the provider sent only
   consumed tokens (`total_tokens` / `prompt_tokens` / `completion_tokens`),
   but did NOT send the window size. The numbers themselves are **exact, from the
   provider's tokenizer** — not an estimate. Format: `<used> tokens` (no qualifier).
   The `(est.)` qualifier is reserved for a future path where the system
   estimates consumption **without** provider data (chars/4, a local
   tokenizer, etc.). There is currently no such path in the code, so
   `(est.)` must not appear in the UI.

`UsageRecord.kind` separates records between the **main agent** (`agent`)
and the **task manager** (`tm`). The record is written at the moment the model is called,
via `TokenStatsCallbackHandler` or `UsageEventCallback` in
`stats/callback.py`.

The UI must show this data for:
- the **agent** — always, whenever a record exists,
- the **TM** — only if `agent.has_task_manager is True` (the criterion is already
  used in `updateTmControls()`, `main.js:1398`).

## Architectural contract (important)

### What is NOT polling (and is allowed)

- Local rendering of an `AgentInfo` already received from the server (REST snapshot
  on page load + push updates over WS). We make no new REST requests.

### What is polling (and is forbidden)

- Periodic fetch of `/api/agents/{name}/stats`.
- Periodic re-read of `token_stats.jsonl` from disk.
- Periodic RPyC `get_status()` / `get_metadata()`.

### How data reaches the UI (event-driven)

1. **Page load** (`GET /api/agents`): `AgentInfo` already contains all status bar
   fields (`provider`, `model`, `started_at`). We add 6 new fields:
   `agent_context_used`, `agent_context_size`, `agent_context_semantics`,
   `tm_context_used`, `tm_context_size`, `tm_context_semantics`. The backend
   fills them from the **latest** record in `token_stats.jsonl` for each
   `kind` (per `usage_semantics`). They are always pulled in on discovery
   (`_discover_agents`) and on reconcile (`_reconcile_from_registry`).

2. **WS push** for online updates: every single call to the
   model creates a usage record and immediately sends `model_context_changed`
   from the agent process. `deep_worker_web` converts each such event into a
   separate `agent_context_changed` with the fields
   `{agent_name, agent_context_used, agent_context_size, agent_context_semantics,
     tm_context_used, tm_context_size, tm_context_semantics}`.
   One model call must produce exactly one status bar update. This applies
   to user requests, tool result handling, tool call planning,
   internal reasoning steps, the TaskManager, and any other reason a model is called.
   A status transition, a prompt/turn boundary, and the request source do not
   take part in this contract. Polling is not used.

3. **The existing stream channel** `agent_message.message.type === "token_update"`
   (`stream_processor.py:204-214`) **contains only `total_tokens`, without
   `kind` (agent vs tm) and without `usage_semantics` / `context_size`**.
   This channel:
   - comes ONLY from the main agent (TaskManager.invoke goes through
     `manager_agent.ainvoke({...})` directly, without `notify_callback` —
     see `task_manager/task_manager.py:546`), so it applies only to the
     agent, not to the TM;
   - does NOT report the provider's `context_size`;
   - does NOT distinguish `usage_semantics` (the provider sent a snapshot vs
     only consumed tokens).
   Decision: we use this channel as a **live-rough best-effort hint**
   for the main agent ONLY when the REST snapshot has not yet provided a snapshot
   (i.e. `agent_context_semantics is None`). We show it as
   `<total_tokens> tokens` (exact from the provider; `(est.)` is
   reserved for paths without provider data). **Never overwrite** a snapshot with it and
   **never write** it into the TM context (no kind marker).

### What we do NOT do

- We do not touch the existing `/api/agents/{name}/stats` (the aggregator stays
  as it was, for the Stats tab).
- We do not add a new REST endpoint.
- We do not introduce periodic updates.
- We do not modify `stream_processor.py` — the incoming `token_update` already
  contains everything it contains; the rest comes from the JSONL.

## Changes

### 1. Backend — `deep_worker_cli/stats/token_stats.py`

Add a new function and a field in the aggregate:

```python
@dataclass
class ContextSnapshot:
    """Latest context data point for a (kind, usage_semantics) bucket."""

    context_used: int | None = None
    context_size: int | None = None
    usage_semantics: str | None = None  # "context_snapshot" or "tokens"
    ts: str | None = None


def latest_context_by_kind(
    records: list[UsageRecord],
) -> dict[str, ContextSnapshot]:
    """Return ``{kind: latest snapshot}`` across all relevant kinds.

    Filter:
    - Only ``kind in {"agent", "tm"}`` are considered. ``kind="other"``
      (sub-agent / nested model calls, ``_KNOWN_KINDS`` line 95) is
      deliberately excluded — the UI never reflects those.

    Latest = max by parsed UTC ``ts``, tie-break by file order (later
    line wins on identical ts). Only records with non-null
    ``context_used`` OR non-null ``context_size`` are considered; a
    snapshot record with both fields = None is malformed and excluded.

    Returns an empty dict if no relevant records exist (UI renders ``-``).
    The keys of the returned dict are always a subset of ``{"agent",
    "tm"}``.
    """
```

Extend `BucketTotals` with a new field:

```python
class BucketTotals:
    ...
    latest_by_kind: dict[str, ContextSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            ...
            "latest_by_kind": {
                kind: {
                    "context_used": s.context_used,
                    "context_size": s.context_size,
                    "usage_semantics": s.usage_semantics,
                    "ts": s.ts,
                }
                for kind, s in self.latest_by_kind.items()
            },
        }
```

And in `aggregate_records()`, fill `totals.latest_by_kind` in the same pass
as the `context` aggregate (after the token aggregates). `kind ∈
{agent, tm}` — there is never a third (`_KNOWN_KINDS` already
restricts it).

**CRITICAL**: the new key `"latest_by_kind"` does NOT break existing
clients of the REST aggregate (it is just an additional key in `to_dict()`;
`stats.js` already works with `context.latest_context_used` and ignores
unknown fields).

### 2. Backend — `deep_worker_web/app/schemas/agent_schemas.py`

Add 6 fields to `AgentInfo`:

```python
agent_context_used: int | None = Field(default=None, ...)
agent_context_size: int | None = Field(default=None, ...)
agent_context_semantics: str | None = Field(default=None, ...)
tm_context_used: int | None = Field(default=None, ...)
tm_context_size: int | None = Field(default=None, ...)
tm_context_semantics: str | None = Field(default=None, ...)
```

Semantics:
- `agent_context_used` / `tm_context_used` — int or None.
- `agent_context_size` / `tm_context_size` — int or None
  (None ⇔ semantics=tokens, no window size).
- `*_context_semantics` ∈ {`"context_snapshot"`, `"tokens"`, `None`}.
- All None ⇔ the agent has no records in the JSONL yet (a new process, or
  no turn has happened yet).

### 3. Backend — `deep_worker_web/app/services/agent_service.py`

In **three** places where `AgentInfo` is created / updated, add a
helper:

```python
def _fill_context_fields(self, info: AgentInfo) -> None:
    """Read token_stats.jsonl for ``info.name`` and fill the 6 new fields."""
    from deep_worker_cli.storage_paths import agent_root
    from deep_worker_cli.stats.token_stats import read_stats_file

    try:
        data_dir = agent_root(info.name, create=False)
    except Exception:
        return  # never break status update
    try:
        records = read_stats_file(data_dir / "token_stats.jsonl").records
    except Exception:
        return
    snapshots = latest_context_by_kind(records)
    agent_snap = snapshots.get("agent")
    tm_snap = snapshots.get("tm")
    info.agent_context_used = (agent_snap.context_used if agent_snap else None)
    info.agent_context_size = (agent_snap.context_size if agent_snap else None)
    info.agent_context_semantics = (agent_snap.usage_semantics if agent_snap else None)
    info.tm_context_used = (tm_snap.context_used if tm_snap else None)
    info.tm_context_size = (tm_snap.context_size if tm_snap else None)
    info.tm_context_semantics = (tm_snap.usage_semantics if tm_snap else None)
```

Apply in:
- `_discover_agents()` and `_reconcile_from_registry()` only for the initial
  restoration of the last snapshot when the web backend connects.
- In normal operation the only update source is `model_context_changed`,
  sent by `TokenUsageRecorder.record()` after every model call.
- `status_change`, `_update_agent_status` and `send_message.on_success` have no
  right to initiate or suppress a Context update.

**Hot-path note**: `GET /api/agents` reads the in-memory `self._agents`, which is
updated by every `model_context_changed`. So REST always returns the
latest value already sent to the status bar.

**Cardinality contract**: N recorded model calls must produce N
`agent_context_changed` events, even if two adjacent calls reported the same
number. The diff-check does not apply to model-call events.

### 4. Backend — new WS event `agent_context_changed`

In `agent_service.py`, after `_fill_context_fields(info)`, we mirror the WS event:

```python
await self._broadcast_message({
    "type": "agent_context_changed",
    "agent_name": agent_name,
    "agent_context_used": info.agent_context_used,
    "agent_context_size": info.agent_context_size,
    "agent_context_semantics": info.agent_context_semantics,
    "tm_context_used": info.tm_context_used,
    "tm_context_size": info.tm_context_size,
    "tm_context_semantics": info.tm_context_semantics,
})
```

### 5. Frontend — `deep_worker_web/static/index.html`

In `.agent-info-bar` (provider/model/uptime) add two spans:

```html
<span class="status-label">Context:</span>
<span id="agent-context" title="Token usage. Snapshot from provider (used/size) when both fields are reported, or just used (exact from the provider's tokenizer) when only token counts are reported.">-</span>
<span id="agent-tm-context-wrap" style="display: none;">
    <span class="status-label">TM:</span>
    <span id="agent-tm-context" title="TaskManager context. Same format as the agent's context.">-</span>
</span>
```

The TM wrapper is shown via `style.display` if
`agents[name].has_task_manager === true`.

### 6. Frontend — `deep_worker_web/static/js/main.js`

A pure function (easy to unit-test in node):

```js
// Pure — takes the 3 context fields, returns the rendered string.
function formatAgentContext(used, size, semantics) {
    if (used == null && size == null && semantics == null) return '-';
    const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(n);
    if (semantics === 'context_snapshot' && size != null) {
        return `${fmt(used ?? 0)}/${fmt(size)}`;
    }
    if (semantics === 'context_snapshot') {
        // Anomaly: provider declared snapshot but omitted size. Render
        // explicit "unknown" marker so the user sees the gap rather than
        // mistaking a snapshot-without-size for a normal tokens-only
        // reading.
        return used != null ? `${fmt(used)}/?` : '-';
    }
    // semantics === 'tokens' or unknown → provider only reported consumed
    // tokens. The number itself is EXACT from the provider's tokenizer —
    // never label it "(est.)". The qualifier is reserved for a future
    // estimation path (chars/4, local tokenizer fallback, etc.) that
    // computes a token count WITHOUT provider data.
    return used != null ? `${fmt(used)} tokens` : '-';
}
```

And three places of application:
1. **Page load / agent re-select** in `selectAgent(name)` — read
   `agents[name].agent_context_*` / `tm_context_*`, render both
   spans.
2. **A new handler `handleAgentContextChanged(data)`** in
   `handleWebSocketMessage` — update `agents[name]` and re-render the
   info bar if `data.agent_name === selectedAgent`.
3. **The existing `handleAgentMessage`** — add a branch
   `if (message && message.type === "token_update")`:
   - update ONLY `#agent-context`, never `#agent-tm-context`;
   - **only if** `agents[name].agent_context_semantics == null`
     (i.e. there is no snapshot yet — this is the first message, nothing to overwrite);
   - show `<total_tokens> tokens` (exact from the provider; `(est.)`
     is reserved for a future path WITHOUT provider data) via
     `formatAgentContext`.

`handleAgentDiscovered` must also mirror the fields into `agents[name]`,
so that page load and the live channel use the same render path.

### 7. CSS — `deep_worker_web/static/css/style.css`

The `.status-label` styles already exist; add `#agent-tm-context-wrap` with
`margin-left: 12px` to visually separate it from the main context.

## Tests (TDD)

### Unit backend

`deep_worker/deep_worker_cli/tests/unit_tests/test_token_stats_latest_by_kind.py`:

- empty records → `{}`.
- only token records (semantics=tokens, no context_used) → `{}`.
- only snapshot records (semantics=context_snapshot, with used/size)
  → `{"agent": snapshot}`.
- mixed per kind (kind=agent and kind=tm, different semantics) → both
  keys, correct values.
- tie-break by ts: two records of the same kind, the later one wins.
- **file-order tie-break**: 3 records of the same kind with identical ts —
  the last one in the file wins (an explicit test for the stated semantics).
- malformed record (missing `kind`) — skipped without crashing.
- **kind="other" is excluded**: a `kind="other"` record with context does NOT
  get into the result (the UI never renders sub-agent calls).
- **negative malformed snapshot**: a record
  `usage_semantics="context_snapshot"` with both context fields None
  is excluded.
- **broken jsonl swallowed**: plant a corrupt JSONL →
  `_fill_context_fields` does not raise an exception outward (the status
  update does not fail).
- `BucketTotals.latest_by_kind.to_dict()` contains the expected keys.

**Mutations** (RED-first, must be CAUGHT):
- M1: drop the kind filter → fails the mixed-kind test.
- M1.5: forget to exclude `kind="other"` → fails test_kind_other_excluded.
- M2: drop the ts tie-break → fails the tie-break test.
- M2.5: drop the file-order tie-break on identical ts → fails
  test_file_order_tie_break.
- M3: drop `usage_semantics` from the output → fails snapshot detection.
- M4: drop `tm` handling → fails the TM test.
- M5: ignore empty context_used (accept only used+size together) →
  fails the edge case where `context_used is None, context_size is not None`.
- M11: `_fill_context_fields` propagates an exception on corrupt JSONL →
  fails test_broken_jsonl_is_swallowed.
- M12: `_fill_context_fields` reads disk on every broadcast without a cache →
  fails the perf-throttling test (100 broadcasts → max 1 read per
  recent window).
- M13: the backend does not gate `tm_*` fields by `has_task_manager` →
  fails test_backend_clears_tm_when_no_task_manager (if the agent has
  `has_task_manager=False`, `info.tm_context_*` must be None, even
  if the JSONL contains a `kind="tm"` record — the UI gate hides it visually,
  but the REST snapshot must be clean).

### Unit web — `deep_worker_web/tests/unit/test_agent_context_mirror.py`

- `AgentInfo` round-trip with the 6 new fields.
- `AgentService._fill_context_fields(info)` (or helper) — for a
   planted `token_stats.jsonl` fills the 6 fields correctly.

### Integration web

`deep_worker_web/tests/integration/test_every_model_call_refreshes_context.py`:

- two consecutive model calls of the main agent produce exactly two
  `agent_context_changed` events with the corresponding values;
- a TaskManager model call updates the TM Context without `status_change`;
- `GET /api/agents/{name}` returns the latest value of each kind.

### UI tests

`deep_worker_web/tests/unit/test_format_agent_context.mjs`:

- `formatAgentContext(null, null, null)` → `'-'`.
- `formatAgentContext(42100, 200000, 'context_snapshot')` →
  `'42.1k/200k'`.
- `formatAgentContext(1500, null, 'tokens')` → `'1.5k tokens'`
  (exact from the provider; no `(est.)`).
- `formatAgentContext(0, 1000, 'context_snapshot')` → `'0/1k'`.
- `formatAgentContext(null, 1000, 'context_snapshot')` → `'0/1k'`
  (treat missing used as 0, not as `-`).
- `formatAgentContext(800, null, 'context_snapshot')` → `'800/?'`
  (anomaly: snapshot declared but size missing — NOT `est.`).
- `formatAgentContext(800, null, 'tokens')` → `'800 tokens'`
  (semantics=tokens without size — exact from the provider; `(est.)`
  is reserved for a future path WITHOUT provider data).

E2E UI (Playwright-style, via the existing test_send_real_ui.py):

- Bring up the page with an agent; seed `token_stats.jsonl` with a snapshot;
  render `.agent-info-bar` → it contains `42.1k/200k` for the context
  and the same format for the TM (if `has_task_manager`).
- Without a snapshot (only `tokens` records) → renders `<N> tokens`
  (exact from the provider; `(est.)` is NOT added, since this is provider
  data).
- Without records → `-`.
- **test_isolation_preserves_snapshot**: first render a snapshot
  (`semantics=context_snapshot`, used=42k, size=200k), then simulate a
  `token_update` (total_tokens=5000) → the UI stays at `42.1k/200k`, and does not
  become `5k tokens` (exact from the provider; NOT `(est.)`).

### Unit frontend (format edge-cases)

`deep_worker_web/tests/unit/test_format_agent_context.mjs`:

- `formatAgentContext(null, null, null)` → `'-'`.
- `formatAgentContext(42100, 200000, 'context_snapshot')` →
  `'42.1k/200k'`.
- `formatAgentContext(1500, null, 'tokens')` → `'1.5k tokens'`
  (exact from the provider; no `(est.)`).
- `formatAgentContext(0, 1000, 'context_snapshot')` → `'0/1k'`.
- `formatAgentContext(null, 1000, 'context_snapshot')` → `'0/1k'`
  (treat missing used as 0, not as `-`).
- `formatAgentContext(800, null, 'context_snapshot')` → `'800/?'`
  (anomaly: snapshot declared but size missing — NOT `est.`).
- `formatAgentContext(800, null, 'tokens')` → `'800 tokens'`
  (semantics=tokens without size — exact from the provider; `(est.)`
  is reserved for a future path WITHOUT provider data).

### Mutations for UI/JS

- M6: drop `has_task_manager` gating → the TM block is visible for an agent without a TM.
- M7: drop the `token_update` semantics check (overwrite the snapshot with the stream)
  → fails test_isolation_preserves_snapshot (the snapshot must remain after
  the live update).
- M8: drop TM-context in the token_update handler → fails test_isolation.
- M9: drop the `est.` marker (put it back on provider numbers)
  → fails the UI test (one sees "1500 tokens (est.)" although the data is from
  the provider — this is a logic bug, see the 2026-08-17 discussion).
- M10: drop `formatAgentContext` → fails the format unit test.

## Scope / non-goals

- Minimal changes: 1 dataclass + 1 function in CLI stats; 6 fields in
  the schema; 1 helper + 3 places of application in agent_service; 2 spans in HTML;
  1 pure function + 3 render points + 1 case in a switch in JS.
- We do **not** touch `stream_processor.py` (the stream channel is kept as is).
- We do **not** add new REST endpoints.
- We do **not** do polling.
- We do **not** touch the Stats tab (`/api/agents/{name}/stats`); it
  keeps working with the old `latest_context_used` of the aggregate.
- We do **not** separate providers: standard LangChain (semantics=tokens)
  and custom ones (semantics=tokens or context_snapshot) are handled
  uniformly via the JSONL.

## Documentation updates

- `IMPLEMENTATION_STATUS.md` — a note about the new feature.
- A comment "why snapshot-based, not streaming TM" next to the helper
  in `agent_service.py`.
