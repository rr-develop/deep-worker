# DESIGN: Claude CLI API error must never become the agent's final answer

Status: PROPOSED
Date: 2026-08-04
Policy anchors: `docs/POLICIES.md` §9 (No silent API errors), §10 (honest
integration tests), No-Agent-Error-State Policy.

---

## 1. Problem (observed in production)

The Claude CLI answered a turn with the *text*:

```
API Error: 400 messages.18.content.0.tool_result.content.1: Input tag
'image_url' found using 'type' does not match any of the expected tags:
'document', 'image', 'search_result', 'text', 'tool_reference'
```

This string was stored in the checkpointer as ordinary **`AIMessage.content`**
(`tool_calls=[]`, `usage_metadata=None`, `response_metadata={}`), i.e. the
transport failure became *the agent's answer*.

Consequences measured on the live agent:

* turn 19:00:36 → 19:00:38 finished in ~2 s, `log_token_usage: rm={}`, zero
  tools invoked — no work was done on the user's request;
* the TaskManager treated the error text as the model's reply, issued
  corrective instructions ("this is your own failure…"), looped, and finally
  declared `status=unable`;
* the web UI showed **no** error — the agent looked like it had answered.

That is a textbook violation of POLICIES.md §9: a provider API error was
swallowed and disguised as a benign assistant reply.

### 1.1 Why the current code does this

`chat_model.py` builds `response_text` **exclusively** from `TextBlock`
(lines 1364-1368). The CLI repacks the upstream HTTP 400 into a normal
`TextBlock`, so the provider faithfully returns it as assistant speech.

The SDK *does* expose failure signals which we never read:

| SDK field | Defined at | Read by our code? |
|---|---|---|
| `ResultMessage.is_error: bool` | `types.py:1012` | **no** |
| `ResultMessage.subtype`, `.errors`, `.stop_reason` | `types.py:1009-1022` | **no** |
| `AssistantMessage.error: AssistantMessageError` | `types.py:923` | **no** |
| `AssistantMessage.usage`, `.stop_reason` | `types.py:924-926` | **no** |

`chat_model.py:1442-1453` consumes `ResultMessage` only for `session_id` and
token stats. A grep over `claude_code_provider/` finds **zero** occurrences of
`is_error` / `subtype ==` / `event.error` in production code.

### 1.2 Ground truth from the CLI transcript (verbatim, measured)

Read from the CLI's local on-disk transcript directory. The CLI marks these turns
**inconsistently** — this is the crux of the detector design:

| transcript line | `isApiErrorMessage` | `apiErrorStatus` | `stop_reason` | `usage` |
|---|---|---|---|---|
| 523, 525, 527, 529 (`d398f65f`) | *absent* | *absent* | `end_turn` | all-zero |
| 524, 526, 528, 530 (`270068eb`) | *absent* | *absent* | `end_turn` | all-zero |
| 536 (`270068eb`) | `True` | present | `stop_sequence` | all-zero |

So **we cannot rely on `isApiErrorMessage` alone** — in 8 of 9 recorded cases
the CLI did not set it, and `stop_reason` was the perfectly ordinary
`end_turn`. The one invariant present in **all nine** records is:

* the whole turn is a single text block starting with `API Error:`, **and**
* `usage` is entirely zero (`input_tokens == output_tokens ==
  cache_creation_input_tokens == cache_read_input_tokens == 0`), **and**
* no tool call was produced.

A genuine model answer always bills input tokens, so all-zero usage on a
text-only turn is a reliable structural corroborator.

### 1.2b Which fields are SDK signals and which are transcript-only

The CLI's on-disk transcript key `isApiErrorMessage` is **not** a `claude_
agent_sdk` field — a grep over the `claude_agent_sdk` source finds it nowhere.
The SDK exposes exactly **one** terminal-error boolean, `ResultMessage.
is_error` (`types.py:1012`); the transcript key exists only on disk. Earlier
drafts drew them as two independent signals; that is corrected here (round-2
REQUIRED nit c). The detector consumes exactly the SDK-exposed bool
`result_is_error`, populated from `ResultMessage.is_error`. `result_subtype`,
`result_errors`, `result_text`, `stop_reason` are `enrichment only` — used
to build the exception's message text (§3.2 precedence), never as
independent proof of error on their own.

### 1.2a Observed `usage` schema (why "all values are zero" is naive)

Every distinct `usage` key found across the incident transcript:

```
cache_creation, cache_creation_input_tokens, cache_read_input_tokens,
inference_geo, input_tokens, iterations, output_tokens, server_tool_use,
service_tier, speed
```

Note `service_tier`, `inference_geo`, `speed`, `iterations` (often `None`) and
the nested dicts `cache_creation`, `server_tool_use`. A predicate like "every
value is falsy" would be both wrong and fragile. Only the four integer token
counters are meaningful, which is exactly what `_TOKEN_COUNTER_KEYS` fixes.

### 1.3 Out of scope (explicitly)

*Why* the payload contained `image_url` (the multimodal/session-synthesis
question) is a **separate defect** and is NOT addressed here — by the user's
explicit instruction. This design only guarantees that **whatever** API error
occurs, it is surfaced as an error and never returned as the agent's answer.

Also out of scope: the DEFECT A fix (`status_callback.on_tool_error`). The
error arrives as a `TextBlock`, so `on_tool_error` is never called — that fix
is neither at fault nor a remedy here.

---

## 2. Goals / non-goals

**Goals**

G1. An API error returned by the CLI MUST NOT become `AIMessage.content`.
G2. It MUST be surfaced to the user as a visible, persistent UI error
    (POLICIES.md §9, `type="error"`) and appended to conversation history.
G3. The agent MUST stay usable (`IDLE`, not `ERROR`) — the failure is
    transient/recoverable at the account level; the next turn may succeed.
G4. The error MUST be persisted into conversation history and **included in
    the next request** to the model as a SystemMessage, so that — *if* that
    request reaches the model — it can continue working on the user's request
    and honestly report what failed.
    **Explicit limitation (reviewer MAJOR 2):** the runtime guarantees
    persistence and inclusion in the next request, and NOTHING more. If the
    underlying payload/session defect persists, the next request will fail
    with the same provider error *before* the model ever processes that
    SystemMessage. We do NOT promise the model "will continue"; we promise the
    error is never disguised as an answer, is visible, and is honestly
    re-reported every turn. There is NO internal retry and NO corrective loop.
G5. Detection must not depend on a CLI flag that is only sometimes set.

**Non-goals**

N1. Fixing the `image_url` payload (separate task).
N2. Changing the fatal/recoverable taxonomy or the TaskManager contract.
N3. Any new thread, any polling. (Project policy.)

---

## 3. Design

### 3.1 New recoverable exception type

Add to `claude_code_provider/chat_model.py`, next to `ClaudeUsageLimitError`:

```python
class ClaudeApiResponseError(RuntimeError):
    """Raised when the Claude CLI returns an upstream API error AS the turn.

    Carries structured context (MAJOR 1) so diagnostics stay honest and a
    future failover policy has something to key on. ``str(exc)`` renders the
    real upstream text via the precedence in §3.2 — never an empty string,
    never ordinary assistant prose.
    """

    def __init__(self, message, *, assistant_error=None, result_subtype=None,
                 result_errors=(), stop_reason=None):
        super().__init__(message)
        self.assistant_error = assistant_error
        self.result_subtype = result_subtype
        self.result_errors = tuple(result_errors)
        self.stop_reason = stop_reason
```

No failover behaviour is introduced now — a bare 400 is not sufficient grounds
to switch providers. The fields exist so the diagnosis is honest today and a
policy can be added later without re-plumbing.

**Classification: recoverable, provider-API.** Register the class *name* in
the existing provider-agnostic registry
`deep_worker_cli/exceptions.py`:

```python
PROVIDER_API_ERROR_TYPES = frozenset({
    "GrokAcpClientError",
    "GrokConcurrentPromptError",
    "CodexProtocolError",
    "ClaudeApiResponseError",      # ← added
})
```

It is deliberately **NOT** added to `FATAL_EXCEPTION_TYPES`: the account is
healthy, only this request failed. This satisfies G3.

Consequence — reusing machinery that already exists, no new plumbing:
`agent_process.py:4665` calls `is_provider_api_error(e)` → notify
`type="error"` with the real API text (persistent web panel, G2) →
`_append_recoverable_error_to_history(...)` (G2/G4) → agent settles to `IDLE`
(G3). The SystemMessage carrying the error is included in the next request
the runtime sends (G4) — see the explicit limitation in G4 and §3.4:
whether the model actually processes it depends on the request succeeding.

### 3.2 Detector

New module `claude_code_provider/api_error_detect.py` — pure, side-effect
free, unit-testable, **never raises**:

```python
_API_ERROR_PREFIX = "API Error:"

# Recognised integer token counters, taken from the OBSERVED transcript
# schema (§1.2a). Anything else in ``usage`` is ignored for the verdict.
_TOKEN_COUNTER_KEYS = frozenset({
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
})

class UsageVerdict(enum.Enum):
    KNOWN_ZERO = "known_zero"   # counters present, all int, all == 0
    NONZERO    = "nonzero"      # at least one recognised counter > 0
    UNKNOWN    = "unknown"      # nothing usable to judge on

def classify_usage(usage: Mapping[str, Any] | None) -> UsageVerdict:
    """Three-valued classification — ``None``/``{}`` is UNKNOWN, not zero."""

def looks_like_api_error_text(text: str) -> bool:
    return text.strip().startswith(_API_ERROR_PREFIX)

@dataclass(frozen=True)
class ApiErrorDetection:
    message: str                       # what str(exc) will render
    assistant_error: str | None
    result_subtype: str | None
    result_errors: tuple[str, ...]
    stop_reason: str | None

def detect_api_error(
    *, response_text, assistant_error, result_is_error,
    result_subtype, result_errors, result_text, stop_reason, usage,
) -> ApiErrorDetection | None:
    """Return a structured detection, or None. NEVER raises.

    The detector is guard-agnostic: the caller decides whether to consult it
    based on the authoritative pending-tool-calls state (§3.2b). Once
    consulted, the detector classifies the terminal turn on its own merits —
    prior tool blocks in the same turn do NOT suppress a terminal API error
    (reviewer round-2 MAJOR).
    """
```

`classify_usage` semantics (**BLOCKER fix**):

* `None`, `{}`, or a mapping with **no** recognised counter → `UNKNOWN`;
* a recognised counter present but not an `int` (str/float), or a `bool`
  (`bool` is an `int` subclass — explicitly excluded) → that key is ignored;
  if no usable counter remains → `UNKNOWN`;
* every usable recognised counter present and `== 0` → `KNOWN_ZERO`;
* any usable recognised counter `> 0` → `NONZERO`;
* any usable recognised counter `< 0` → `UNKNOWN` (malformed — the SDK
  should never emit a negative token count; treat as unusable, not as zero,
  and never as evidence of a real answer) — reviewer round-2 REQUIRED nit (a).

**Decision rule** — evaluated by the caller only when NOT returning
tool_calls (§3.2b). Once entered, prior tool blocks in the same turn are
irrelevant to classification:

1. `assistant_error` set → **error** (structured signal, sufficient alone).
2. Else `result_is_error is True` (the SDK's single terminal-error bool,
   `ResultMessage.is_error` — see §1.2b) → **error**. `result_subtype` /
   `result_errors` / `result_text` are used **only** to enrich the message
   text, not as independent proof of error.
3. Else `looks_like_api_error_text(response_text)` **AND**
   `classify_usage(usage) is KNOWN_ZERO` → **error**.
4. Else → not an error.

Rationale (accepting reviewer Q1 verdict): zero-usage alone is NOT enough, and
absent usage must NOT be read as zero. Rule 4 is the *conservative fallback*
for the unmarked real-world case (§1.2) and fires only when the textual marker
is corroborated by a **measured** all-zero token count. If the CLI ever
rewords the message away from the prefix, rules 1–3 (structured flags) are
expected to catch it; when no signal at all is available it is more honest to
miss a detection than to reclassify a legitimate answer as an error. Turns in
this repo legitimately quote `API Error:` (docs, logs, this very design) — they
bill tokens and are therefore `NONZERO`.

**Error message construction (MAJOR 1 fix).** `message` is never the raw
`response_text` blindly. Precedence:

1. `response_text` **iff** it is non-empty *and* carries the `API Error:`
   marker (the genuine upstream text — the production case);
2. else `result_text` (`ResultMessage.result`) if non-empty;
3. else `"; ".join(result_errors)` if any;
4. else a synthesised description from `assistant_error` /
   `result_subtype` / `stop_reason`, e.g.
   `Claude API error (assistant_error=invalid_request, subtype=error_during_execution)`.

This guarantees we never raise `ClaudeApiResponseError("")` and never present
ordinary partial assistant prose as the failure reason.

### 3.2b Guard: `returning_tool_calls`, not "saw any tool block" (round-2 MAJOR fix)

The previous draft's `saw_any_tool_block` guard was **wrong** and had to be
retracted. Its failure mode is exactly what the reviewer identified:

* A built-in tool (WebSearch, ToolSearch) runs *inside* the CLI, produces a
  `ToolUseBlock` in the same turn, and the turn can still terminate with
  `ResultMessage(is_error=True)` or a marker + known-zero text turn. The old
  guard would suppress classification on the basis of a prior, unrelated tool
  block, and the API error would silently become the assistant's answer —
  restoring the very bug this design exists to fix (G1 violation).
* Similarly, a proxy tool call might be resolved earlier in the turn, then
  `receive_response` continues and emits a terminal API error text.

The authoritative guard is not "did any tool block ever appear" but "are we
about to return `tool_calls` to LangGraph". Concretely:

```python
returning_tool_calls = bool(
    pending_tool_calls or proxy_mcp.has_pending_calls()
)
```

The detector is consulted **iff** `not returning_tool_calls` — the same
condition already used for the `_SESSION_LIMIT_MARKER` guard at
`chat_model.py:1563-1566`. When we ARE returning tool calls, the terminal
turn is a tool-dispatch turn and classification is not our concern. When we
are NOT, whatever the terminal signals say (structured or textual) is the
authoritative verdict and prior tool blocks in the turn do **not** immunise it.

Consequence for the detector signature: `saw_tool_block` is removed. The
receive loop no longer maintains `saw_any_tool_block`.

### 3.3 Wiring in `ainvoke`

Capture, inside the `receive_response` loop, without changing existing flow:

* on `ClaudeAssistantMessage`: remember `event.error` and `event.usage` (last
  non-`None` wins — the terminal assistant message is authoritative). We do
  **not** track a `saw_any_tool_block` flag (retracted in §3.2b).
* on `ResultMessage`: remember `event.is_error`, `event.subtype`,
  `event.errors`, `event.result`, `event.stop_reason`, and `event.usage` as a
  fallback when no assistant message carried usage.

Then, in the existing text-only guard block — **immediately after** the
`_SESSION_LIMIT_MARKER` check at lines 1559-1568, mirroring its shape and
placement. The `returning_tool_calls` guard is the same condition already
used for the usage-limit check — no new authoritative source (§3.2b):

```python
returning_tool_calls = bool(
    pending_tool_calls or proxy_mcp.has_pending_calls()
)
if not returning_tool_calls:
    detected = detect_api_error(
        response_text=response_text,
        assistant_error=last_assistant_error,
        result_is_error=bool(result_is_error),
        result_subtype=result_subtype,
        result_errors=result_errors,
        result_text=result_text,
        stop_reason=result_stop_reason,
        usage=last_usage,
    )
    if detected is not None:
        raise ClaudeApiResponseError(
            detected.message,
            assistant_error=detected.assistant_error,
            result_subtype=detected.result_subtype,
            result_errors=detected.result_errors,
            stop_reason=detected.stop_reason,
        )
```

Ordering: the usage-limit check runs **first** and stays untouched — a usage
cap remains fatal, which is the existing, correct behaviour.

The raise propagates out of `ainvoke` to the `agent_process._handle_invoke`
catch-all, which already implements the recoverable path (§3.1).

### 3.4 What the user and the model end up seeing — guaranteed vs best-effort

**Guaranteed by the runtime (MAJOR 2 fix — narrowed from the earlier draft):**

* **Web UI**: persistent error panel with the real text
  `Provider API error (ClaudeApiResponseError): API Error: 400 …` — not a
  toast, not silence.
* **History**: the recoverable error is appended, so it is inspectable later.
* **No `AIMessage` with the error text**: `ainvoke` raises; the checkpointer
  never records the failure as the assistant's answer.
* **Agent status**: stays `IDLE`; the invocation returns `success=False` with
  the real reason. **No** internal retry and **no** corrective loop.
* **Next request**: the recoverable-error SystemMessage is *included* in the
  next request the runtime sends.
* **TaskManager**: no longer mistakes the error for a model reply.

**Explicitly NOT guaranteed:**

* We do **not** promise the model will "carry on" or even see the
  SystemMessage. If the underlying payload/session defect persists (e.g. the
  same malformed `image_url` block is still in history), the next request will
  hit the same provider error *before* Claude ever processes the added
  SystemMessage. In that case the runtime honestly re-reports the error every
  turn — which is the correct behaviour under POLICIES §9 — rather than
  hiding it or spinning.
* Recovery of the *user's task* depends on the caller fixing the trigger (or
  on a separate defect being addressed — e.g. the `image_url` issue). This
  fix's contract is: **the API error is surfaced, not disguised**. Nothing
  more.

---

## 4. Alternatives rejected

* **Make it fatal (`AgentStatus.ERROR`).** Rejected: the account is fine and
  the next request may well succeed; flipping to ERROR needs operator action
  and contradicts the No-Agent-Error-State Policy for recoverable failures.
* **Detect on text only.** Rejected: false positives on turns that legitimately
  quote `API Error:` (this very repo's docs and logs).
* **Treat absent/empty `usage` as "empty"/all-zero.** Rejected on reviewer
  BLOCKER: a legitimate text-only reply whose transport did not record `usage`
  would be misclassified as an error whenever it starts with `API Error:`.
  Rule 4 requires `KNOWN_ZERO` — measured all-zero on at least one recognised
  integer counter — never inferred zero.
* **Reuse `response_text` verbatim as `str(exc)` for rules 1–3.** Rejected on
  reviewer MAJOR 1: could produce `ClaudeApiResponseError("")` or surface a
  normal partial assistant reply as the failure reason. §3.2's precedence
  ladder + structured fields on the exception fix this.
* **Detect on `isApiErrorMessage` only.** Rejected by measurement: absent in
  8 of 9 real records (§1.2). This is the trap that makes the naive fix useless.
* **Retry the request inside the provider.** Rejected: hides the error
  (violates §9), and the 400 here is deterministic — a retry loops forever.
* **Swallow and return an empty answer.** Explicitly outlawed by §9.

---

## 5. Test plan

### 5.1 Unit (`tests/test_api_error_detect.py`, `tests/test_chat_model_api_error.py`)

**Detector — `classify_usage` (BLOCKER pin-tests):**

U1. `None` → `UNKNOWN`.
U2. `{}` → `UNKNOWN`.
U3. `{"service_tier": "standard", "speed": None}` (only unrecognised keys) → `UNKNOWN`.
U4. `{"input_tokens": "0"}` (string, not int) → `UNKNOWN`.
U5. `{"input_tokens": True}` (bool) → `UNKNOWN` (`bool` excluded).
U6. `{"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}` → `KNOWN_ZERO`.
U7. `{"input_tokens": 0}` (single recognised counter, value 0) → `KNOWN_ZERO`.
U8. `{"input_tokens": 12, "output_tokens": 0}` → `NONZERO`.
U9. `{"input_tokens": 0, "unknown_field": 999}` → `KNOWN_ZERO` (unknowns ignored).
U10. `{"input_tokens": -1}` (negative int) → `UNKNOWN` (round-2 REQUIRED nit a — malformed, not treated as zero).

**Detector — decision rule:**

D1. real 400 text + `KNOWN_ZERO` usage + no flag → detected (**production case**).
D2. real 400 text + `NONZERO` usage → NOT detected (answer quoting an error).
D3. real 400 text + `usage=None` (`UNKNOWN`) → **NOT detected** (BLOCKER pin — no false positive on absent usage).
D4. real 400 text + `usage={}` (`UNKNOWN`) → **NOT detected** (BLOCKER pin).
D5. real 400 text + `{"service_tier": "standard"}` (`UNKNOWN`) → **NOT detected**.
D6. normal answer + `KNOWN_ZERO` usage → NOT detected (no marker).
D7. `assistant_error="invalid_request"` + normal text → **detected**, `str(exc)` is the synthesised description (MAJOR 1 pin), never empty.
D8. `assistant_error` set + `response_text=""` → **detected**, `str(exc)` non-empty and includes `assistant_error=invalid_request`.
D9. `result_is_error=True` + normal-looking text + `NONZERO` usage → **detected** (SDK bool wins — round-2 REQUIRED nit c).
D10. `result_is_error=True` + `errors=["upstream 503"]` + `result="Retryable"` + `response_text=""` → **detected**, `str(exc)` prefers `result_text` (MAJOR 1 pin), else joins `errors`.
D11. `result_is_error=True` + genuine `API Error: 400 …` text → detected, `str(exc)` uses the marker-carrying `response_text` (precedence 1).
D12. `result_is_error=False` + `result_subtype="error_during_execution"` + `result_errors=["..."]` + normal text + `NONZERO` usage → **NOT detected** (round-2 REQUIRED nit c — enrichment fields alone are not proof).
D13. malformed inputs (bytes, list, nested weirdness) → never raises; returns `None` in the ambiguous case.

Note: the detector itself no longer takes a `saw_tool_block` / tool-guard
parameter — the guard is done by the caller via `returning_tool_calls`
(§3.2b, tested by A4/A5/A6).

**`ainvoke` (fake SDK stream, harness style of `tests/test_chat_model.py`):**

A1. stream = `AssistantMessage([TextBlock("API Error: 400 …")], usage=all-zero)` + `ResultMessage(is_error=False)` → raises `ClaudeApiResponseError` carrying the real text; the exception's `assistant_error`, `result_subtype`, `result_errors`, `stop_reason` fields are populated from the stream.
A2. stream = `AssistantMessage([TextBlock("You've hit your session limit…")])` → still raises `ClaudeUsageLimitError` (no regression; usage-limit guard runs first).
A3. normal answer stream (non-zero usage, normal text) → returns `AIMessage` unchanged.
A4. **`returning_tool_calls=True`** — pending proxy tool call at the end of the turn + text that would otherwise trigger rule 3 (marker + `KNOWN_ZERO`) → returns `AIMessage(tool_calls=[…])` unchanged; detector NOT consulted (round-2 MAJOR pin: guard is authoritative pending state, not history).
A5. **Prior built-in `ToolUseBlock` in the same turn → terminal `ResultMessage(is_error=True)`** with no pending calls → **raises `ClaudeApiResponseError`** (round-2 MAJOR pin: history of tool blocks must NOT suppress terminal API error).
A6. **Resolved proxy tool earlier in turn → later terminal marker + `KNOWN_ZERO` text turn**, no pending calls at end → **raises `ClaudeApiResponseError`** (round-2 MAJOR pin, same rationale).
A7. `AssistantMessage(error="invalid_request")` + empty text + `ResultMessage` → raises `ClaudeApiResponseError` with non-empty `str(exc)`.
A8. `ResultMessage(is_error=True, errors=["…"], result="…")` with no assistant text → raises `ClaudeApiResponseError`, `str(exc)` from `result`/`errors`, never empty.

**Taxonomy:**

T1. `is_provider_api_error(ClaudeApiResponseError("x"))` is `True`.
T2. `is_fatal_exception(ClaudeApiResponseError("x"))` is `False`.
T3. `str(ClaudeApiResponseError("real text", assistant_error="invalid_request"))` includes `"real text"`; structured fields are preserved on the instance.

### 5.2 Integration (POLICIES.md §10 — no mocks on the path under test)

The test drives a **real** `deep-worker` subprocess through its normal
lifecycle. Our detector, `ainvoke`, `chat_model` wiring,
`agent_process._handle_invoke` recoverable branch,
`_append_recoverable_error_to_history`, RPyC notify pipe and session_manager
run unmocked.

We only need to inject one thing: the specific CLI response bytes we want the
provider to receive. We do that WITHOUT adding any production hook.

**Chosen mechanism (round-3 BLOCKER fix): pytest-owned `sitecustomize.py`
placed on the child's `PYTHONPATH` — no production hook.**

Rationale for the change (reviewer round-3 BLOCKER, corrected):
`PYTHONSTARTUP` is only consulted before the *first prompt of an interactive
interpreter* — CPython does NOT run it for a normal script/console-script
entry point. A `deep-worker` child is non-interactive, so a
`PYTHONSTARTUP` shim would never execute and the child would hit the real
SDK. The correct standard hook that fires on every non-`-S` startup is
`sitecustomize`, imported by `site.py` during normal interpreter
initialisation, before the entry point runs.

Concrete mechanism:

* The shim directory lives under
  `claude_code_provider/tests/_support/` and contains **exactly two files**:
  - `sitecustomize.py` — the top-level module Python's `site` machinery
    auto-imports on startup. Nothing else in the tree defines a
    `sitecustomize` module, so this shadowing is unambiguous while the
    directory is on `sys.path`.
  - `_fake_sdk_boot.py` — the actual monkeypatching logic (kept out of
    `sitecustomize` so the top-level module stays tiny and side-effect-free
    when the gate is off).
* The test injects the shim by prepending that directory to the child's
  `PYTHONPATH`. Python's normal startup imports `sitecustomize` first thing
  (before the `deep-worker` console-script entry point runs), which — and
  only which — imports `_fake_sdk_boot` and installs the fake transport.
* Double activation gate:
  1. the shim directory must be first on `PYTHONPATH` (only the pytest
     fixture puts it there);
  2. `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST=1` must be set (only the
     pytest fixture sets it — a stray copy of this env var on a prod host
     cannot activate the shim, because prod's `PYTHONPATH` does not contain
     the test-support directory).
  If either gate is missing, `sitecustomize` returns immediately without
  side effects (no import of `_fake_sdk_boot`, no monkeypatch).
* When active, the shim monkeypatches exactly one symbol in the
  `claude_agent_sdk` transport (the client factory used by
  `chat_model._client_manager`) to yield the recorded stream from the
  `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_SCRIPT` JSON file (verbatim §1.2
  payload). It does not touch `api_error_detect.py` or
  `chat_model.ainvoke`; it only shapes the bytes the SDK yields to
  `receive_response`.
* Interaction with `-S`: `deep-worker` is not launched with `-S`. If a
  future refactor ever adds it, the integration test explicitly asserts
  `-S` is NOT in the child's argv (a canary — see below — would not fire
  under `-S` and the test would fail early with a clear message rather than
  silently exercising the real SDK).

**Canary (round-3 BLOCKER requirement).** A broken bootstrap must not
present as an ambiguous timeout. The shim emits a one-shot canary the moment
it activates:

* the pytest fixture opens an `os.pipe()`, passes the write-end fd to the
  child via `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_CANARY_FD=<n>` and
  `pass_fds=(n,)` on `subprocess.Popen`;
* `_fake_sdk_boot`, on successful monkeypatch install, writes a fixed
  byte-string (`b"CANARY_OK\n"`) to that fd and closes its copy;
* the test reads exactly that byte-string from the read-end via an
  event-driven `loop.add_reader(...)` future BEFORE issuing the invoke.
  This is a genuine file-descriptor event, not polling.
* If the canary never arrives (broken shim, missed hook, `-S` present,
  gate off), the test fails with an unambiguous
  `"seam bootstrap did not fire (canary missing)"` — not the invoke
  timeout.

**Threat boundary and no-prod-hook proof:**

* Production `deep_worker_cli` / `claude_code_provider` code has **zero**
  imports from `claude_code_provider/tests/`.
  `test_no_prod_import_of_test_support` (lint-style, greps the prod tree)
  RED-fails on regression.
* The `tests/_support/` directory is never on the production `sys.path`;
  the test-support `PYTHONPATH` prefix is the ONLY way to bring in the
  `sitecustomize` shim, and only the pytest fixture sets it.
* Mutation test `test_bypass_detector_reddens`: the same integration
  scenario, but with the real detector short-circuited to always return
  `None`, MUST end with the API text in `AIMessage.content` and no
  `type="error"` notify — proving the assertions really do fail when the
  detector is bypassed. Preserved from round 2 (reviewer round-3 explicit
  requirement).

**Test skeleton — `tests/test_api_error_integration.py`:**

1. Create the `os.pipe()` for the canary; record the read/write fds.
2. Spawn a real `deep-worker` agent (`subprocess.Popen`) with:
   - `env` prepending `<repo>/claude_code_provider/tests/_support/` to
     `PYTHONPATH`;
   - `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST=1`;
   - `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_SCRIPT=<verbatim §1.2 payload>.json`;
   - `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_CANARY_FD=<write-end fd number>`;
   - `pass_fds=(<write-end fd>,)`;
   - argv must NOT include `-S` (assert);
   - agent config disables all mutating tools (read-only agent).
3. Await the canary on the read-end via `loop.add_reader` (fd event) with a
   short deadline (≤5 s). Missing canary → immediate, unambiguous failure.
4. Await `status=idle` by connecting via RPyC and reading
   `get_session_metadata()` on the initial connection — the process is only
   registered in the RPyC registry once idle-ready (true process-level
   event, no `stat`/mtime polling, no `to_thread`).
5. Send a real invoke via RPyC. Subscribe to notify events via the existing
   `subscribe_messages` event surface — a real event, not a poll.
6. Assert:
   - `invoke` returns `success=False`;
   - status is still `idle` (**not** `error`);
   - a notify with `type="error"` was emitted carrying the real API text;
   - the last `AIMessage` in the thread state does **not** contain the API
     text (checkpointer inspection);
   - conversation history contains a `SystemMessage` with the error;
   - no mutating tool was invoked (config disabled them; assertion reads
     the RPyC-exposed tool-call log).
7. `finally`: close both fds, `SIGTERM` the agent (via `deep-worker --stop`
   or direct signal on the recorded pid), wait ≤5 s event-driven on process
   exit, `SIGKILL` if still alive. Overall budget ≤60 s
   (`per the bounded integration-test feedback policy`).

Why fault injection is legitimate here: we cannot force Anthropic to return a
400 on demand (§10's narrow exemption), so we shape the real reply at the
real transport boundary rather than mocking our own detector.

---

## 6. Files touched

| File | Change |
|---|---|
| `claude_code_provider/api_error_detect.py` | **new** — pure detector |
| `claude_code_provider/chat_model.py` | `ClaudeApiResponseError`; capture `error` / `usage` / `is_error` / `subtype` / `errors` / `result` / `stop_reason`; compute `returning_tool_calls`; raise after the usage-limit guard |
| `deep_worker/deep_worker_cli/deep_worker_cli/exceptions.py` | add `"ClaudeApiResponseError"` to `PROVIDER_API_ERROR_TYPES` |
| `claude_code_provider/tests/test_api_error_detect.py` | **new** — unit |
| `claude_code_provider/tests/test_chat_model_api_error.py` | **new** — unit |
| `claude_code_provider/tests/test_api_error_integration.py` | **new** — integration |
| `claude_code_provider/tests/_support/sitecustomize.py` | **new** — pytest-only top-level `sitecustomize`; on-startup shim entry point; gated on `PYTHONPATH` prefix + `DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST=1`; imports `_fake_sdk_boot` and writes canary; no side effects when gate is off |
| `claude_code_provider/tests/_support/_fake_sdk_boot.py` | **new** — actual fake-transport install + canary write; imported only by the shim `sitecustomize.py` above |
| `claude_code_provider/tests/test_no_prod_import_of_test_support.py` | **new** — lint-style: greps production tree, asserts zero imports of `tests/_support/*` |

No production code imports the seam. No architecture, contract, or policy
change. No new threads. No polling.

---

## 7. Reviewer feedback log

### Round 3 (2026-08-04) — resolutions

**BLOCKER (`PYTHONSTARTUP` doesn't fire for non-interactive entry points)
— RESOLVED.** Correct: `PYTHONSTARTUP` is only consulted before an
interactive prompt. Replaced with pytest-owned `sitecustomize.py` on the
child's `PYTHONPATH` prefix — the standard on-startup hook that Python
imports via `site.py` before the entry point runs (unless `-S`). See §5.2
"Chosen mechanism (round-3 BLOCKER fix)". Canary via `os.pipe()` fd
(`loop.add_reader`, event-driven) proves the shim executed; a broken
bootstrap fails with an unambiguous "seam bootstrap did not fire" message,
not an ambiguous invoke timeout. §6 filenames updated:
`tests/_support/sitecustomize.py` + `tests/_support/_fake_sdk_boot.py`.
Mutation-test `test_bypass_detector_reddens` and the no-prod-import lint
test preserved.

### Round 2 (2026-08-04) — resolutions

**MAJOR (`saw_any_tool_block` guard hole) — RESOLVED.** The `saw_any_tool_
block` guard is retracted (§3.2b). The detector no longer takes any
tool-history parameter; the caller consults it iff `returning_tool_calls`
(`pending_tool_calls or proxy_mcp.has_pending_calls()`) is `False` — the same
authoritative source the usage-limit guard already uses. Prior tool blocks
(built-in or resolved proxy) no longer immunise a terminal API error. New
pin-tests A4 (pending proxy call + marker-like text → tools, not error), A5
(built-in tool block + terminal `is_error` → detected), A6 (resolved proxy +
terminal marker+known-zero → detected).

**MAJOR (integration seam not fully specified) — RESOLVED (round 2), then
CORRECTED in round 3.** Original round-2 fix used `PYTHONSTARTUP`, which
CPython consults *only* before an interactive interpreter's first prompt and
therefore would never fire for the non-interactive `deep-worker` entry point.
Replaced with a pytest-owned `sitecustomize.py` on the child's `PYTHONPATH`
prefix (Python's standard on-startup hook, imported by `site.py` before the
entry point runs unless `-S`). Double gate: `PYTHONPATH` prefix + env var
`DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST=1`. A one-shot canary written to a
dedicated `os.pipe()` fd (event-driven `loop.add_reader`, no polling) proves
the shim actually executed before the invoke — a broken bootstrap fails
unambiguously as "seam bootstrap did not fire" rather than as an ambiguous
timeout. Production code has zero imports from `tests/_support/`
(`test_no_prod_import_of_test_support` lint-test enforces it, RED-fails on
regression). Mutation test `test_bypass_detector_reddens` preserved.
Readiness wait is a genuine RPyC connect event (registry surfaces the
process only when idle-ready); notify wait is the existing
`subscribe_messages` event surface. See §5.2.

**REQUIRED nit (a) `classify_usage` negative int — RESOLVED.** New rule:
any usable recognised counter `< 0` → `UNKNOWN` (malformed; the SDK must not
emit negative token counts; treat as unusable, never as zero). Pin-test U10.

**REQUIRED nit (b) §3.1 "model sees and continues" wording — RESOLVED.**
Replaced by "SystemMessage carrying the error is included in the next request
the runtime sends (G4) — whether the model actually processes it depends on
the request succeeding" (consistent with the tightened G4/§3.4).

**REQUIRED nit (c) two independent SDK signals — RESOLVED.** `isApiError
Message` is a transcript-only key, not an SDK field (verified by grep over the
`claude_agent_sdk` source). Detector now consumes the single SDK bool
`result_is_error` from `ResultMessage.is_error`; `result_subtype`,
`result_errors`, `result_text`, `stop_reason` are enrichment-only, used only
to build the message text and NEVER as independent proof of error. §1.2b
added to document this explicitly. Old `is_api_error_flag` parameter
removed. Pin-tests D9 (SDK bool alone with normal text is sufficient), D12
(enrichment fields alone without the SDK bool are NOT sufficient).

### Round 1 (2026-08-04) — resolutions

**BLOCKER (usage=None/{}) — RESOLVED.** `classify_usage` is three-valued
(`KNOWN_ZERO` / `NONZERO` / `UNKNOWN`); rule 3 fires only on `KNOWN_ZERO`.
`None`, `{}`, and mappings with only unrecognised/malformed fields are
`UNKNOWN` and never trigger the text-marker fallback. Pin-tests U1–U9 and
D3–D5 lock the invariant. **Q1 answer accepted:** zero-usage alone is not
sufficient; marker AND known-zero is the conservative fallback.

**MAJOR 1 (error payload for structured-signal rules) — RESOLVED.**
`ClaudeApiResponseError` carries `assistant_error`, `result_subtype`,
`result_errors`, `stop_reason`. `str(exc)` is built via the §3.2 precedence
ladder (marker-carrying text → `result_text` → joined `result_errors` →
synthesised description) — never `""`, never ordinary partial assistant
prose. **Q2 answer accepted:** structured fields present, no failover
introduced now.

**MAJOR 2 (G4/§3.4 over-promising) — RESOLVED.** G4 and §3.4 explicitly
separate *guaranteed* (persistence, inclusion in the next request, visible
persistent error, no retry, no `AIMessage` with the error text) from *not
guaranteed* (that the model actually processes the SystemMessage, or that
the task recovers). **Q3 answer accepted:** classification stays recoverable,
not fatal.

No open questions remain. Requesting sign-off.
