# The "context was compacted mid-turn" event

Status: implemented, including the standing key of §5.1 (decision **B**). The
receiving side is a downstream consumer project.
Format agreed on: 2026-10-09 (UTC).

## 1. Why

The consumer's requirement:

> Summarization during document processing is an incident, not a normal mode.

The consumer cannot detect this on its own: only deep_worker knows about
summarization. An interrupted turn reaches it **looking like an ordinary
successful one**: `success: true`, the channel is intact, nothing was thrown.
Accepting it as an ordinary turn means putting an under-extracted document into
the graph marked "done", and nobody will ever come back to it.

The cost of a mistake is asymmetric, and this is the rationale for the third
state (§5): failing a good document needlessly costs one repeated extraction;
accepting a compacted one needlessly costs a silent hole in the graph.

## 2. What counts as the event and what does not

The event is **only a genuine size-triggered summarization that fired during a
turn**: the path `abefore_model` → `_arun_compaction(force=False)` with
`should_summarize == True` (`deepagents/middleware/summarization.py`).

NOT an event, and this is essential:

| Path | Why it is not an event |
|---|---|
| `apply_pseudo_summary` (`summarization.py:1277`, `verbatim=True`) | Pseudo-summarization is `--sum-query`, a context reset before every document. This is the consumer's **normal mode** and the first half of fulfilling REQ-90. The model is not called, the size is not checked. A flag here would fail **every document of every run**, the main risk of this change, as flagged by the consumer's maintainers. |
| `apply_requested_summary` (`force=True`) | An explicit operator command (`exposed_summarize` / `compact_history`), not a size-triggered incident. It has its own report, `agent_process.py` `request_summarization`. |
| Argument truncation (`_truncate_args`) | Rewrites the message channel but summarizes nothing. Important: it returns **exactly the same shape** (`RemoveMessage(REMOVE_ALL_MESSAGES)` + messages), so the return shape is unusable as a detector; only the `summarized` flag distinguishes it. |

The last row implies where the stamp must be placed: `_arun_compaction` is the
**shared body** for the size-triggered path and for `force=True`. Therefore the
stamp lives in `abefore_model`, not inside the shared body. Moving it into the
shared body is exactly the regression that `test_requested_summary_is_not_an_incident`
pins down.

## 3. Format (agreed with the consumer)

The flag travels at the **top level** of the `invoke` reply, next to `success` /
`recovered`, and NOT inside `response`, whose `content` the consumer reads as
text:

```json
{
  "success": true,
  "context_compacted": true,
  "context_summarized": {
    "occurred": true,
    "observable": true,
    "reason": "size_trigger",
    "trigger": "fraction:0.85",
    "tokens_before": 182340,
    "tokens_after": 24180,
    "messages_before": 412,
    "messages_after": 9,
    "history_path": "<agent_home>/summaries/<agent>.jsonl",
    "at_utc": "2026-10-09T12:34:56.789Z"
  },
  "response": {"role": "assistant", "content": "DONE"}
}
```

Rules:

* **Absence of the key = "no summarization happened".** Only an explicit `true`
  counts. A report on every turn would fail every document at rollout.
* `at_utc` is UTC, ISO-8601 with `Z`. A hard rule of both projects: the consumer
  matches the timestamp against its own run timeline, and local time would shift
  the incident by the host's offset.
* The reply text **is not a contract**: matching on wording breaks on the first
  change to it (`recovered` is read the same way).
* Both spellings (the boolean `context_compacted` + the `context_summarized`
  report) are set together. The consumer reads three spellings at two levels, an
  intentional redundancy so that a coordinated release of the two repositories is
  not required.
* Separately from them travels the standing `context_compaction_observable: false`,
  a property of the provider, not a fact about this turn (§5.1). It is NOT
  per-turn and NOT an incident.

Backward compatibility is two-sided: a sender on old code will not send the key,
a receiver on old code will ignore it. Neither half breaks.

### The consumer's reaction (for reference)

The document is marked as failed with the reason "summarization mid-document", a
WARNING `context_compacted_mid_document` goes to the log, and the incident has its
own field in the run report. The failure is **transient**: the document goes to a
retry rather than being closed as `.error`. What was done is not undone: partial
fragments and earlier nodes remain.

## 4. Delivery: why deterministic and without polling

`abefore_model` is a LangGraph hook; its return value goes into the graph state,
not into the turn's reply. Therefore the flag cannot simply be returned from the
hook.

Wiring (no new channels were added):

1. `SummarizationMiddleware.last_turn_compaction` is a field on the instance, set
   in `abefore_model` when `result.summarized`.
2. `agent_process._reset_turn_compaction_report()` runs at the start of
   `_handle_invoke_owned`, **before any possible return**, so that a turn does not
   inherit the previous turn's incident.
3. `agent_process._with_compaction_flag()` is a wrapper at **all three** successful
   reply sites (ordinary, streaming, TaskManager). A flag that arrived at only one
   site of three would produce an incident that reproduces every other time.
4. The middleware instance is obtained through the existing
   `_find_summarization_middleware`, the same mechanism pseudo-summarization
   already uses.

There is no polling, no `sleep`, no waiting outside the turn: the flag is put into
the same dict the consumer already reads as the turn's result.

The absence of the middleware (summarization disabled) is not an error: the
wrapper is soft, and the turn does not fail because there is no incident.
`_find_summarization_middleware` stays fail-loud where pseudo-summarization needs
it.

## 5. The third state: unobservable external compaction

A provider may compact the context **by itself**, without our knowledge. In that
case deep_worker can neither confirm nor refute the event, and this is **not**
"it did not happen".

Established from the protocol, not from the absence of search results:

* **ACP**: `HANDLED_SESSION_UPDATE_KINDS` (`providers/acp/acp_types.py:22-32`) has
  no "context compacted" kind; `usage_update` reports the size, not the fact of
  compaction; unknown kinds are logged and ignored (`:20-21`). `TERMINAL_STOP_REASONS`
  (`:61-69`) also has nothing about compaction.
* **codex**: `providers/codex_provider/chat_model.py:789-793`: the server-side
  `thread/compact/start` exists in the protocol, but the provider runs
  `codex exec` and does not use it.
* **claude-code / claude-code-sdk**: no mentions of compaction at all.

Therefore the `observable` field **inside the report** is always `true`: the report
is only ever created on the observable path, and both stamping sites hardcode
`True` (`summarization.py:1488`, `:1883`). The value `false` never occurs there;
the field is kept as a reserve in case a provider one day learns to report its own
compaction.

**Standing** unobservability is signaled by a separate top-level key,
`context_compaction_observable: false` (§5.1), and it does **not** fail the
document: a loud line is written, and there is no extraction retry. It must not be
confused with the per-turn report: treating the standing flag as an incident is
exactly what would produce permanent downtime instead of a safety margin.

### 5.1 The standing key `context_compaction_observable` (decision B)

Decided by the consumer's maintainers (option **B**): deep_worker sends a separate
top-level key on providers where compaction is unobservable in principle.

```json
{"success": true, "context_compaction_observable": false, "response": {…}}
```

The argument that reverses the priority: all three of the consumer's extraction
agents run on `codex`, per each one's `session_metadata.json`. Unobservability is
not a hypothetical edge but **the current configuration of all three**. Without
the key, the REQ-90 detector is blind exactly where it is uniquely needed, and yet
reports "no summarization happened", the very silence the requirement was written
against.

**Two keys, two meanings, and neither side may confuse them:**

| Key | Meaning | Consumer's reaction |
|---|---|---|
| `context_compacted` + `context_summarized` | A FACT about this turn: a genuine size-triggered summarization fired | incident, document goes to retry |
| `context_compaction_observable: false` | A STANDING property of the provider, sent on every turn | loud line `context_compaction_not_observable`, document is NOT failed |

That is why the standing key is deliberately **not** expressed as
`context_summarized: {observable: false}`: that shape is a per-turn report, and
sent on every turn it would produce not a safety margin but permanent downtime
(every document of every run would fail). The per-turn `observable` inside the
report describes **an event we saw**; the standing key describes **the provider**.
Different subjects, both values are true simultaneously; see
`test_real_compaction_on_an_unobservable_provider_still_reports_the_incident`.

**Where the answer comes from.** Not from a new list of provider names, but from
`hot_cache.SESSION_STATEFUL_PROVIDERS`, the single definition of the classification
"history lives on the provider's side". That property **is** the unobservability
itself: keeps history on its side ⇒ may compact it without our noticing. A second
handwritten copy of the list would drift from the first when a provider is added,
silently and toward claiming an observability that does not exist
(`test_observability_uses_the_single_source_of_truth`).

There is currently no capability port in the tree that could answer "I compact
silently": `SummarizerPort` (`context_limit/summarizer_port.py:71`) declares only
`asummarize`/`areset_after_summary`, and no declaration method such as `supports_*`
exists. So the choice was not "string vs capability" but "string vs nothing".

**The contract is the behavior of the key, NOT how it is computed.** The consumer
may rely on the key arriving where compaction is unobservable and not arriving
where it is observable; what deep_worker computes this from is an internal detail,
replaceable without a coordinated release. Therefore the reliance on
`SESSION_STATEFUL_PROVIDERS` should NOT be recorded in the consumer's manifest:
otherwise the appearance of a capability port (a one-line replacement here) would
require editing someone else's manifest, turning an implementation detail into a
cross-service interface.

An unrecognized provider identity (`None`, an empty string, an unfamiliar name) is
**not** grounds to claim a blind spot: there is no key. The same logic as
`_is_session_stateful`, where the answer to "isn't it stateless?" made every
unknown provider and every `MagicMock` session-stateful.

Residual limitation: if the CLI compacts the context on its own, the exact moment
still will not arrive; this can only be closed by a signal from the CLI itself,
which the protocols lack (§5). The key tells the consumer exactly this: "it cannot
be confirmed or refuted here", not "the event did not happen".

## 6. Tests

`deep_worker_cli/tests/unit_tests/test_midturn_compaction_report.py` has 26 tests
on **real** classes (`SummarizationMiddleware`, `AgentProcess`), with no
homemade fakes of the production shape. A full run of `tests/unit_tests`: 4604
passed, 6 skipped.

The pseudo path is verified by **execution**, not by reading the source: the test
calls the real `apply_pseudo_summary` together with the real `_aoffload_to_backend`
(only the backend is replaced, and only the async methods the helper actually
calls, `adownload_files` / `awrite`). This is essential: the offload helper is
**shared** with the size-triggered path, so a stamp placed in it would also mark
pseudo-summarization, and a test with a replaced helper cannot see that.

Proven by mutations (each goes red for a substantive reason: an `AssertionError`
on the subject under test, not a `TypeError` on a signature):

| Mutation | Fails |
|---|---|
| Remove the stamp in `abefore_model` | `test_size_trigger_during_turn_is_recorded` |
| `_with_compaction_flag` → no-op | `test_flag_rides_at_top_level_not_inside_response` |
| Move the stamp into the shared `_arun_compaction` | `test_requested_summary_is_not_an_incident` |
| Remove the flag from one site of three | `test_every_success_reply_site_is_stamped` |
| Stamp on the fact of history eviction in the shared `_aoffload_to_backend` (instead of the `summarized` flag) | `test_pseudo_summary_leaves_no_report`, `test_pseudo_summary_reply_carries_no_flag` |

The last mutation is exactly the failure the consumer's maintainers required us to
avoid: a false flag on the normal context reset. The assert shows an actual
`context_compacted: True` in a reply where it must not be.

### Standing-key mutations (§5.1)

The consumer's maintainers' caution in the opposite direction: a "the flag was
set" test passes even if the flag is set ALWAYS. This is covered separately; each
mutation goes red for a substantive reason:

| Mutation | Fails |
|---|---|
| Set `context_compaction_observable` **always** (without checking the provider) | `test_standing_key_is_absent_on_observable_providers`, `test_unknown_provider_does_not_claim_a_blind_spot` |
| Send the standing key in the shape `context_summarized: {observable: false}` | `test_standing_key_is_never_shaped_as_a_per_turn_report` + all 5 parametrizations |
| Leak the per-turn report into the shared `_aoffload_to_backend` on codex + `--sum-query` | `test_pseudo_summary_on_codex_sends_only_the_standing_key` (and the two earlier ones) |
| Remove the standing key entirely | 8 tests, including all 5 providers |

The second row is exactly the error the consumer's maintainers asked us to guard
against: that shape is routed by the receiver into "fail the document", and on
every turn it produces permanent downtime. The third is the costliest possible
outcome of the change: the consumer's agents run on codex AND use `--sum-query`
for every document.
