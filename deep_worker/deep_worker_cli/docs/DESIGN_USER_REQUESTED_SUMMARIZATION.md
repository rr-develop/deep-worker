# User-requested conversation summarization

## 1. Goal

Allow a user to compact an agent's active conversation on demand,
independently of the automatic context-size threshold.

- `deep-worker <agent> --summarize` uses the same default prompt, retention,
  offload, summary shape, and failure policy as automatic compaction.
- `deep-worker <agent> --summarize --query TEXT` and `--query-file PATH` use a
  one-shot custom summarization prompt.
- The web agent dialog exposes default summarization only.
- Existing `--sum-query` / `--sum-query-file` pseudo-summary semantics remain
  unchanged.

## 2. CLI contract and one-shot propagation

`--summarize` is one-shot and never persisted. Without it, `--query` and
`--query-file` retain their current ordinary-turn meaning. With it, the query
text is the custom summary prompt and no ordinary turn runs. `--summarize`
combined with either `--sum-query` form fails with exit 2 before daemonization.
The existing query/query-file and sum-query/sum-query-file mutexes remain.

For a live agent, CLI dispatch calls a dedicated summarize RPC. For start or
restart, argparse resolves `(summarize=True, summary_prompt)` before
daemonization; `run_agent` passes both into the child `AgentProcess`
constructor; the constructor stores `_pending_summarize` and
`_pending_summary_prompt`; and the post-initialization hook copies then clears
both before calling the shared summarize operation. Neither field exists in
`AgentConfig` or launch config. Start, restart, and live dispatch execute once;
a second restart without flags cannot replay the operation.

## 3. One real async compaction pipeline

Refactor the existing async automatic path into
`_arun_compaction(state, runtime=None, *, force, summary_prompt=None,
backend=None, thread_id=None)`, used by
both `abefore_model` and the new public operation. It owns truncation,
triggering, cutoff/partition, offload, prompt formatting, model invocation,
content extraction, and replacement construction. Its only new semantic inputs
are `force` and an optional one-shot `summary_prompt`. The synchronous
`before_model` retains its existing implementation.

Add:

`apply_requested_summary(thread_id, current_messages, summary_prompt=None)`.

It calls the helper with `force=True`; `abefore_model` calls it with
`force=False`. Forced mode bypasses only `_should_summarize`:

1. ensure ids and apply configured tool-argument truncation;
2. determine the normal retention cutoff;
3. when cutoff is not positive, return `summarized=False`; include a
   truncation-only replacement if truncation changed state, otherwise no
   replacement;
4. otherwise partition using the existing retention logic;
5. offload the evicted prefix before model invocation;
6. generate the summary with the common prompt/model/content pipeline;
7. return the full replacement: summary-shape message plus preserved suffix.

The typed result contains `summarized`, optional replacement messages, and
optional `history_path`. It is a pure state producer; checkpoint mutation stays
in `AgentProcess`.

Custom prompts format the conversation exactly once. `{messages}` is replaced
by the shared serialized payload. Without that placeholder, the prompt is
treated as instructions and the payload is appended. Escaped literal braces
`{{` / `}}` are supported by the one formatter; unmatched braces, unknown
fields, repeated `{messages}`, and other invalid formats fail loud before model
invocation. The common helper owns formatting, trimming, model invocation,
content extraction, and errors; there is no ad-hoc `ainvoke` and
`self.summary_prompt` is never mutated.

Because the requested operation runs outside LangGraph `Runtime`, it requires
`self._backend` to be a concrete `BackendProtocol`; a factory fails loud before
offload. Production already supplies the concrete composite backend. The
caller-provided `thread_id` goes directly to `_aoffload_to_backend`; there is no
fallback to `_get_thread_id()` or generated `session_*` path.

The feature adds no `Thread`, `to_thread`, polling, sleep, or blocking-wait
site. It reuses the existing `_aoffload_to_backend` callback, including its
existing `asyncio.to_thread` call.

## 4. AgentProcess and RPyC

`AgentProcess.request_summarization(prompt=None)`:

- holds `_invoke_lock` across state read, offload/model, and checkpoint update;
- reads current messages through `aget_state`;
- calls `apply_requested_summary`;
- calls `aupdate_state` exactly once only when replacement exists, using
  `RemoveMessage(REMOVE_ALL_MESSAGES)` plus the returned messages;
- performs zero checkpoint writes for no-evictable/no-truncation;
- returns serializable `summarized`, `history_path`, and message counts.

A dedicated `exposed_summarize(prompt=None)` keeps real summary, pseudo-summary,
and normal-send semantics separate. Concurrent turns and summaries serialize
on `_invoke_lock`; no polling is added.

## 5. Web API and UI

`POST /api/agents/{agent_name}/summarize` accepts no prompt. It verifies the
agent and delegates to `AgentService.summarize_agent`. The service uses the
non-blocking connection gate and existing strong-reference
`fire_and_forget_async` owner around `root.summarize()` with `timeout=None`.
Disconnect exceptions flow to its callback.

HTTP immediately returns a request id. Exactly one terminal WebSocket event is
emitted with that id: `agent_summarized` or `agent_error`.
`agent_summarized` includes the true `summarized` flag, so `False` is a
completed no-op, not a false compaction claim.

The agent dialog gets a default-only "Summarize context" button, disabled while
the request is accepted and integrated with existing toast/status handling.

## 6. TDD and acceptance

- middleware tests cover forced default summary below threshold, custom prompt,
  retention, truncation-only/no-op, offload-first, model failure, factory
  rejection, and explicit thread path;
- an equivalence test gives identical threshold-triggering input to automatic
  and requested async paths and asserts identical replacement, offloaded
  content, and path;
- CLI matrix covers both mutex pairs, conflicts with both sum-query forms,
  ordinary query without summarize, start, restart, live dispatch, and
  second-restart-no-replay;
- AgentProcess/RPyC tests cover locking, checkpoint write counts, serialization,
  and fail-loud behavior;
- web tests cover request id, retained fire-and-forget task, exactly one
  terminal event, no-op representation, and button behavior;
- a static regression rejects new polling, sleep, `Thread`, and `to_thread`
  sites while allowing the unchanged callback;
- executable integration acceptance launches `deep-worker` with provider
  `codex`, model `gpt-5.6-sol`, the real per-agent SQLite checkpointer, and a
  temporary `HOME`; seeds state below threshold; requests summary through real
  RPyC; and bridges the RPyC completion callback into one `asyncio.Event`,
  awaited under one `asyncio.timeout` deadline. Process-exit cleanup uses
  pidfd/select where available, never sleep/stat polling. `try/finally` stops
  the agent and removes the sandbox. With
  `DEEP_WORKER_REQUIRE_INTEGRATION=1`, missing credentials/provider is failure,
  not skip, and the required gate must pass. It verifies exact summary-shape
  checkpoint, preserved suffix, lossless JSONL, and model-produced content
  rather than pseudo substitution. A negative control invokes the ordinary
  below-threshold graph path over the same seed and proves no compaction
  without forced bypass.

## 7. Compatibility and failure behavior

No launch-config schema field is added. Existing commands, automatic
summarization, and pseudo-summary behavior remain compatible. Offload or model
errors leave the checkpoint untouched. As in existing real compaction, model
failure may happen after a successful pre-model JSONL offload; atomicity is
guaranteed for checkpoint replacement, not across checkpoint and history-file
storage.
