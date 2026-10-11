# DESIGN: Full Dialog is a static, isolated snapshot — never updates, never blocks the main UI

**Status:** ✅ DESIGN APPROVED (review, 2026-08-16) → implementation
**Date:** 2026-08-16
**Scope:** `deep_worker_web` only (port 7887). The alternative dev web UI is out of scope (not used).

---

## 1. Problem (as reported)

> I repeatedly observe the full output window (Full Dialog) freezing in the web UI.
> This is especially noticeable with a large context and can completely hang the
> browser. […] It is caused by the window trying to update itself when new
> messages arrive. […] The window must output the messages as of the moment of
> the call (when the user clicked the button). IT MUST NOT TRY TO UPDATE
> ANYTHING!!!!! This leads to a complete system freeze.

Required contract (owner's words, verbatim intent):

* **C1** — Full Dialog is a **snapshot taken at the moment the button is
  clicked**.
* **C2** — The opened window **never updates itself** (no push, no poll, no
  reload, no scripts).
* **C3** — Opening / reading Full Dialog **must not be able to hang the
  browser or the main UI**, regardless of context size.

## 2. Investigation — what the code actually does today (verified, not theorised)

Every claim below was checked against the source in the development tree
(identical to the served tree, both at `b164482`) and
against the **live** server on 127.0.0.1:7887 on 2026-08-16.

| # | Fact | Evidence |
|---|------|----------|
| F1 | The button opens `window.open('/api/agents/<name>/dialog-text?verbosity=…', '_blank')`. | `static/js/main.js:418-424` |
| F2 | The endpoint is a `PlainTextResponse` with `Content-Length` — one static `text/plain` document, no streaming, no HTML, **no JavaScript**. | `app/routers/agents.py:628-698`; `curl -I` → `content-type: text/plain; content-length: N` |
| F3 | The opened tab has **0 scripts, 0 network requests after load, content byte-identical for 20 s** while agents keep producing messages; both tabs answer JS in ~2 ms. | Real headless Chromium via Playwright driving the real UI: `select agent → click #full-dialog-btn` (a local repro script) |
| F4 | The main page never appends to the DOM on incoming messages — `handleAgentMessage` writes `textContent` of one element; `handleAgentResponse` only reloads Last Response. There is no history panel that re-renders. | `static/js/main.js:998-1094`, `477-499` |
| F5 | The web side awaits the agent **non-blocking**: `exposed_get_history` is `async def` ⇒ netref call → `asyncreq` → awaitable; wrapped in `wait_for(…, 30 s)`. `/health` p50 = 0.8 ms *while* Full Dialog requests run. | `agent_service.py:1904-1951`, `rpyc_async/core/netref.py:289`; live probe |
| F6 | Payload is small: across all 15 running agents `verbosity=full` is 28 B … 255 KB, longest line 1 547 chars, served in 5–140 ms; 360 samples over 3 min: max 0.14 s, no errors except two agents with a corrupt DB / not connected. | a local latency log |
| F7 | Even a synthetic 20 MB `text/plain` popup loads in 2.5 s and stalls the opener tab ≤ 0.6 s in Chromium. | a local benchmark script |
| F8 | `static/dialog.html` (a JS page with `fetch` + manual *Refresh* button) is **dead code**: nothing links to it since `679daa7` (2026-04-06). It has no timer either. | `grep -rn dialog.html static/` → only a CSS comment in `stats.html` |
| F9 | Web log (2026-08-04 → 08-15): every operator `dialog-text` request returned `200` (one `503 connecting…` right after the agent's RPyC link died and it was restarted). | the web server log |

**Conclusion of the investigation.** The self-updating mechanism named in the
report does **not exist** in the current code: F1–F4 show the tab is a static
document with no script, and F3 demonstrates it empirically in a real browser.
I could not reproduce any hang. What *does* remain — and is what this design
fixes — is that the contract C1–C3 is currently **implicit and unenforced**:

* **G1 — No isolation.** `window.open(url, '_blank')` without `noopener`
  keeps an opener relationship. Same-site popups then share the opener's
  browsing-context group and (in Chromium) its renderer process, so anything
  heavy in one tab can freeze the other. This is the *only* code-level
  channel through which "the Full Dialog tab" and "the whole browser/main UI"
  are coupled, and it is exactly the coupling the report describes.
* **G2 — No freshness contract.** The response carries no `Cache-Control`.
  Nothing forbids a browser from serving a previously fetched dialog from
  cache / bfcache, i.e. showing something *other* than the click-time
  snapshot.
* **G3 — Not self-evidently a snapshot.** The document has no timestamp, so
  the operator cannot see *when* it was taken nor tell whether it changed;
  a stale or slow tab is indistinguishable from an "updating" one.
* **G4 — No regression guard.** Nothing in the test suite pins "no scripts,
  no opener, no cache, static body". A future "live view" could be added
  silently — the report shows that risk is unacceptable to the owner.
* **G5 — Dead `dialog.html`** contradicts the contract (a JS-driven dialog
  page with fetch logic) and invites accidental reuse.
* **G6 — Slow agent = spinning tab.** If the agent's loop is busy (big
  checkpoint write, summarization/re-index, wedged turn) `get_history`
  waits up to 30 s; the tab shows a spinner then a 500 — visually a "hang".
  Bounded today; kept bounded, but the operator must see *why* (see §3.4).

## 3. Design

### 3.1 Process/context isolation of the popup (G1)

`static/js/main.js` `openFullDialog()`:

```js
window.open(url, '_blank', 'noopener,noreferrer');
```

`noopener` severs `window.opener`; per HTML spec the new top-level browsing
context is created in a **new browsing-context group** ⇒ Chromium/Firefox
place it in a separate renderer process. A heavy Full Dialog tab can no
longer stall the main page (WebSocket, buttons) and vice-versa. The return
value (now `null`) is unused today. Same change for `openAgentStats()` for
consistency (it is the same pattern).

### 3.2 Explicit snapshot freshness (G2)

`app/routers/agents.py` `get_dialog_text`: return a `PlainTextResponse` with
headers

```
Cache-Control: no-store
X-Content-Type-Options: nosniff
```

`no-store` = each click fetches a fresh snapshot; a reload of the tab (F5)
re-fetches — that is a *user* action, still C1-compliant. `nosniff` pins
`text/plain` rendering (no HTML sniffing → no script execution ever).

### 3.3 Self-describing snapshot banner (G3)

The router prefixes the body (all five verbosity modes) with two lines:

```
=== Full Dialog snapshot: agent=<name>  verbosity=<v>  taken=<YYYY-MM-DDTHH:MM:SS+ZZ:ZZ> ===
=== Static snapshot — this page never updates. Click "Full Dialog" again for a fresh one. ===
<blank line>
<existing rendered text, unchanged>
```

Timestamp is the web server's local time with UTC offset (same clock the
operator sees in agent uptime). The rendered body itself (formatter output,
`--- User ---` blocks, introspection views) is **not** modified — all
existing formatter tests remain valid; e2e tests use `in` matching.

### 3.4 Bounded wait stays bounded, but is visible (G6)

No timeout change (30 s, `agent_service.py:1910`). Two visibility tweaks:

* On `asyncio.TimeoutError` the router answers **504** with body
  `Agent '<name>' did not answer get_history within 30s (agent busy or
  wedged). The main UI is unaffected; retry later.` — the tab shows a clear
  reason instead of a generic 500.
* `logger.warning` with the elapsed time when a Full Dialog fetch takes
  > 5 s (bounded breadcrumb, no content), so slow-agent episodes are
  diagnosable from `deep_worker_web.log` after the fact.

### 3.5 Remove dead `static/dialog.html` (G5)

Delete the file. Its only reference is a comment in `stats.html` (updated).
No route serves it; no test imports it (verified by grep).

### 3.6 Non-goals

* No pagination / truncation of the dialog (contradicts the "no artificial
  limits" policy of `DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md` §6).
* No change to the agent-side `_handle_get_history` or the formatter.
* No streaming / chunked response (would *create* the "updating" impression).

## 4. Tests (TDD — RED first, then GREEN)

1. **`tests/unit/test_full_dialog_static_snapshot.py`** (new, FastAPI
   `TestClient` + stubbed `get_agent_dialog_formatted`, same fixtures as
   `test_api_agents.py::TestDialogTextEndpoint`):
   * `Cache-Control: no-store` and `X-Content-Type-Options: nosniff` present;
     `Content-Length` present (non-streaming); `text/plain; charset=utf-8`.
   * Body starts with the two banner lines; line 1 contains `agent=<name>`,
     `verbosity=<v>` and an ISO-8601 timestamp with offset; the stubbed
     text follows verbatim after one blank line — for **all five** verbosity
     values.
   * Two consecutive requests produce two different `taken=` stamps when
     the clock is monkeypatched (proves the stamp is per-request).
   * `asyncio.TimeoutError` from the service → **504** and the message names
     the agent and the 30 s bound.
   * Slow path (> 5 s, clock monkeypatched) emits one `WARNING` record.
2. **`tests/unit/test_frontend_full_dialog_static.py`** (new, static guard,
   same style as `tests/integration/test_no_blocking_rpyc_in_handlers.py`):
   reads `static/js/main.js` and asserts
   * `openFullDialog` **and** `openAgentStats` call
     `window.open(<url>, '_blank', 'noopener,noreferrer')`;
   * no `setInterval` / `setTimeout` / `WebSocket` / `EventSource` /
     `fetch(` inside the `openFullDialog` function body;
   * **post-conditions of step 4 (deletion)** — the file
     `static/dialog.html` no longer exists and the string `dialog.html`
     appears nowhere under `static/` (today the file exists since `6df8885`
     and `stats.html:9` mentions it in a CSS comment; that is the RED state).
3. **`tests/integration/test_full_dialog_tab_is_static.py`** (Playwright,
   marked `ui`, honest, against a running server; ports the repro of F3):
   open main page → select agent → click button → in the new tab assert
   `document.scripts.length == 0`, `window.opener === null`, **zero**
   network requests after `load`, `document.body.innerText` byte-identical
   across 3 samples spanning ≥ 6 s, first line matches the banner regex,
   opener tab answers `1+1` in < 100 ms during the whole window.
   Bounded: hard 60 s budget, guaranteed browser close in `finally`
   (per the bounded-integration-tests rule).

RED expectations before implementation: (1) header/banner/504 asserts fail;
(2) `noopener` asserts fail, `static/dialog.html` still exists and is
referenced from `stats.html:9`; (3) `window.opener` is non-null.

## 5. Implementation steps

1. Write tests 1–3, run, confirm RED (record the exact failures).
2. `main.js`: `noopener,noreferrer` in `openFullDialog` / `openAgentStats`.
3. `agents.py`: build `PlainTextResponse(banner + text, headers=…)`; 504 on
   `asyncio.TimeoutError`; > 5 s warning.
4. Delete `static/dialog.html`; fix the comment in `stats.html:9`; update
   `IMPLEMENTATION_STATUS.md:125` (it still lists `/static/dialog.html` as a
   complete feature).
5. GREEN; `pytest deep_worker_web/tests/unit` full run; `pyright` on touched
   files; live check with a fresh headless Chromium (F3 script) after
   restarting the web server (`deep_worker_web` serves static from disk, but
   the router change needs a restart).
6. Send delta for code review; commit on `dev` after sign-off;
   FF `master`.

## 6. Acceptance criteria

* AC1 — Opened tab: `window.opener === null`, 0 scripts, 0 requests after
  load, body constant (test 3).
* AC2 — Response headers `Cache-Control: no-store`,
  `X-Content-Type-Options: nosniff`, `Content-Length` (test 1).
* AC3 — First line shows agent, verbosity, click-time timestamp; second
  line states the page never updates (test 1).
* AC4 — Slow/wedged agent → 504 with an explicit reason within 30 s; main UI
  `/health` stays < 50 ms meanwhile (test 1 + live probe).
* AC5 — `static/dialog.html` gone; static guard green (test 2).
* AC6 — `tests/unit` fully green, no pyright regressions.

## 7. Risk & rollback

Pure additive/defensive changes; body content unchanged apart from a 2-line
prefix. Rollback = revert the single commit. If any external consumer parses
`dialog-text` as raw formatter output (none found: only the browser and the
tests use it), the banner can be gated behind `?banner=0` — not planned
unless review asks for it.

**Known residual risk (raised in review, kept out of scope):** the rendering
step `split_into_turns` + `render_dialog` (`dialog_formatter.py:203,378`) is
synchronous CPU work executed on the web event loop
(`agent_service.py:1946-1947`) without `run_in_executor`. At today's sizes
(≤ 255 KB) it costs milliseconds; because the policy is "no artificial
limit", a multi-megabyte dialog would block the loop (and `/health`) for the
render duration. This is *not* the reported mechanism and does not affect
C1–C3. Decision: the render **stays on the loop** in this change; the > 5 s
warning of §3.4 makes such episodes visible. Option for a follow-up: move
`split_into_turns`/`render_dialog` into `loop.run_in_executor(None, …)` —
they are pure functions of the payload, so the move is mechanical.

## 8. Open point for the reviewer / owner — RESOLVED (review, 2026-08-16)

The reported *mechanism* (self-refresh on new messages) is not present in
the code and the hang was not reproducible (F1–F9). This design therefore
enforces the contract and removes the one real coupling (G1) plus the
visibility gaps (G2–G6). If the owner can name the browser, agent and
verbosity of a concrete hang episode, a targeted repro will be added to
test 3.

Review verdict: **DESIGN APPROVED.** Not over-scope — the banner (G3), the
504-on-timeout (G6; today `TimeoutError` falls into `except Exception` →
500) and the deletion of dead `dialog.html` (G5) all serve C1–C3 directly,
are additive and leave the formatter untouched. The one additional candidate
mechanism raised (sync render on the loop) is recorded in §7.
