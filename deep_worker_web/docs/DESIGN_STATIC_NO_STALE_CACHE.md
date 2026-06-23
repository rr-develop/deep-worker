# DESIGN: Static assets must never be served stale (Full Dialog introspection bug)

**Status:** 📋 PROPOSED → implementation
**Date:** 2026-06-23

---

## 1. Problem (reproduced, not theorised)

The user reports: in the running web UI (`/path/to/production`, master,
port 7887) selecting the 4th (**System prompt**) or 5th (**Available tools**)
entry of the Full Dialog verbosity combo and clicking **Full Dialog**
invariably shows the **user/agent conversation**, never the system prompt or
the tool list.

The previous agent "fixed" this three times and each time it was false. The
reason: the *source* (`static/index.html`, `static/js/main.js`,
`app/services/agent_service.py`, `app/routers/agents.py`) is in fact correct.
Verified live:

- `curl …/dialog-text?verbosity=system_prompt` → `--- System Prompt …`
- `curl …/dialog-text?verbosity=tools` → `--- Available Tools …`
- a **fresh** headless chromium driving the real UI (select agent → pick
  `system_prompt`/`tools` → click) → correct view for all 6 running agents.

So why does the *user's* browser show conversation? Because the user's browser
runs a **stale cached** `index.html` / `main.js` from before the feature
landed. Reproduced deterministically in
`tests/_repro_stale_cache.py`:

```
stale index.html option count = 3
stale dropdown option values = ['final', 'text', 'full']
after select_option('system_prompt') the value is: 'FAILED(TimeoutError)'
opened-tab first line: '--- Summarization --- You are in the middle of a conversation…'
==> SHOWS CONVERSATION INSTEAD OF SYSTEM PROMPT: True
```

A stale 3-option dropdown cannot emit `system_prompt`/`tools`; the value stays
on a history verbosity and Full Dialog opens the conversation — **exactly the
user's symptom.**

### Root cause

`app/main.py` mounts the static directory with a bare
`StaticFiles(directory=…)`. The served responses carry an `ETag` and
`Last-Modified` but **no `Cache-Control`**. Per RFC 9111 §4.2.2 a response with
no explicit freshness lifetime is subject to *heuristic caching*: the browser
may reuse the cached `main.js` / `index.html` **without revalidating**. The
HTML also references the script with a bare `src="/static/js/main.js"` — no
version query — so there is nothing to bust the cache on deploy.

Net effect: after any UI change, returning users keep running the previous
JS/HTML until they hard-reload. For a localhost operator tool this is pure
downside — there is no CDN, no bandwidth concern, and "the page must reflect
the code that is on disk" is the whole point.

## 2. Goal

A returning browser MUST always run the `index.html` / `main.js` that are
currently on disk. No hard-reload, no incognito, no manual cache clear.

## 3. Non-goals

- No change to any RPyC contract, the `dialog-text` endpoint, the dropdown
  options, or the formatters — those are already correct.
- No fingerprinted-filename build pipeline (overkill for a localhost tool;
  adds a build step we don't have).
- No new threads, no polling, no blocking, no architecture change.

## 4. Design

Introduce a tiny `StaticFiles` subclass that stamps a revalidate-always
`Cache-Control` header on every static response:

```python
class NoStaleStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp
```

`no-cache` does **not** mean "don't cache" — it means "cache, but
**revalidate every time** before reuse". The browser keeps sending its
`If-None-Match` with the existing `ETag`; when the file is unchanged the server
answers `304 Not Modified` (a few bytes), when it changed the server answers
`200` with the new bytes. Cost on localhost is negligible and the operator can
never run stale UI again.

This is mounted in place of the bare `StaticFiles` at `app/main.py`.

### Why not a version query string instead?

A `?v=<mtime>` on the `<script>` tag busts `main.js`, but the *document*
itself (`index.html`) is what carries that query — and `index.html` is exactly
what gets cached stale, so the busting query never updates. Stamping
`Cache-Control` on the document is the only thing that fixes the document.
The subclass covers HTML, JS, CSS uniformly.

## 5. Tests (TDD)

1. **Unit** (`tests/unit/test_static_no_stale_cache.py`, RED first): GET
   `/static/index.html` and `/static/js/main.js` through the FastAPI
   `TestClient` and assert the response carries
   `Cache-Control: no-cache` (and `must-revalidate`). Fails before the fix
   (no header), passes after.
2. **Integration / honest UI** (`tests/test_introspection_bug_repro.py`,
   already GREEN against fixed source): drive the real browser, select agent,
   pick `system_prompt` then `tools`, click Full Dialog, assert the opened tab
   is the introspection view and contains **no** history markers.
3. **Repro artifact** (`tests/_repro_stale_cache.py`): documents the stale
   path that produced the user's symptom (kept as evidence, not a CI gate).

## 6. Acceptance

- Unit test for `Cache-Control: no-cache` on static HTML/JS passes.
- `pytest tests/unit` stays green; no `pyright` regressions.
- Live: re-requesting `/static/js/main.js` after an edit serves the new file
  on the next normal navigation (no hard-reload).
- Live UI: picking `system_prompt`/`tools` and clicking Full Dialog shows the
  introspection view, for a browser that previously cached old assets.
