# Design: Grok Imagine tools via built-in passthrough (subscription-only)

**Status:** IMPLEMENTED (Phase 1: GenerateImage only; image_edit/video blocked pending Gate 0)
**Author:** test_grok / user request
**Date:** 2026-06-19
**Companion:** DESIGN_GROK_PROVIDER.md §11A (channel B),
RESOLUTION_Q1_DISALLOWED_TOOLS.md,
Grok skill ~/.grok/skills/imagine/SKILL.md
**Explicitly out of scope:** xAI API wrappers (api.x.ai, per-token billing). User
requires **SuperGrok subscription only** (~/.grok/auth.json → ACP cached_token).

---

## 0. Goal

Enable deep-worker agents with `--provider grok` to use Grok Build **Imagine**
image generation **without a separate API key**, by extending the existing
**channel B built-in passthrough**
(builtin_tool_passthrough.py).

The model calls the ACP built-in `GenerateImage` **inside the Grok CLI process**
(on subscription quota). Deep-worker **records** the call and result in LangGraph
history — it does **not** re-execute generation.

```
User → deep-worker → GrokChatModel → grok agent stdio (ACP)
                              ↑                    ↓
                    LangGraph ToolNode      Grok runs image_gen locally
                    (passthrough awaits)    → tool_call_update completed
                              ↑                    ↓
                    ToolMessage(path/meta) ← Future resolved
```

---

## 1. Motivation

| Need | Today | After this design |
|------|-------|-------------------|
| Generate images on subscription | Only in Grok Build TUI / harness GenerateImage | GenerateImage round-trips through LangGraph |
| Edit images | No ACP built-in exposed by Grok 0.2.54 | Blocked pending CLI support |
| Short video (optional) | /imagine-video slash command in TUI | image_to_video passthrough **if** CLI exposes it in agent stdio |
| Pay per API token | Not desired | **Not used** |

Codex explicitly **disables** image_generation via --disable. Grok cannot
disable built-ins (Q1), so the correct integration path is **observe + passthrough**,
not proxy MCP and not API duplication.

---

## 2. Hard constraints (inherited from DESIGN_GROK_PROVIDER §0)

1. Every tool call and result **must** appear in result["messages"] (LangGraph).
2. **No polling** — passthrough body awaits asyncio.Future resolved by ACP reader.
3. Passthrough tools **must not** be published to proxy MCP (bind_tools filter stays).
4. Auth stays **cached_token** — no XAI_API_KEY requirement.
5. **No code changes until Gate 0 (live verification) passes** — see §4.

---

## 3. Candidate tool names (from Grok Build)

Sources: ~/.grok/skills/imagine/SKILL.md, ~/.grok/docs/user-guide/04-slash-commands.md.

| Tool | Purpose | Phase |
|------|---------|-------|
| image_gen | Text → image | **1 (required)** |
| image_edit | Image + prompt → image | **1 (required)** |
| image_to_video | Image → short video clip | **2 (conditional)** |
| reference_to_video | Multi-reference video | **2 (conditional)** |

**Not in scope unless live-verified in agent stdio:**

- Text-to-speech / speech-to-text — ACP promptCapabilities reports
  audio: false (grok 0.2.54). No built-in names documented for TTS/STT in
  Imagine skill. Revisit only if a future Grok CLI exposes them as tool_call
  titles on the ACP stream.

Slash commands (/imagine-video) are **TUI affordances**, not guaranteed to
map 1:1 to ACP tool_call.title values — Gate 0 must capture exact titles.

---

## 4. Gate 0 — live verification (BLOCKER)

**Do not implement until each probe below is run on the target host and recorded
in docs/RESOLUTION_GROK_IMAGINE_LIVE_CHECK.md (new file, created by the probe).**

### 4.1 Probes

For each candidate tool, run an isolated grok agent stdio session (same pattern
as Q1 scripts in RESOLUTION_Q1_DISALLOWED_TOOLS.md):

1. session/new with cwd = agent working dir, optional HTTP MCP (can be empty).
2. session/prompt asking Grok to use **exactly one** named tool with a trivial
   prompt (e.g. image_gen: "a red circle on white background").
3. Capture **all** session/update notifications until session/prompt result.

Record per tool:

| Field | Why |
|-------|-----|
| tool_call.title | Must match KNOWN_GROK_BUILTINS entry |
| tool_call.rawInput keys | Passthrough forwards verbatim |
| tool_call_update.content shape | Text only vs image blocks vs file paths |
| tool_call_update.rawOutput | Paths, URLs, duration_ms, error codes |
| Wall-clock duration | Informs bounded-wait policy (§8) |
| Subscription quota errors | User-visible failure modes |

### 4.2 Pass criteria

- **Phase 1 ships** only if image_gen **and** image_edit both emit standard
  tool_call / tool_call_update pairs on the ACP stream.
- **Phase 2 ships** only if at least one video tool passes the same test.
- If tools exist only in interactive TUI (grok without agent stdio) → design
  is **blocked** until xAI exposes them on ACP; document failure, do not ship.

### 4.3 Known risk: promptCapabilities.image: false

Handshake on grok 0.2.54 reports:

```json
"promptCapabilities": {"image": false, "audio": false, "embeddedContext": true}
```

This means **prompt blocks** cannot carry inline images — it does **not** prove
that **tool** image_gen is absent. Gate 0 resolves the ambiguity.

---

## 5. Architecture — channel B extension only

No new proxy MCP tools. No tools.py HTTP wrappers. Changes are confined to:

```
grok_provider/
  acp_types.py              # extend KNOWN_GROK_BUILTINS (+ IMAGINE subset constant)
  builtin_tool_passthrough.py  # optional per-tool descriptions
  chat_model.py             # routing, payload rendering, tool contract hints
  tests/                    # unit + gated integration

deep_worker/deep_worker_cli/
  (optional) default_agent_prompt.md  # mention Imagine passthrough for grok provider
  (optional) bounded_tool_call.py     # per-tool wait override — see §8
```

agent_process.py already calls build_passthrough_tools() for provider == "grok".
**No change required there** once new names are in KNOWN_GROK_BUILTINS.

---

## 6. Implementation plan

### PR-1 — Generic passthrough fix (prerequisite)

**Problem:** Design §11A promised unknown built-ins route through GrokBuiltin, but
_map_tool_call surfaces the raw title (e.g. image_gen). ToolNode only registers
GrokBuiltin as the wildcard — **name mismatch** → turn breaks before Imagine work.

**Fix in chat_model.py _map_tool_call:**

```python
title = upd.get("title")
if title in KNOWN_GROK_BUILTINS:
    surfaced_name = title
elif title:  # unknown built-in
    logger.warning("grok_provider: unknown built-in %r — routing via GrokBuiltin", title)
    surfaced_name = "GrokBuiltin"
    raw_input = {**raw_input, "_grok_original_title": title}
else:
    surfaced_name = "GrokBuiltin"
```

Store _grok_original_title in grok_meta on the resulting ToolMessage for UI/debug.

**Tests:** extend test_chat_model.py + test_builtin_passthrough.py — unknown
title image_gen (when not yet in KNOWN) must invoke GrokBuiltin body and resolve.

### PR-2 — Register Imagine built-ins (after Gate 0)

Add verified names to KNOWN_GROK_BUILTINS in acp_types.py:

```python
IMAGINE_GROK_BUILTINS: frozenset[str] = frozenset({
    "image_gen",
    "image_edit",
    # Phase 2:
    # "image_to_video",
    # "reference_to_video",
})
KNOWN_GROK_BUILTINS = frozenset({...}) | IMAGINE_GROK_BUILTINS
```

Each gets a passthrough BaseTool via existing _make_tool(name).

**Do not** add Imagine tools to _BUILTIN_EQUIVALENTS — there is no deep-worker MCP
equivalent to prefer; Grok built-in is the **only** subscription path.

### PR-3 — Richer ToolMessage payload for media results

Today _render_completed_payload flattens content to text. Gate 0 may show:

- Text path: /home/.../outputs/abc.png
- Structured rawOutput: {path, mime, width, height, ...}
- Content blocks with type: image (if ACP sends them)

**Extend rendering:**

1. Parse rawOutput for path / file_path / url keys (exact keys from Gate 0).
2. If path exists and file is readable → attach to grok_meta:
   {artifact_path, mime, kind: "image"|"video"}.
3. ToolMessage content remains **human-readable text** (path + short summary) for
   history/TUI; optionally add LangChain **multimodal** content blocks if
   read_image / web UI can consume them (follow-up — see §9).
4. Never silently drop paths — user must be able to open the artifact from history.

### PR-4 — Tool contract / agent guidance

Extend _build_tool_contract (or a separate static block for grok provider) when
provider == "grok":

```
Imagine (subscription, Grok built-ins):
- image_gen / image_edit are available as Grok built-ins when you need visuals.
- Do NOT search for their schema on disk.
- For charts/diagrams with exact numbers, prefer code (HTML/SVG), not image_gen.
```

Load imagine skill in Grok Build is **not** automatic in deep-worker — optionally
symlink or copy skill into agent's skills dir (enable_skills=True) for prompt-quality
only; passthrough does not depend on the skill file.

### PR-5 — Bounded wait policy for slow Imagine tools

Passthrough tools run under BoundedToolCallMiddleware (default **120s**). Image/video
generation often exceeds 120s → same failure mode as test_grok Shell incident.

**Option A (preferred):** per-tool override in BoundedToolCallMiddleware:

```python
IMAGINE_TOOL_WAIT_SECONDS = 600.0  # 10 min; tune from Gate 0 timings
TOOL_WAIT_OVERRIDES = {
    "image_gen": IMAGINE_TOOL_WAIT_SECONDS,
    "image_edit": IMAGINE_TOOL_WAIT_SECONDS,
    "image_to_video": IMAGINE_TOOL_WAIT_SECONDS,
    "reference_to_video": IMAGINE_TOOL_WAIT_SECONDS,
    "GrokBuiltin": IMAGINE_TOOL_WAIT_SECONDS,  # when original title is Imagine
}
```

Detect Imagine-via-GrokBuiltin using _grok_original_title in args.

**Option B:** env DW_IMAGINE_WAIT_SECONDS parallel to DW_LONG_RUNNING_WAIT_SECONDS.

**Never** cancel the passthrough Future on timeout — existing shield + job-handle
policy applies; grok keeps generating on its side.

### PR-6 — Tests

| Test | Type | What |
|------|------|------|
| test_unknown_builtin_routes_to_grok_builtin | unit | title=image_gen (unlisted) → name GrokBuiltin |
| test_imagine_tools_in_known_set | unit | after PR-2, passthrough tool count |
| test_render_imagine_payload_with_path | unit | mock ACP update → grok_meta.artifact_path |
| test_imagine_passthrough_round_trip | integration GROK_INTEGRATION_TESTS=1 | live image_gen trivial prompt, file exists on disk |
| test_imagine_does_not_publish_to_proxy_mcp | unit | bind_tools metadata excludes image_gen |

Mirror test_builtin_passthrough_round_trip (Shell) from test_integration.py.

### PR-7 — Documentation

- This file → status IMPLEMENTED after merge.
- RESOLUTION_GROK_IMAGINE_LIVE_CHECK.md — Gate 0 results.
- DESIGN_GROK_PROVIDER.md §11A — add Imagine names to built-in table.
- scripts/run_integration_tests.py — optional grok_imagine_integration suite.

---

## 7. What we deliberately do NOT build

| Approach | Reason excluded |
|----------|-----------------|
| xAI API grok-imagine-* in tools.py | User refuses separate API billing |
| Proxy MCP wrapper that calls Grok HTTP | Duplicates CLI, needs API key or reverse-engineering |
| Disable Imagine to force read_image only | Grok ignores --disallowed-tools (Q1) |
| Harness GenerateImage in deep-worker | Cursor-specific; not available to deep-worker --provider grok agents |

---

## 8. Failure modes & UX

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| ToolNode "tool not found" | Name mismatch (pre-PR-1) | PR-1 generic routing |
| 120s in_progress notice, empty answer | Bounded wait < generation time | PR-5 longer wait for Imagine |
| promptCapabilities.image: false | Can't attach ref image in prompt | Use image_edit with path arg (Gate 0) |
| Quota / moderation error in rawOutput | Subscription limits | Surface is_error in ToolMessage; no retry loop in provider |
| Artifact path outside working_dir | Grok writes to ~/.grok/... | grok_meta.artifact_path + optional copy-into-working_dir tool (future) |

---

## 9. Open questions (resolve in Gate 0)

1. Exact rawOutput schema for image_gen / image_edit on grok 0.2.54+.
2. Whether video tools appear as tool_call in agent stdio or only via TUI slash commands.
3. Whether results are local paths, file://, or base64 inline blocks.
4. Typical p50/p95 latency per tool (drives IMAGINE_TOOL_WAIT_SECONDS).
5. Should read_image auto-chain after image_gen (agent policy vs provider automation)?

**Recommendation:** keep provider dumb (path in grok_meta only); let the model call
read_image if it needs vision analysis.

---

## 10. Rollout

1. Run Gate 0 probes on host with SuperGrok login.
2. If pass → PR-1 (generic fix) can ship **immediately** — helps all future built-ins.
3. PR-2..5 phased behind Gate 0 evidence.
4. Enable in production by default for all --provider grok agents — no flag needed
   (same as Shell passthrough). Optional --no-imagine **not** proposed: cannot
   prevent Grok from calling tools anyway (Q1).

---

## 11. Success criteria

- [ ] deep-worker test_agent --provider grok -p "generate a cat with image_gen, save path" completes with success: true.
- [ ] dwcli --history test_agent shows image_gen tool_call + ToolMessage with path.
- [ ] Artifact file exists at recorded path.
- [ ] No entry in proxy MCP tools/list for image_gen.
- [ ] Integration test passes with GROK_INTEGRATION_TESTS=1 only (no API key in CI).

---

## 12. PR dependency graph

```
Gate 0 (live check doc)
    ↓
PR-1 generic GrokBuiltin routing
    ↓
PR-2 KNOWN_GROK_BUILTINS += Imagine
    ↓
PR-3 payload rendering ──→ PR-4 tool contract
    ↓
PR-5 bounded wait overrides
    ↓
PR-6 tests + PR-7 docs
```

PR-1 is independently valuable even if Gate 0 fails (future-proofs unknown built-ins).
