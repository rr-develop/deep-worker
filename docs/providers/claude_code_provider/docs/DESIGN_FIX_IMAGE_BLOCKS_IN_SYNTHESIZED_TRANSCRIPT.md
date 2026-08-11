# Design: fix image content blocks in synthesized Claude transcripts

**Status: PROPOSED**
**Date: 2026-08-05**
**Branch: `dev`**

## 1. Symptom (verbatim, from production)

A production agent (provider `claude-code`) could not read images and got a
hard API 400 on every subsequent turn — even a
trivial `2+2`:

```
API Error: 400 messages.18.content.0.tool_result.content.1: Input tag
'image_url' found using 'type' does not match any of the expected tags:
'document', 'image', 'search_result', 'text', 'tool_reference'
```

Source: the agent's error log,
lines 7298, 7420, 7422, 8446, 8568, 8570.

## 2. Root cause (proven, not inferred)

The failing block is at `…tool_result.content.1` — i.e. it lives inside a
**`tool_result` block of the conversation transcript**, not in a live MCP
response.

`claude_code_provider/session_synthesis.py:87` writes `ToolMessage.content`
into the synthesized `.jsonl` transcript **verbatim**:

```python
def _tool_result_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content        # ← OpenAI-style image_url leaks straight through
    return str(content)
```

`read_image` (`deep_worker_cli/tools.py:311`) legitimately returns LangChain
(OpenAI-style) multimodal blocks:

```python
[{"type": "text", ...},
 {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]
```

On the **live** path this is already converted correctly by
`claude_code_provider/chat_model.py:152 _build_mcp_content()` →
`{"type": "image", "data": …, "mimeType": …}` (MCP shape), and the vendored SDK
(`claude_agent_sdk/__init__.py:442`,
`_internal/query.py:552`) handles that shape. **The live path is not broken.**

The transcript-synthesis path is the *only* site that bypasses conversion.

### On-disk evidence (poisoned transcripts, verbatim)

A real CLI transcript (`3f5ad0ba-dd01-4ffc-93ae-c6f5304f8127.jsonl`)
lines 20, 24, 26, 438, 480 contain, inside `tool_result.content`:

```json
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ…"}}
```

### Why it poisoned the session permanently

Once such an entry is in the transcript, every `--resume` replays it, so the
400 recurs on any later request. Commit `6d41c2c` ("Reset poisoned Claude
sessions after API errors") treats that *symptom*; this design removes the
*source*.

### Why "it used to work"

The history-restore / transcript-synthesis path is newer than `read_image`.
Before it existed, image blocks only travelled the live MCP path, which was
(and still is) correct.

## 3. Ground truth for the target format

Taken from **real CLI-written transcripts** (not from guesswork), produced by
Claude's own native `Read` tool:

A real CLI transcript (`e79937ac-8a23-43a0-82f6-73dd554a7de5.jsonl`)
line 9, CLI `version 2.1.220`, `tool_use_id` of a `Read` call:

```json
{"type": "image",
 "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0…"}}
```

Confirmed identically in `0671a8a0-…jsonl` and `592ca104-…jsonl`.

This matches the Anthropic Messages API vision spec
(<https://platform.claude.com/docs/en/build-with-claude/vision>): image content
blocks are `{"type": "image", "source": {"type": "base64", "media_type": …,
"data": …}}`.

**Key distinction (this is the crux of the fix):** there are *two different*
image encodings in play, and they must not be conflated.

| Path | Consumer | Required shape |
|---|---|---|
| Live tool result | in-process MCP server → SDK | `{"type":"image","data":…,"mimeType":…}` (MCP) |
| Synthesized transcript | Claude CLI → Messages API | `{"type":"image","source":{"type":"base64","media_type":…,"data":…}}` (Anthropic) |

`_build_mcp_content` produces the first. The transcript needs the second.
Reusing `_build_mcp_content` here would be **wrong** and would still 400
(`mimeType` is not a valid key for the Messages API).

## 4. Scope and non-goals

### In scope

Convert LangChain `image_url` blocks to the Anthropic `image`/`source` shape
when writing `tool_result.content` in `session_synthesis.py`.

### Explicit non-goals (do NOT change)

1. **`read_image`'s return shape stays LangChain `image_url`.** It is a
   provider-agnostic deep-worker tool also consumed by `grok_provider` and
   `codex_provider` (each has its own `_build_mcp_content`). Changing the tool
   would break those providers. The fix belongs at the claude-transcript
   boundary, which is exactly where the defect is.
2. **No change to `_build_mcp_content`** or the live MCP path — proven working.
3. **No change to the vendored SDK.**
4. **Claude's native `Read`** is not adopted as a replacement: deep-worker
   routes file tools through the proxy-MCP by architecture, and changing that
   is an architecture change requiring explicit approval.

### Known adjacent defect — reported, NOT fixed here

`session_synthesis.py:170` stringifies `HumanMessage.content`:

```python
content = msg.content if isinstance(msg.content, str) else str(msg.content)
```

A multimodal *HumanMessage* (user-attached image) would therefore be written as
a Python `repr` string. That does not produce a 400 (it is valid text), so it
is a separate, lower-severity defect. Flagging it explicitly rather than
silently widening this fix's blast radius. Recommend a follow-up issue.

## 5. Design

### 5.0 Governing principle: fail-closed validation at the boundary

Reviewer blocker 1 and 2 are accepted in full. The converter's job is **not**
"reshape whatever arrives" — it is: *emit into the transcript only IMAGE blocks
that are provably valid for the Anthropic Messages API, and degrade every
invalid or unrecognized block to a diagnostic text block.* A converter that can
still emit an invalid `image`/`source` block just relocates the poisoning
instead of removing it.

**Scope of "provably valid" — explicit, per reviewer's consistency finding.**
This principle is deliberately limited to **image blocks**, which are the
defect under repair and the only block type this fix constructs. Non-image
API-supported blocks (`text`, `document`, `search_result`, `tool_reference`)
retain their **pre-existing pass-through contract**: they are forwarded as-is,
exactly as today, with only their `type` inspected. This design does **not**
introduce schema validation for them.

That is an honest statement of a real limitation, not an oversight: a malformed
`document` block would still pass through. Validating those schemas is a
different, larger job (it needs its own ground truth per type, and we measured
**zero** occurrences of these types in 401 real transcripts — see §5.2), and
silently expanding the implementation to cover it would widen this delta beyond
what was reviewed. Flagged as a follow-up rather than half-implemented.

Two consequences, both mandatory:

* **Validation precedes emission, including for already-Anthropic blocks.**
  Unconditional idempotent pass-through would preserve poison, so an incoming
  `{"type":"image","source":{…}}` is emitted only if it passes the same checks
  as a freshly converted one.
* **Nothing raises.** Restore is an optimization; a validation failure degrades
  to text (existing contract of `session_restore.resolve_restore_session`).

### 5.1 Validation rules for an image block

An image block is emitted **only if all** of the following hold:

1. **Media type whitelist** — `media_type` ∈ `{image/jpeg, image/png,
   image/gif, image/webp}` exactly (lowercased, parameters stripped). Per
   Anthropic vision docs, these are the only supported formats. SVG, BMP,
   `application/octet-stream`, or an arbitrary attacker-supplied MIME is
   rejected → text fallback. This closes the SVG/arbitrary-MIME hole.
2. **Bounded decode, then strict base64.** Order matters and is normative:

   a. **Encoded-size check (before any decoding) — this IS the Anthropic
      limit.** The API cap applies to the **base64-encoded** payload, so it is
      checked directly and exactly on the encoded string:

      * require the data to be ASCII (`data.encode("ascii")`); a
        `UnicodeEncodeError` means it is not valid base64 → reject;
      * require `len(data.encode("ascii")) <= _MAX_BASE64_IMAGE_BYTES`.

      This is an **exact** check of the documented contract, not an estimate.
      It also fully closes the allocation budget: nothing oversized ever
      reaches `b64decode` or Pillow. The earlier draft's lower-bound
      approximation `(len - 2) // 4 * 3` is therefore removed — it existed
      only because the limit was mistakenly modelled on decoded bytes, and it
      is both unnecessary and less precise than this check.
   b. **Strict decode** — `base64.b64decode(data, validate=True)`, rejecting an
      invalid alphabet; padding errors raise `binascii.Error`, which is caught.
      Empty data is rejected. A regex match alone is *not* accepted as proof of
      validity.
   c. **Decoded sanity bound (secondary, NOT the API contract)** — decoded
      length must be non-zero. A decoded-size bound is implied by (a) —
      encoded ≤ 10,000,000 ⇒ decoded ≤ 7,500,000 — and is *not* presented as
      "the Anthropic 10 MB limit". Conflating the two is exactly the error
      corrected below.

   This ordering closes both the allocation-budget objection *and* the
   encoded-vs-decoded error: the size contract is checked exactly, before any
   allocation.
3. **MIME/magic agreement (spoof guard)** — the decoded bytes' magic number
   must agree with the declared `media_type` (PNG `\x89PNG\r\n\x1a\n`, JPEG
   `\xff\xd8\xff`, GIF `GIF87a`/`GIF89a`, WebP `RIFF….WEBP`). A `.png`-declared
   payload that is actually something else is rejected. This is a cheap
   header check on already-decoded bytes — no extra decode budget.
4. **Size limit — applies to the ENCODED payload (corrected).**

   **This corrects a real error in the previous revision.** That revision
   capped the *decoded* bytes at 10,000,000. The Anthropic Vision documentation
   states the limit as:

   > "The maximum size per image is: **10 MB (base64-encoded)** when using the
   > Claude API directly."
   > — <https://platform.claude.com/docs/en/build-with-claude/vision>

   The cap is on the **base64-encoded** payload, not the decoded bytes.
   Arithmetic showing why the previous revision was unsafe (computed, not
   estimated):

   * decoded 10,000,000 B → encoded `4 * ceil(10e6/3)` = **13,333,336 B ≈
     13.33 MB** — over the API limit, so the API could reject it and re-poison
     the transcript, i.e. the exact failure this design exists to prevent;
   * conversely, encoded cap 10,000,000 B → decoded ≤ **7,500,000 B**.

   The normative constant is therefore named for what it actually bounds:

   ```python
   _MAX_BASE64_IMAGE_BYTES = 10_000_000   # ENCODED bytes; decimal MB, not MiB
   ```

   The decimal-vs-binary reasoning still holds and is retained: 10,000,000 <
   10,485,760, so if the docs mean MiB we are strictly inside the limit; if
   they mean decimal MB we sit exactly on it. Choosing the larger reading would
   risk emitting a block the API rejects.

   This single named constant is used by the encoded-size check (§5.1.2a) and
   by every test — no bare literal is repeated anywhere, so checks and tests
   cannot drift apart. The old `_MAX_IMAGE_BYTES` name is deliberately gone:
   keeping a decoded-sounding name for an encoded cap is how this class of
   confusion recurs.

   Note this is deliberately independent of `read_image`'s unrelated and
   incorrect 20 MB cap (§5.3), which remains a follow-up. (See §5.3 for why the
   limit is enforced here and not in `read_image`.)
5. **Dimensions ≤ 8000×8000 px** — via Pillow header-only inspection
   (`Image.open(BytesIO(...))` reads just the header for `.size`; no full
   decode). Pillow 12.0.0 is already present and is already a dependency of
   `read_image`'s dimension probe. Any Pillow exception (including
   `UnidentifiedImageError`) ⇒ reject, never raise. **Decoding budget:** header
   parse only, and it is reached only after the §5.1.2 byte bound has already
   capped the input at `_MAX_BASE64_IMAGE_BYTES` (encoded), i.e. ≤ 7,500,000
   decoded bytes.

   **Decompression-bomb handling — corrected per reviewer.** Pillow's guard is
   *not* uniformly an exception. Verified by reading Pillow 12.0.0's
   `Image._decompression_bomb_check` source in this environment:

   ```python
   if pixels > 2 * MAX_IMAGE_PIXELS:  raise DecompressionBombError(...)
   if pixels > MAX_IMAGE_PIXELS:      warnings.warn(..., DecompressionBombWarning)
   ```

   So between `MAX_IMAGE_PIXELS` (89,478,485) and 2× that, Pillow only *warns*
   — an image in that band would have sailed through had we relied on the
   exception alone. The earlier draft's claim was wrong; the reviewer was
   right to flag it.

   Resolution: the authoritative guard is our own **8000×8000 check** (64 M px,
   which is *below* Pillow's warn threshold, so any bomb-band image is rejected
   by our dimension rule regardless of Pillow's behaviour). Additionally, the
   header parse runs inside `warnings.catch_warnings()` with
   `simplefilter("error", Image.DecompressionBombWarning)`, converting the
   warning into an exception that our existing catch-all turns into a
   rejection. Belt and braces: neither mechanism alone is trusted.

Rejection produces a **diagnostic text block that never contains raw base64**:

```
[Image omitted from restored history: <reason>; media_type=<declared>, <N> bytes]
```

`<reason>` is a fixed enum string (`unsupported media type`, `invalid base64`,
`media type mismatch`, `too large`, `dimensions too large`, `unreadable
image`). Neither the payload nor a truncated prefix of it is ever interpolated
— that would re-introduce bulk into the transcript and leak image bytes into
text.

### 5.2 `_tool_result_content` becomes block-aware, fail-closed

```python
def _tool_result_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_transcript_block(b) for b in content] or ""
    return str(content)
```

`_transcript_block` mapping — a **closed** schema policy, resolving the §7
wording conflict the reviewer flagged:

| Input block | Output |
|---|---|
| `{"type": "text", …}` | unchanged |
| `{"type":"document"\|"search_result"\|"tool_reference", …}` | unchanged (pass-through, `type` checked only — see §5.0 scope) |
| `{"type": "image_url", …}` with data URL, **passing §5.1** | Anthropic `image`/`source` |
| `{"type": "image_url", …}` with data URL, **failing §5.1** | diagnostic text block |
| `{"type": "image_url", …}` non-data URL | `{"type":"text","text":"[Image: <url>]"}` |
| already-Anthropic `{"type":"image","source":{…}}` | re-emitted **only if** it passes §5.1; else diagnostic text |
| MCP `{"type":"image","data":…,"mimeType":…}` | normalized to `source`, **only if** it passes §5.1; else diagnostic text |
| unknown `dict` block | `{"type":"text","text": str(block)}` |
| non-dict block | `{"type":"text","text": str(block)}` |

The supported-type whitelist is exactly the set the real API 400 enumerates:
`document`, `image`, `search_result`, `text`, `tool_reference`. Those are
preserved rather than flattened to text, so a legitimate future block type is
not silently destroyed.

**Empirical check on the pass-through set:** scanning 401 real CLI transcripts,
`tool_result.content` contained only `text` blocks
(13,740) and bare strings (11,715); zero `document`/`search_result`/
`tool_reference` occurrences anywhere. So the pass-through arms are defensive,
not hot paths — which is why the *unknown* arm must stay fail-closed to text.

An all-rejected list degrades to `""` rather than an empty list (an empty
`content` list is invalid for the API).

### 5.3 Where the 10 MB / dimension limits are enforced — and where they are not

The limits are enforced **in the claude transcript converter**, per reviewer
blocker 1. Rationale: an existing 10–20 MB payload produced by today's
`read_image` (which advertises 20 MB) would otherwise pass straight into the
transcript as a valid-looking `image`/`source` block and re-poison resume. The
guard therefore has to live on the boundary that actually writes the transcript.

This does **not** change `read_image` or any provider-agnostic contract: no
other provider's path is touched, and the tool's return value is unchanged.

Separately, `read_image`'s advertised 20 MB cap
(`tools.py:_MAX_IMAGE_SIZE_BYTES`, measured on raw file bytes) contradicts the
documented 10 MB *base64-encoded* API limit — and, because base64 inflates by
~4/3, a 20 MB file becomes ~26.7 MB encoded, roughly 2.7× the cap.
That remains a genuine defect in a provider-agnostic tool affecting grok/codex
too; it is reported as a follow-up rather than silently widened into this
claude-only delta. The converter guard above means claude sessions are safe
regardless of how that follow-up is resolved.

## 6. Test plan (TDD, no mocks where it matters)

### RED first

1. `test_synthesized_tool_result_has_no_image_url` — synthesize a session from
   a `ToolMessage` whose content is exactly `read_image`'s real return value;
   assert no `image_url` anywhere in the written `.jsonl`. **Must fail before
   the fix.**
2. `test_synthesized_tool_result_uses_anthropic_image_shape` — assert the block
   equals `{"type":"image","source":{"type":"base64","media_type":"image/png",
   "data":"…"}}`, byte-identical to the shape captured from the real CLI
   transcript above.

### Unit — happy path

3. jpeg/png/gif/webp media types round-trip, each with **real** encoded bytes
   generated by Pillow (not fake `"AAA"` strings), so magic-number validation
   is exercised honestly.
4. String content unchanged (regression guard for the dominant case).
5. Mixed text+image list preserves order and count.
6. `read_image`'s literal output, fed through `synthesize_session_file`,
   produces a transcript that json-parses and contains a valid image block.
7. MCP `mimeType` block normalized to `source` shape (valid payload).
8. Already-Anthropic **valid** block re-emitted unchanged (idempotence).

### Unit — fail-closed validation (reviewer blocker 1)

9. Unsupported MIME (`image/svg+xml`, `image/bmp`,
   `application/octet-stream`) → diagnostic text, no `image` block emitted.
10. MIME spoof: `media_type=image/png` but payload is real JPEG bytes → rejected.
11. Invalid base64 alphabet (`"!!!!"`) → rejected, no raise.
12. Bad padding → rejected via `binascii.Error`, no raise.
13. Empty image data → rejected.
14. **Encoded-size limit boundary cases** (per reviewer's final blocker), all
    expressed in terms of `_MAX_BASE64_IMAGE_BYTES`:
    * encoded length exactly `cap` → accepted (assuming all other criteria pass);
    * encoded length `cap + 1` → rejected;
    * padding boundaries (`=`, `==`) around the cap behave consistently;
    * **the decisive case**: a payload whose *decoded* size is under 10 MB but
      whose *encoded* size exceeds the cap → **rejected**. Under the previous
      (decoded-based) revision this payload was accepted, so this test fails on
      the old contract and passes on the corrected one.
    * non-ASCII data (e.g. a smart quote in the string) → invalid-base64
      rejection, no `UnicodeEncodeError` escaping.
14b. **The encoded check really happens before any decoding**: an oversized
    encoded string is rejected **without** calling `b64decode` and **without**
    touching Pillow. Asserted by monkeypatching `base64.b64decode` and
    `Image.open` to raise `AssertionError` if called; the test passes only if
    neither is invoked. This is what distinguishes an enforced budget from a
    claimed one.
15. Dimensions > 8000×8000 → rejected (real oversized image, Pillow header path).
16. Decompression-bomb band: an image whose pixel count sits **between**
    `MAX_IMAGE_PIXELS` and `2 × MAX_IMAGE_PIXELS` — i.e. where Pillow only
    *warns* — is still rejected. Guards the exact gap the reviewer identified.
    A `DecompressionBombError` (above 2×) is likewise caught → rejected, no raise.
17. Malformed already-Anthropic block (missing/!=base64 `source.type`, missing
    `data`/`media_type`) → rejected, **poison does not survive idempotence**.
18. Malformed MCP block (missing `mimeType`/`data`) → rejected.
19. **No raw base64 leakage**: for every rejection case, assert the emitted text
    contains neither the payload nor any ≥32-char substring of it, and that the
    block stays under a small size bound.

### Unit — schema policy (reviewer blocker 2)

20. `document`, `search_result`, `tool_reference`, `text` blocks all pass
    through byte-identical (no flattening of API-supported types).
21. Unknown dict block type → text fallback.
22. Non-dict list element → text fallback.
23. All-rejected list degrades to `""`, never an empty list.

### Integration (real, no mocks) — reviewer MAJOR

24. Synthesize a transcript containing a **real** PNG, resume it through the
    **production** path, and assert **all** of the following.

    **Corrected resume-evidence contract (reviewer blocker 2).** The earlier
    draft's assertion "returned UUID equals the synthesized UUID" was wrong.
    Verified in `client_manager.py:543`, production resumes with:

    ```python
    options = replace(options, resume=resume_session_id, fork_session=True)
    ```

    `fork_session=True` deliberately forks to a **new** session id so we get our
    own writable transcript and never corrupt the source. So the live session
    UUID is *expected* to differ from the synthesized source UUID. Asserting
    equality would either fail spuriously or push the test into a
    non-production `fork_session=False` configuration — exactly as the reviewer
    predicted. Evidence is therefore three-part:

    * **(a) Options evidence** — the `ClaudeAgentOptions` actually handed to
      `ClaudeSDKClient` carry `resume == <our synthesized uuid>` and
      `fork_session is True`. Captured from the real options object on the
      production path, not by reconstructing it in the test.
    * **(b) Semantic recall evidence** — a **fingerprint** unique to the
      synthesized transcript (a nonce string embedded in a prior turn, plus a
      known fact recoverable only from the image) is recalled after resume. A
      fresh fallback session cannot know the fingerprint, so this distinguishes
      genuine resume from silent fallback. This matters because
      `client_manager.py:567` has an explicit
      `replace(options, resume=None, fork_session=False)` fallback on connect
      failure — the precise silent-fallback path that would otherwise green a
      naive test.
    * **(c) Fork UUID** — the returned/live UUID is asserted to be a valid
      session id that *may* differ from the source; difference is expected and
      is **not** treated as failure.

    Plus: the turn completes with **no API 400**.

25. **Mutation RED** on the old behaviour: with `_tool_result_content` reverted
    to verbatim pass-through, test 24 must FAIL with the verbatim 400
    `Input tag 'image_url' …`. This proves the test actually covers the defect
    rather than passing for unrelated reasons.

### Live end-to-end verification (26–27) — explicitly requested by the user

**Which test carries the source-fix proof.** Test **#24** (integration) is the
authoritative proof that this fix repairs the synthesis path. The live tests
below are *additional* end-to-end confidence, and #27 is constructed so it
cannot pass without the fix. This division is stated explicitly because the
reviewer correctly caught that the earlier wording over-promised.

**Why a naive live restart would prove nothing.** Verified in
`session_restore.py` (module docstring, lines 6–8) and
`resolve_restore_session`: a plain restart of the same agent with a valid
persisted hint whose transcript tail matches history takes **Approach A** —
it resumes the *existing CLI-written* transcript and never calls
`_synthesize`. The live MCP path already writes correct image blocks, so such
a test would go green with or without this fix. That is exactly the false-green
the reviewer identified.

26. **Live vision check (does the model actually see the image?).** Start a
    throwaway `claude-code` agent, hand it a generated image containing known
    text/shape, ask what it sees, and assert the answer matches the known
    content — proving the model genuinely *sees* the image rather than merely
    not erroring. Declared scope: this exercises the **live MCP path only**. It
    is *not* evidence for the synthesis fix.

27. **Live synthesis-path check (constructed to require the fix).** After the
    image turn of #26, force production restore down **Approach B**
    deterministically — invalidate the candidate hint (remove/poison the
    persisted `session_uuid`) while keeping the LangGraph history intact, so
    `resolve_restore_session` finds no usable candidate and must synthesize.
    Then assert:

    * a **new synthesized source file** was written;
    * `options.resume` equals that file's UUID with `fork_session=True`
      (options evidence, as in #24);
    * after the fork, the model recalls both the **nonce** from the prior turn
      and an **image-only fact** (a detail obtainable solely from the picture) —
      a fresh fallback session can know neither;
    * no API 400.

    **Mutation RED for the live scenario:** with `_tool_result_content`
    reverted to verbatim pass-through, #27 must fail with the verbatim
    `Input tag 'image_url' …` 400. Without this, #27 could not claim to cover
    the defect.

    *(These supersede the earlier draft's stray item "12", which the reviewer
    correctly flagged as misnumbered.)*

All integration and live tests: hard overall time budget, event-driven waiting,
and a guaranteed `SIGTERM`→`SIGKILL` teardown in `finally`.

## 7. Risk assessment

* **Blast radius:** one function in one claude-only module. `grok`/`codex`
  providers, `read_image`, and the live MCP path are untouched.
* **Backward compatibility:** string content — the overwhelmingly common case —
  takes an unchanged code path (measured: 11,715 bare-string `tool_result`
  contents across 401 real transcripts). Among *list* blocks, `text` and the
  other API-supported types (`document`, `search_result`, `tool_reference`)
  pass through unchanged; only image blocks are normalized, and only unknown
  block types are flattened to text. This is the precise, fail-closed
  restatement of what §5.2 tabulates — superseding the earlier loose phrasing
  "non-image list blocks unchanged", which the reviewer correctly flagged as
  contradicting the `anything else -> text` row.
* **Failure mode:** conversion is total and never raises; worst case a block
  degrades to text, which is strictly better than a 400 that bricks the session.
* **Existing poisoned transcripts on disk** are not rewritten by this fix.
  They are already handled by the `6d41c2c` session-reset mechanism, which
  rotates away from a poisoned session. Retro-cleaning user transcript files is
  out of scope and would be a destructive operation on user data.

## 8. Policy compliance

* No polling, no new threads, no blocking calls added.
* No architecture, contract, or policy change.
* No `--no-verify`; pre-commit must pass.
* Work on `dev` per default-branch policy.
