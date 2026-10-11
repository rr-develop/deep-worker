# DESIGN — Image blocks must survive large-tool-result eviction

Status: proposed (awaiting review)
Scope: `deep_worker/deepagents/deepagents/middleware/filesystem.py`
Related, already committed: `b6bdb81` (Claude transcript synthesis) — a *different* defect.

## 1. User-visible symptom

`mcp__deepworker__read_image` never shows the model an image. On a claude-code-sdk
agent the standard MCP path produces a tool result that the model can only see
as a Python `repr` preview, e.g.

```
[{'type': 'text', 'text': 'Image: photo.jpg (576x1280), image/jpeg, 96840 bytes'},
 {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ...'}}]
```

The model sees *text describing a list*, not an image. Reproduced live with
two local JPEG files (96 840 B and 84 584 B).

## 2. Root cause (confirmed in code, not inferred)

`FilesystemMiddleware._process_large_message` (`filesystem.py:1050`) and its async
twin `_aprocess_large_message` (`filesystem.py:1122`) stringify the **entire**
`ToolMessage.content` before the size check:

```python
else:
    # Multiple blocks or non-text content - stringify entire structure
    content_str = str(message.content)
```

The existing docstring states the assumption explicitly (`filesystem.py:1071-1073`):

> ToolMessage supports multimodal content blocks (images, audio, etc.), but these
> are uncommon in tool results. For simplicity, all content is stringified and
> evicted.

That assumption is false for `read_image`, whose result is *by construction* a
base64 data URL. A 96 KB JPEG is ~129 KB of base64 — far above
`NUM_CHARS_PER_TOKEN * _tool_token_limit_before_evict`. Therefore the image is
**always** evicted and **always** destroyed. This is deterministic, not
size-dependent or intermittent.

After eviction the message content is a plain `str`. Downstream conversion is
then powerless — `ClaudeCodeChatModel._build_mcp_content` (`chat_model.py:152`)
already converts `image_url` → MCP `image` correctly, but by the time it runs
there is no block left to convert, only a string.

### 2.1 Why this is provider-agnostic

The eviction middleware runs in `deepagents`, above the provider layer. Every
provider loses the image identically. claude-code-sdk additionally suffered a 400
when such a block reached a synthesized transcript verbatim — that separate
defect is already fixed by `b6bdb81` and is **not** re-litigated here.

### 2.2 Relationship to the previous fix

`b6bdb81` fixed transcript synthesis (poisoned `--resume`, permanent 400). It
was necessary and remains correct, but it does **not** make images readable: the
live path never gets an image block in the first place. This design closes that
remaining gap. The two fixes are complementary and independent.

## 3. Requirements

R1. A tool result containing image blocks must reach the provider layer with
    those image blocks intact, so provider-specific conversion can run.
R2. Context-window protection must be preserved: large **text** must still be
    offloaded exactly as today.
R3. Provider-agnostic. No change to grok/codex/claude provider code, no change
    to `read_image`, no change to the tool contract.
R4. Fail-safe: a malformed or unrecognised block must never crash eviction and
    must never silently vanish.
R5. Bounded context cost, **in aggregate**. Preserving images must not
    reintroduce unbounded growth: an image is preserved because it is
    *semantically irreplaceable*, not because size stopped mattering. Requires a
    per-block cap, a block-count cap **and** a total per-result byte budget
    (§5.3); per-block caps alone would permit ~80 MB in one message.

## 4. Rejected alternatives

A. **Add `read_image` to `TOOLS_EXCLUDED_FROM_EVICTION`.**
   Rejected. It is tool-name-based, so any other tool returning an image (MCP
   servers, future tools) stays broken. It also disables *text* eviction for
   that tool, so a huge text block alongside an image would bypass protection.
   The correct axis is **block type**, not tool name.

B. **Offload the image and leave a path reference.**
   Rejected: that is exactly today's behaviour and the reason the feature is
   broken. The model cannot re-read a binary image through `read_file`.

C. **Convert to Anthropic shape here.**
   Rejected: violates R3. `deepagents` is provider-agnostic; the LangChain
   `image_url` shape is the correct interchange form. Conversion belongs in the
   provider (`_build_mcp_content`), which already does it.

D. **Raise the eviction threshold.**
   Rejected: does not fix the class of bug, and removes context protection.

## 5. Design

### 5.1 Block partitioning (the core change)

Replace whole-content stringification with a partition that **preserves original
block order**. For `list` content, classify each block by index:

- **preserved** — image blocks (`type == "image_url"`, and the Anthropic/MCP
  shapes `type == "image"`), subject to the caps in §5.3;
- **evictable** — everything else (text and unknown blocks).

Then two **independent** outcomes are decided — this separation is normative,
because conflating them would let a capped block resurrect (see §5.1.1):

1. Classify every block, applying the §5.3 caps. Classification either leaves the
   list untouched or produces **transformations** (an image downgraded to a
   no-payload diagnostic because it is oversized, malformed, over-count or
   over-aggregate).
2. Compute `content_str` from **evictable blocks only** (representation pinned in
   §5.2.1) and apply the existing size check to decide whether a **text offload**
   happens.
3. Materialize:
   - offload needed → write evictable text to `/large_tool_results/{id}` and
     rebuild the message in original order (§5.2.2);
   - no offload but classification **did** transform → rebuild the message from
     the classified blocks, with **no** file write and **no** notice;
   - no offload and classification transformed **nothing** → return the exact
     original message object, byte-identical to today.

So images ride along with the notice rather than being serialized into it.

#### 5.1.1 Caps must always be materialized (normative)

A cap decision is **never** conditional on the text-offload decision. If
classification rejected a payload, the returned message must reflect that
rejection even when no offload occurs.

The dangerous case is the common one: a 30 MB image with a one-line metadata
text. The evictable text is tiny, so the size check says "no eviction" — and a
naive `return message` would hand back the original list **including the very
payload the caps just rejected**, making every §5.3 cap ineffective in exactly
the situation they exist for. Hence the three-way materialization above: the
"return original" branch is reachable **only** when classification was a no-op.

Pinned by tests #21-#25.

### 5.2 Behaviour matrix

| Content | Today | After |
|---|---|---|
| `str`, small | unchanged | unchanged |
| `str`, large | evicted to file | evicted to file (identical) |
| list, text only, small | unchanged | unchanged |
| list, text only, large | evicted (stringified list) | evicted (§5.2.1 repr) |
| list with image, **evictable text** large | **image destroyed** | text evicted, **image preserved in place** |
| list with image, total large **only because of the image** | **image destroyed** | **unchanged** — no offload (§5.1 branch c) |
| list with image, total small | unchanged, image preserved | unchanged, image preserved |
| unknown blocks | stringified | stringified (evictable) — unchanged |

**Correction (was wrong in revision 1).** Row 7 (`total small`) is *not* broken
today: lines 1095-1097 return the original `message` object before any
replacement is built, so a small multi-block image result already survives
intact. Stringification there is only a *measurement* step.

The real defect is **row 6**, which is the actual `read_image` shape: one large
image plus a line of metadata. Today the base64 `repr` inflates the *measured*
string over the threshold, so the content is evicted and the image destroyed —
even though the only genuinely large thing is the image itself. Under §5.1 the
size check runs on the **evictable subset only**, so that metadata line stays
below the threshold, no offload occurs, and the original list is returned with
the image intact. Row 5 (genuinely large accompanying text) is the only case
where an offload still happens, and there the image is preserved in place.

#### 5.2.1 Pinned evictable representation

To avoid silently changing what gets written to the offload file:

- content is `str` → written verbatim (unchanged);
- exactly one evictable block, of type `text` → its `text` value (preserves the
  existing readability special case);
- otherwise → `str(list_of_evictable_blocks)`, i.e. the same stringified-list
  form as today, just computed over the evictable subset.

This is *not* claimed to be byte-identical to today when images are present —
that is the intended fix (the base64 no longer lands in the file). For
image-free content it **is** byte-identical, and that is pinned by regression
tests.

#### 5.2.2 Image **order** preservation (labels are NOT preserved)

The contract provided here is **relative image order**, not label association.
Stating it plainly: when a text offload occurs, inline labels are **lost**.

The replacement content is built by walking the original blocks in order:

- the first evictable block position is replaced by the single
  `TOO_LARGE_TOOL_MSG` notice text block;
- subsequent evictable blocks are dropped (their content lives in the file);
- preserved image blocks are emitted **at their original relative positions**.

So `text("Image 1:"), image A, text("Image 2:"), image B` becomes
`notice, image A, image B`. A still precedes B, but **both labels are gone** —
the model can no longer tell from the message which image was "Image 1".

This is an accepted, explicit tradeoff, permitted only because the offload was
genuinely necessary (the accompanying text was large) and the notice names the
file where that text — labels included — can be read back. It is **not**
preserved association and is not described as such anywhere in this document.

When no offload occurs the original blocks, labels included, are untouched
(§5.1 step 3), which is the common `read_image` case.

### 5.3 Caps (R5)

`deepagents` is provider-agnostic and deliberately does **not** validate MIME,
magic bytes or dimensions — provider adapters remain **authoritative** for
validity. The caps here are therefore a conservative *deepagents safety ceiling*
on context cost only. Passing them does **not** assert the block is acceptable to
any particular provider or API; different providers and request limits differ.

- `_MAX_PRESERVED_IMAGE_ENCODED_BYTES = 10_000_000` — per-block encoded cap,
  measured as exact ASCII/base64 character count **before** any decode. No
  decoding, no Pillow dependency in `deepagents`.
- `_MAX_PRESERVED_IMAGE_BLOCKS = 8` — per-result block count cap.
- `_MAX_PRESERVED_IMAGE_TOTAL_BYTES = 10_000_000` — **aggregate** encoded budget
  across the whole tool result (closes the 8 × 10 MB ≈ 80 MB hole).

**Deterministic aggregate behaviour.** Blocks are considered in original order.
A running total is kept; a block is preserved only if it passes the per-block cap
**and** its encoded size fits the remaining aggregate budget. The first block
that would exceed the budget — and every image block after it — is downgraded.
No re-ordering, no greedy "best fit": order-deterministic and trivially testable.

A downgraded block is **not** silently dropped (R4). It becomes a short
diagnostic text block carrying **no payload bytes**, e.g.
`[Image omitted: encoded payload 13.4 MB exceeds 10.0 MB per-image limit]` or
`[Image omitted: aggregate image budget 10.0 MB exhausted]`, and that block then
follows the normal evictable path. Non-data URLs (`http(s)://`) are left as-is in
the evictable partition — no fetching, no new egress, no time-of-replay contract.

### 5.4 Sync/async parity

`_process_large_message` and `_aprocess_large_message` are near-duplicates.
Partitioning/classification is extracted into module-level pure helpers used by
both, so the two cannot drift. Only backend I/O stays duplicated.

## 6. Test plan (TDD, RED first)

RED reproducer (must fail before the fix):
1. `read_image`-shaped result (text + `image_url` data URL) above the threshold →
   assert the returned content still contains an `image_url` block. Fails today.

Unit (sync **and** async, asserted equivalent):
2. Large text-only list → evicted exactly as today (regression pin).
3. Large `str` → evicted exactly as today (regression pin).
4. Small content → returned unchanged, byte-identical (pins §5.2 row 6).
5. Image + huge text → text offloaded, image preserved, notice present.
6. Offloaded file must **not** contain the base64 payload.
7. Anthropic-shape (`type: image`) and MCP-shape blocks preserved too.
8. Oversize image → downgraded to diagnostic text, no payload leak.
9. `> _MAX_PRESERVED_IMAGE_BLOCKS` images → excess downgraded, cap honoured.
10. Malformed image blocks (missing url/data, non-dict, empty) → never raise.
11. Non-data URL → left in evictable partition, no fetch.
12. Sync/async parity on identical inputs.
13. `Command`-wrapped multi-message results preserve images per message.
14. Existing middleware suite passes unchanged (no behavioural regression).
15. **Aggregate budget**: N images each under the per-block cap but jointly over
    `_MAX_PRESERVED_IMAGE_TOTAL_BYTES` → deterministic prefix preserved, first
    over-budget block and all later image blocks downgraded, exact boundary
    (`budget`, `budget+1`) pinned.
16. **Image order** (NOT labels): `text,imgA,text,imgB` over threshold → result is
    `notice,imgA,imgB`, asserting A precedes B. The test asserts only relative
    image order and explicitly asserts that the labels are **absent** — pinning
    the honest contract of §5.2.2, not an association guarantee.
17. **Pinned evictable representation** (§5.2.1): single-text special case, and
    stringified-list form byte-identical to today for image-free content.

Caps must be materialized without an offload (§5.1.1) — the resurrection hole.
Each of these uses an image that classification rejects **plus small text**, so
the eviction threshold is NOT crossed, and asserts the returned message contains
the diagnostic and **no rejected payload bytes**:
21. Oversized image + small text → diagnostic present, payload absent, no file
    written, no `TOO_LARGE_TOOL_MSG` notice.
22. Malformed image block + small text → diagnostic present, never raises.
23. Count-excess images + small text → excess downgraded, cap honoured.
24. Aggregate-excess images + small text → deterministic prefix preserved,
    remainder downgraded.
25. **No-op classification** + small text → the returned object is the **exact
    original message** (identity/byte-identity), proving the "return original"
    branch is reachable only when nothing was transformed.

Permanent gated proof — must traverse the **real seam**
(tool result → `FilesystemMiddleware` eviction → provider conversion → vision):
26. Full-agent-lifecycle test on a **generated deterministic image** built by the
    test itself with stdlib only (`zlib`+`struct` raw PNG, no Pillow): an image
    rendered with a nonced visual fact (a random codeword drawn into the bitmap)
    large enough to cross the eviction threshold. The agent must call
    `read_image` through the standard MCP path and the model must **report the
    nonce**. No dependency on mutable local scratch files. Bounded teardown
    (SIGTERM→SIGKILL), hard time budget per policy. A direct SDK/`real_cli` call
    that bypasses the middleware is explicitly **not** accepted as proof.
27. **Mutation RED**: disabling middleware image preservation must make #26 fail
    semantic recognition — proving sensitivity to this exact source change.

Manual acceptance evidence (user-requested, not a committed test):
28. A real claude-code-sdk agent reads **each** file in a local test directory via the
    standard MCP path and names its actual visible content. Generic answers,
    refusals, or describing a "list"/"data URL" count as failure. Recorded in the
    report.

## 7. Out of scope (reported, not fixed here)

- Multimodal `HumanMessage` is still stringified elsewhere.
- `read_image` advertises a 20 MB raw-file cap (~26.7 MB encoded), well above the
  10 MB API limit; aligning the tool's own cap is a separate change to a
  provider-agnostic tool contract.
- No change to `TOOLS_EXCLUDED_FROM_EVICTION`.

## 8. Risk

Low and contained. For image-free results behaviour is byte-identical to today
(rows 1-4 and 7 of §5.2, pinned by regression tests). Small multi-block results
were already correct (row 6) and stay untouched. The only behavioural change is
row 5 — the case that was destroying images.

Residual risk is context growth from preserved images, bounded by the three caps
in §5.3 (per-block, count, aggregate). Worst case per tool result is
`_MAX_PRESERVED_IMAGE_TOTAL_BYTES`, which is a deliberate, testable ceiling
rather than an open-ended one. Provider adapters remain authoritative for
whether a preserved block is actually acceptable to their API.
