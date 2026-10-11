# DESIGN: the bridge must return MCP content, not raw decoded JSON

**Status:** DESIGN — revision 2, after review (2026-09-04).
Incorporated: §6 rewritten around the existing test file (blocking [1]), §5.4
corrects a factual error about `None` (blocking [2]), §4 and §5.2
strengthened with the reviewer's arguments [3]. Awaiting re-read by hash.
**Date:** 2026-09-04
**Branch:** `dev`
**Related:** `DESIGN_LSP_SHARED_SERVICE.md` (§2 MCP↔gRPC bridge — this defect
lives in the last hop of that bridge)

## 1. Symptom

Calling `serena__find_symbol` from an agent returns not a result but a
runtime exception:

```
Tool 'serena__find_symbol' raised an unhandled exception (RuntimeError):
13 validation errors for CallToolResult
content.0.TextContent.type
  Field required [type=missing,
    input_value={'name_path': 'build_mcp_...: 145, 'end_line': 160}}, input_type=dict]
content.0.ImageContent.type
  Field required [...]
content.0.AudioContent.type
  Field required [...]
...
```

Reproduced by a direct call (2026-09-04) and in the history of a test agent
(2026-09-04 16:10:17). The number of errors grows with the number of symbols
found: 13 for one symbol, 26 for two, 247 for nineteen — roughly 13 union
variants per element.

## 2. What is NOT the cause

The test agent concluded: "the bug is in the MCP provider, add `type="text"`
to the response's `TextContent`". This is a wrong localization, and a fix
based on it would have changed nothing: Serena returns **correct** content
blocks, with `type` in place. The data loses its shape later, already inside
our bridge.

Serena itself is also not the cause: the error text shows the symbol was
found correctly (`build_mcp_server`, `'end_line': 160` — matches
`bridge_stdio.py:146-161`).

## 3. Cause

The return path, bottom to top:

| step | code | what happens |
|---|---|---|
| 1 | `serena_backend.py:1120` `_flatten_content` | concatenates the `.text` of all Serena blocks into one string |
| 2 | `serena_backend.py:1117` | the string is put into `pb.ToolResult.content_json` |
| 3 | `bridge_loop.py:217` | `json.loads(content_json)` → Python object |
| 4 | `bridge_stdio.py:159` | that object is `return`ed straight from `@server.call_tool()` |

The `@server.call_tool()` handler must return `list[ContentBlock]`, a
`dict`, or a pair. The MCP SDK normalizes the return value like this
(`mcp/server/lowlevel/server.py`, `Server.call_tool.decorator.handler`):

- `dict` → wrapped by itself in `TextContent(type="text", ...)` and also
  placed in `structuredContent`;
- any other `__iter__` → treated as a **ready-made** `UnstructuredContent`
  and goes into `CallToolResult` validation as is.

We land in the second branch with a list of raw symbol dicts. Each dict is
run through the union `TextContent | ImageContent | AudioContent |
ResourceLink | EmbeddedResource` and fits none of the variants.

**The defect is a skipped step:** the response was unwrapped (step 3) and
not wrapped back into a content block (step 4).

## 4. Blast radius: the defect is NOT universal

An important clarification, obtained by execution rather than by reasoning.
Only tools whose top-level JSON is an **array** break. Tools returning an
object work, because the SDK handles `dict` separately.

Verified by calls (2026-09-04):

| tool | top level | status |
|---|---|---|
| `find_symbol` | array | **BROKEN** (13/26/247 errors) |
| `get_symbols_overview` | object | works |
| `find_referencing_symbols` | object | works |
| `get_diagnostics_for_file` | object (`{}`) | works |

`find_declaration` and `find_implementations` returned their own meaningful
errors in the probe (non-unique regex; `textDocument/implementation` not
supported by the language server) — their response shape was not established
by this probe. The remaining tools (`rename_symbol`, `replace_symbol_body`,
`insert_*`, `safe_delete_symbol`) are mutating and were not checked on a
live tree.

It follows that what needs fixing is **not** "the broken `find_symbol`" but
the return normalization for all 11 tools at once. The shape of Serena's
response is its own internal business, and one cannot rely on "it's always an
object here".

**A stronger argument than the one above** (found by the reviewer, verified
by execution). The proposed wrapper was run against shapes that are not in
the table: a bare string (`"renamed 3 occurrences"`), `int`, `bool`, `[]` —
each produced exactly one valid `TextContent`. That is, the solution covers
even the shapes the probe did NOT establish, not only the array. This is the
real reason to fix all 11 at once: not "we enumerated the shapes", but "the
normalization does not depend on which shape the source picks tomorrow".

## 5. Solution

Wrap the result in a content block in `bridge_stdio.py`, at the single return
point:

```python
@server.call_tool()
async def _call_tool(name, arguments):
    result = await session.call_tool(name, arguments or {})
    return [types.TextContent(type="text", text=_render(result))]
```

where `_render` produces stable text: `json.dumps(result, indent=2,
ensure_ascii=False, default=str)`. `None` is NOT wrapped — see §5.3.

`default=str` costs one line and covers a shape that is otherwise hard to
catch: a non-JSON-serializable payload would make `json.dumps` raise, and the
SDK would turn that into `isError` (`server.py:541`). For Serena, whose
response comes from `json.loads`, this shape is unreachable, but the wrapper
must not depend on the source staying the same.

### 5.1 Why not a bare `dict`

It is tempting to return a `dict` and let the SDK wrap it itself. Rejected:
this fixes objects and does **not** fix arrays, which is exactly the broken
case. Always returning a single `TextContent` covers both shapes equally.

### 5.2 Why not a proto change

The alternative is to replace `string content_json` with `repeated
ContentBlock` in `lsp_service.proto` and carry the original block split
through. Rejected:

- the split is already lost at step 1, so the proto would have to be fixed
  together with the backend — a substantially larger radius. And it is not
  only about size: `_flatten_content` (`serena_backend.py:1120-1126`) does a
  `"".join` over `.text` and **silently drops** any block without that
  attribute. So `repeated ContentBlock` in the proto would carry emptiness
  through until the backend is rewritten — the fix would have to go in the
  reverse order, starting from step 1;
- for symbol navigation it gains nothing: Serena returns text, not images or
  resources;
- `content_json` appears in the contracts of existing tests
  (`test_serena_backend.py:183,197`, `test_grpc_server_metrics.py:62,127`),
  which would have to be rewritten with no benefit to the user.

If non-text blocks ever need to be passed through, a proto change will become
justified. Today it would be work done in advance.

### 5.3 Invariant: "an error must not look like emptiness"

The module declares it in its docstring (`bridge_stdio.py:7-13`): a failure
must arrive as an MCP error, not as a successful empty result, otherwise the
model reads "no matches" and confidently lies.

The fix preserves it: `ToolCallFailed` still propagates as an exception from
`session.call_tool` and is turned by the SDK into an error result. Only the
success path is wrapped.

### 5.4 `None` stays a visible error (correction after review)

The first revision of this document said that `None` from
`bridge_loop.py:215` "remains a success — but with text `""`". **This was
wrong, and a fix based on that wording would have been an invariant
regression.**

Today `None` is not a success. `NoneType` has no `__iter__`, so it falls
into the `else` at `server.py:519` and yields an error result. Verified by
running the SDK's real handler through `build_mcp_server` with a fake
session:

| session return | `isError` | text |
|---|---|---|
| `None` | **True** | `Unexpected return type from tool: NoneType` |
| `[{...}]` | **True** | `13 validation errors for CallToolResult` |
| `{"ok": True}` | False | `{"ok": true}` |

Wrapping `None` in `TextContent("")` would turn a visible error into a
successful empty result — exactly what the docstring of `bridge_stdio.py:7-13`
warns against: the model would read the emptiness as "no matches" and
confidently lie.

Therefore `None` is **not wrapped**: it goes into the SDK as is and remains
`isError`. The decision is reinforced by the fact that `None` is here
indistinguishable from data corruption. `bridge_loop.py:214` returns `None`
for an empty `content_json`, and `_flatten_content`
(`serena_backend.py:1120-1126`) returns `""` in two different cases: when
Serena honestly returned empty content, and when all of its blocks turned out
to be non-text and were **silently dropped** (`join` over `.text` skips a
block without that attribute). The bridge has no means to tell them apart,
and treating possible data loss as "no results" is unacceptable.

Legitimate emptiness looks different and already works:
`get_diagnostics_for_file` returns `{}` — an object, serialized as `"{}"` and
arriving as the success "no diagnostics". An empty array `[]` will arrive
after the fix as `"[]"`, also a success. Both carry the response shape, while
`None` carries nothing.

## 6. Tests

### 6.1 What already exists (correction after review)

The first revision proposed five new tests in a new file and claimed the
existing ones were untouched. That was a mistake: the existing ones should
have been looked at first. `tests/unit_tests/lsp/test_bridge_stdio_server.py`
— 8 tests, all green on `60eb07f` — already covers three of the five:

| proposed | already exists |
|---|---|
| test 2 (object) and test 5 (validity against `CallToolResult`) | `:94 test_successful_call_returns_payload_as_content` |
| test 4 (`ToolCallFailed` does not become a success) | `:108 test_failed_call_is_reported_as_mcp_error_not_empty_result` |

**And here is the main lesson.** Test `:94` passes today only because its
fixture returns `result={"symbols": ["Foo"]}` — a `dict`, i.e. the only shape
that does not break. Swap the curly braces for square ones and it would have
caught the defect from the start. The cause of the defect and the cause of it
going uncaught are the same: both sides looked only at the object.

So "add five new tests" would have given false confidence: three of them
would have duplicated green ones, and the coverage would have looked bigger.

### 6.2 What to do

**Extend the existing file, do not create a second one.** Only two cases are
original — the array and `None`.

1. Parametrize `:94` over return shapes: `dict`, `list`, and — as a separate
   test, because the expectation is the opposite — `None`.
   - `dict` → one `TextContent`, `isError` is not True (guards behavior that
     works today);
   - `list` → one `TextContent`, the symbol's `name_path` is visible in the
     text (regression: without the fix it fails with 13 validation errors);
   - `[]` → `"[]"`, success (legitimate emptiness, §5.4).
2. `None` → `isError is True`. NOT wrapped (§5.4). The test pins fail-loud,
   not an empty success.

The existing `:108` (the error invariant) and `:123`
(`test_arguments_reach_the_session_verbatim`) are untouched — they already
check what is needed and must stay green.

### 6.3 Mutation check — on both axes

Mandatory, and one axis is not enough:

- remove the wrapper → **the array test must fail**;
- with the wrapper removed, **the dict test must keep passing**.

The second proves that the new test catches precisely the array regression
and not "anything at all": a test that fails on any edit does not localize
the defect.

## 7. Change radius

- `deep_worker_cli/lsp/bridge_stdio.py` — the return point + `json` import.
- `tests/unit_tests/lsp/test_bridge_stdio_server.py` — extended per §6.2.

There will be no new test file: `test_bridge_stdio_content.py` from the first
revision is cancelled as a duplicate (§6.1).

Not touched: `bridge_loop.py`, `serena_backend.py`, `lsp_service.proto`, the
gRPC contracts.

## 8. Post-fix verification

A live run of `serena__find_symbol` from a running agent: today it returns an
exception, after the fix it must return symbols. This is the acceptance
check — a unit test on mocks does not replace it.
