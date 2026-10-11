"""MCPToolAdapter — a wrapper over a single MCP tool record.

Contract (see DESIGN_EXTERNAL_MCP_TOOLS.md §3.3 and
DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md):

* ``name`` = **provider-safe** name (`.` and `/` replaced with `__`),
  the final name by which ToolNode and all providers route.
* ``canonical_name`` = the original MCP name ``<server>.<tool>`` (with dots/
  slashes) — the source of truth for the UI and the MCP-wire call.
* ``description`` is taken from MCP ``tool.description``.
* ``_arun`` proxies into ``session.call_tool(tool_name, kwargs)`` —
  with ``_bare_tool_name`` (the original bare name on the MCP-server side, without
  the server prefix and without any mangling).
* Result: if MCP returned a structure — it is serialized into a JSON string,
  if plain text — returned as-is.

This unit test does NOT raise a real MCP — it only mocks ``ClientSession``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
from langchain.tools.tool_node import ToolCallRequest, ToolRuntime
from langchain_core.messages import ToolMessage

from deep_worker_cli.external_mcp.adapter import MCPToolAdapter


class _FakeSession:
    """Imitates exactly the slice of mcp.ClientSession the adapter uses:
    one ``call_tool(name, args)`` method returning an object with
    ``content`` and ``isError`` attributes."""

    def __init__(self, payload: Any, is_error: bool = False) -> None:
        self.payload = payload
        self.is_error = is_error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        # The MCP SDK returns CallToolResult with .content: list[TextContent|...]
        # and .isError. We imitate the minimal shape.
        text_block = SimpleNamespace(type="text", text=self.payload)
        return SimpleNamespace(content=[text_block], isError=self.is_error)


@pytest.mark.asyncio
async def test_adapter_name_is_mangled_canonical_preserved() -> None:
    """``name`` is provider-safe (mangled), ``canonical_name`` is the original."""
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="github",
        tool_name="create_issue",
        description="Create a GitHub issue",
        input_schema={"type": "object", "properties": {}},
    )
    # `.` in `<server>.<tool>` → `__` in the provider-safe name.
    assert a.name == "github__create_issue"
    assert a.canonical_name == "github.create_issue"


@pytest.mark.asyncio
async def test_adapter_name_mangled_for_dotted_tool_name() -> None:
    """Real scenario from a chrome bridge: a tool with a dot in its name
    (`browser.list_tabs`) → mangled `chrome__browser__list_tabs`,
    canonical — `chrome.browser.list_tabs`."""
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="chrome",
        tool_name="browser.list_tabs",
        description="",
        input_schema={"type": "object", "properties": {}},
    )
    assert a.name == "chrome__browser__list_tabs"
    assert a.canonical_name == "chrome.browser.list_tabs"


@pytest.mark.asyncio
async def test_adapter_description_propagated() -> None:
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x",
        tool_name="y",
        description="Hello from MCP",
        input_schema={"type": "object", "properties": {}},
    )
    assert a.description == "Hello from MCP"


@pytest.mark.asyncio
async def test_adapter_uses_provided_mangler_for_name_only() -> None:
    """When the caller passes a ``mangler``, the adapter registers its
    canonical name in it and takes the name from there. The wire call always goes by the
    bare tool_name (without the server prefix, without mangling)."""
    from deep_worker_cli.external_mcp.mangler import MCPToolNameMangler

    mangler = MCPToolNameMangler()
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="chrome",
        tool_name="browser.status",
        description="",
        input_schema={"type": "object", "properties": {}},
        mangler=mangler,
    )
    assert a.name == "chrome__browser__status"
    assert a.canonical_name == "chrome.browser.status"
    # The mangler really received the canonical.
    assert mangler.to_canonical("chrome__browser__status") == "chrome.browser.status"
    # The wire call goes by the bare tool_name, without the server prefix and without mangling.
    await a._arun()
    assert sess.calls == [("browser.status", {})]


@pytest.mark.asyncio
async def test_adapter_arun_forwards_kwargs_to_call_tool() -> None:
    sess = _FakeSession("done")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="github",
        tool_name="create_issue",
        description="",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title"],
        },
    )
    out = await a._arun(title="hi", body="there")
    assert "done" in out
    # The adapter must proxy EXACTLY under the bare tool_name, without the prefix.
    assert sess.calls == [("create_issue", {"title": "hi", "body": "there"})]


@pytest.mark.asyncio
async def test_adapter_serialises_structured_payload() -> None:
    sess = _FakeSession({"id": 42, "url": "https://x/42"})
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x", tool_name="y",
        description="", input_schema={"type": "object", "properties": {}},
    )
    out = await a._arun()
    # A structured payload must come back as a JSON string.
    assert json.loads(out) == {"id": 42, "url": "https://x/42"}


@pytest.mark.asyncio
async def test_adapter_is_error_surfaced_as_exception() -> None:
    sess = _FakeSession("boom", is_error=True)
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x", tool_name="y",
        description="", input_schema={"type": "object", "properties": {}},
    )
    # SafeToolCallMiddleware will turn RuntimeError into ToolMessage(status="error").
    with pytest.raises(RuntimeError):
        await a._arun()


@pytest.mark.asyncio
async def test_adapter_slow_call_uses_existing_bounded_middleware() -> None:
    started = asyncio.Event()

    class _SlowSession:
        async def call_tool(self, name: str, args: dict) -> Any:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    adapter = MCPToolAdapter(
        session=_SlowSession(),
        server_name="slow",
        tool_name="wait",
        description="wait forever",
        input_schema={"type": "object", "properties": {}},
    )
    state = {"messages": []}
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id="toolu_MCP",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "toolu_MCP", "name": adapter.name, "args": {}},
        state=state,
        tool=adapter,
    )

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        result = await adapter._arun()
        return ToolMessage(content=result, tool_call_id="toolu_MCP")

    middleware = BoundedToolCallMiddleware()
    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"seconds": 0.0}
    loop.time = lambda: real_time() + skew["seconds"]  # type: ignore[method-assign]
    try:
        task = asyncio.create_task(
            middleware.awrap_tool_call(request, handler)
        )
        await started.wait()
        skew["seconds"] = middleware.wait_seconds + 1
        await asyncio.sleep(0.02)
        result = await task
        assert isinstance(result, ToolMessage)
        assert result.status != "error"
        assert "in_progress" in str(result.content)
        assert middleware.get_job_registry()
    finally:
        loop.time = real_time  # type: ignore[method-assign]
        for job in middleware.get_job_registry().values():
            job.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job.task

# ---------------------------------------------------------------------------
# Skipping unset optional arguments (regression: Serena/strict-typed MCP)
#
# Unset optional fields MUST NOT be sent to the server as an explicit JSON ``null``:
# in JSON Schema ``{"type": "integer", "default": -1}`` the type is NOT nullable,
# so ``null`` is a validation error, while omitting the key yields the server-side
# default. Measured on live Serena 1.6.1: null -> ERROR, omission -> OK (12 of 22
# tools have strictly-typed optional fields and would break on ``null``).
# ---------------------------------------------------------------------------

_STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {"type": "string"},
        "depth": {"type": "integer", "default": -1},
        "max_answer_chars": {"type": "integer", "default": -1},
    },
    "required": ["relative_path"],
}


def _adapter_with_strict_schema(session: Any) -> MCPToolAdapter:
    return MCPToolAdapter(
        session=session,
        server_name="serena",
        tool_name="get_symbols_overview",
        description="overview",
        input_schema=_STRICT_SCHEMA,
    )


@pytest.mark.asyncio
async def test_unset_optional_args_are_not_sent_as_null() -> None:
    """Unset optional fields must not go to the server as ``null``."""
    session = _FakeSession("ok")
    tool = _adapter_with_strict_schema(session)

    await tool.ainvoke({"relative_path": "a/b.py"})

    sent = session.calls[-1][1]
    assert sent == {"relative_path": "a/b.py"}, (
        f"unset optional fields leaked to the server: {sent}"
    )


@pytest.mark.asyncio
async def test_explicitly_passed_none_is_preserved() -> None:
    """An explicitly passed ``None`` is a deliberate argument — we do not drop it."""
    session = _FakeSession("ok")
    tool = _adapter_with_strict_schema(session)

    await tool.ainvoke({"relative_path": "a/b.py", "depth": None})

    sent = session.calls[-1][1]
    assert "depth" in sent
    assert sent["depth"] is None
    assert "max_answer_chars" not in sent


@pytest.mark.asyncio
async def test_explicit_falsy_values_are_sent() -> None:
    """Falsy values (0, "", False) are not the same as "not set"."""
    session = _FakeSession("ok")
    tool = _adapter_with_strict_schema(session)

    await tool.ainvoke(
        {"relative_path": "a/b.py", "depth": 0, "max_answer_chars": 0}
    )

    sent = session.calls[-1][1]
    assert sent == {"relative_path": "a/b.py", "depth": 0, "max_answer_chars": 0}


# ---------------------------------------------------------------------------
# Regression: the `_UNSET` sentinel must not leak into the JSON schema
# that the adapter hands to the LLM. In a live run, pydantic raised
# PydanticJsonSchemaWarning ("Default value <UNSET> is not JSON
# serializable"), which made the claude-code-sdk provider filter out such tools
# and made Serena invisible in bind_tools proxying.
#
# Contract (strict):
#   * `model_json_schema()` MUST NOT raise warnings;
#   * the type of an optional field in the JSON schema is the one declared by
#     the server (integer, not Any/null);
#   * if the server set a `default`, it MUST appear in the schema exactly as
#     in input_schema; if it did not, there must be no `default` key in the
#     schema (we do not substitute -1/None ourselves);
#   * the required set is preserved.
# ---------------------------------------------------------------------------

_MIXED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {"type": "string"},
        # Optional WITH its own default — must flow into the JSON schema as-is.
        "depth": {"type": "integer", "default": -1},
        # Optional WITHOUT a default — there must be no `default` key.
        "include_kinds": {"type": "array"},
        # Nullable optional without a default.
        "workspace_root": {"type": "string"},
    },
    "required": ["relative_path"],
}


def _build_model_for_mixed() -> type:
    return MCPToolAdapter(
        session=_FakeSession("ok"),
        server_name="serena",
        tool_name="mixed_tool",
        description="d",
        input_schema=_MIXED_SCHEMA,
    ).args_schema  # type: ignore[return-value]


def test_json_schema_does_not_warn_about_unset_sentinel() -> None:
    """Live repro: the provider put the tool's JSON schema into the LLM request
    and crashed on a pydantic warning about a non-serializable ``<UNSET>``."""
    import warnings as _w

    model_cls = _build_model_for_mixed()
    with _w.catch_warnings():
        _w.simplefilter("error")
        model_cls.model_json_schema()  # must not raise


def test_json_schema_preserves_server_default_when_present() -> None:
    """If the server set a ``default``, it must be in the JSON schema exactly as
    in input_schema; the agent relies on that value."""
    model_cls = _build_model_for_mixed()
    js = model_cls.model_json_schema()
    depth = js["properties"]["depth"]
    assert depth.get("default") == -1, depth
    # The sentinel must not surface under another name.
    assert "UNSET" not in json.dumps(js), js


def test_json_schema_omits_default_when_server_did_not_set_one() -> None:
    """Rule: do not substitute your own -1/None if the server did
    not specify a default — either the key is absent, or an explicit None (but
    NOT the sentinel and NOT an invented literal)."""
    model_cls = _build_model_for_mixed()
    js = model_cls.model_json_schema()
    include_kinds = js["properties"]["include_kinds"]
    # Allow two variants: no key, or default=None (nullable).
    if "default" in include_kinds:
        assert include_kinds["default"] is None, include_kinds


def test_json_schema_preserves_server_type_for_optional_fields() -> None:
    """The type of an optional field in the LLM schema must match the server's
    type, otherwise the LLM generates invalid calls (this was the ``Any`` trap)."""
    model_cls = _build_model_for_mixed()
    js = model_cls.model_json_schema()
    # Check each optional field with an explicitly declared type.
    for name, spec in _MIXED_SCHEMA["properties"].items():
        want = spec.get("type")
        if not want:
            continue
        got = js["properties"][name]
        # pydantic may wrap in anyOf for nullable — read both places.
        types = {got.get("type")} | {
            v.get("type") for v in got.get("anyOf") or []
        }
        types.discard(None)
        assert want in types, (
            f"field {name!r}: server declared {want!r}, "
            f"in the JSON schema {sorted(types)!r} (full spec: {got})"
        )


def test_field_python_type_matches_server_type_not_any() -> None:
    """Safety-net contract: the python type of a field in the internal pydantic
    model is a union of the server type and the sentinel, NOT ``Any``. This
    protects the LLM JSON schema even if ``json_schema_extra`` ever breaks or is
    removed (a "field became Any" regression would then surface here)."""
    model_cls = _build_model_for_mixed()
    field = model_cls.model_fields["depth"]
    # Union[int, _UnsetType] — both must be present; goal — NOT Any.
    ann = field.annotation
    ann_repr = str(ann)
    assert "int" in ann_repr, ann_repr
    assert "_UnsetType" in ann_repr, ann_repr
    assert "Any" not in ann_repr, (
        f"optional field kept the server type but became Any: {ann_repr}"
    )


def test_json_schema_required_matches_input_schema() -> None:
    """The required set is the server's contract; do not extend or shrink it."""
    model_cls = _build_model_for_mixed()
    js = model_cls.model_json_schema()
    assert set(js.get("required") or []) == set(_MIXED_SCHEMA["required"])


# ---------------------------------------------------------------------------
# Lock: `_arun` MUST NOT pass None values onto the MCP wire for any set of
# caller-supplied arguments. This guards against a hidden landmine: if someone
# later adds an intermediate step in `_arun` like
# ``args_schema.model_validate(kwargs).model_dump()`` (e.g. to "normalize via
# the model"), then the `_UNSET` sentinel turns into `None` (per our
# core_schema serializer) and `None` goes to the server as
# an explicit JSON `null` — breaking the "optional fields are fully omittable"
# UX contract in the sneakiest way (silently, without a client-side error).
#
# The test deliberately catches exactly that regression, not any valid input.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_never_forwards_none_for_unset_optionals() -> None:
    """No LangChain chain must lead to sending ``None`` for an unset optional
    field."""
    session = _FakeSession("ok")
    tool = MCPToolAdapter(
        session=session,
        server_name="s",
        tool_name="probe",
        description="d",
        input_schema={
            "type": "object",
            "properties": {
                "req": {"type": "string"},
                "opt_int": {"type": "integer", "default": -1},
                "opt_arr": {"type": "array", "default": []},
                "opt_bool": {"type": "boolean", "default": False},
            },
            "required": ["req"],
        },
    )
    # Go through the full public agent path (not through _arun directly)
    await tool.ainvoke({"req": "X"})
    sent = session.calls[-1][1]
    # No None values in the arguments sent on the wire.
    nulls = [k for k, v in sent.items() if v is None]
    assert not nulls, (
        f"UX contract broken: unset optional fields went out as null "
        f"(likely model_dump without exclude_unset was added in _arun). "
        f"Fields: {nulls}, full args: {sent}"
    )
    # And the minimal contract: exactly what the caller passed goes on the wire.
    assert sent == {"req": "X"}, sent
