"""MCPToolAdapter — wraps an MCP tool entry into an ordinary LangChain BaseTool.

The adapter is the single place where we perform the MCP call "by hand". From
there on, within the graph the tool looks like any other ``BaseTool``: it is
picked up by ``SafeToolCallMiddleware`` (exceptions → ToolMessage(status='error')),
``BoundedToolCallMiddleware`` (2-min wait → job-handle), and the regular HITL
``interrupt_on``.

See DESIGN_EXTERNAL_MCP_TOOLS.md §3.3 and
DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md.

Names:
* ``name`` — provider-safe (``.``/``/`` → ``__``); ToolNode and each provider
  (claude-code-sdk/codex/grok/deepseek/openai/gemini) route by it; satisfies
  ``^[a-zA-Z0-9_-]+$``.
* ``canonical_name`` — the original ``<server>.<tool>`` (for UI/introspection).
* ``_bare_tool_name`` — the bare tool_name from MCP ``list_tools`` (no server
  prefix, no mangling); this is exactly what ``session.call_tool(...)`` receives.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from deep_worker_cli.external_mcp.mangler import MCPToolNameMangler


class _UnsetType:
    """Sentinel for "argument not provided by the caller".

    Needed to distinguish a skipped optional argument from a deliberately
    passed ``None``: the former is not sent to the MCP server at all
    (the server applies its own default), the latter goes as an explicit ``null``.

    IMPORTANT: the sentinel lives ONLY inside the adapter's internal
    pydantic model. It must not appear in the JSON schema the adapter
    hands to the LLM via ``model_json_schema()`` — otherwise pydantic
    would raise ``PydanticJsonSchemaWarning: Default value <UNSET> is not
    JSON serializable`` and the provider would filter out such tools.

    ``__get_pydantic_core_schema__`` declares the sentinel acceptable at
    validation time (otherwise pydantic would reject a field typed
    ``py_type | _UnsetType``), and ``__get_pydantic_json_schema__``
    returns an empty ``{}`` — the field's actual JSON schema is supplied
    entirely by ``__json_schema_extra`` (see ``_schema_to_pydantic``).
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: PLW3201  (pydantic protocol)
        cls, _source_type: Any, _handler: Any
    ) -> CoreSchema:
        # Pass the sentinel through any validation as-is; on the wire to
        # MCP goes the filtered dict (see ``_arun``), but pydantic would
        # still try to serialize ``default=_UNSET`` when building the
        # JSON schema — without this serializer it would raise
        # ``PydanticJsonSchemaWarning: Default value <UNSET> is not JSON
        # serializable``, and the provider would filter out such tools.
        return core_schema.no_info_plain_validator_function(
            lambda v: v,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _v: None
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(  # noqa: PLW3201  (pydantic protocol)
        cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # The part of the field schema coming from the sentinel type is
        # empty: the whole content (type/default/description) comes from
        # the original ``input_schema`` via ``json_schema_extra``.
        return {}


_UNSET = _UnsetType()

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(
    server_name: str, tool_name: str, input_schema: dict[str, Any]
) -> type[BaseModel]:
    """Build a permissive pydantic-v2 args model from a JSON Schema.

    We deliberately do NOT enforce strict validation here — MCP servers
    are the source of truth for argument validity; we only need a schema
    rich enough for LangChain's tool-call machinery to bind.

    JSON-schema invariant (for the LLM):
        Every field in the ``model_json_schema()`` output matches the original
        ``input_schema[prop_name]`` (type, description, default — including the
        ABSENCE of ``default`` if the server did not specify one). The ``_UNSET``
        sentinel is an adapter-runtime detail; it does not flow into the schema.

    Runtime invariant (see ``_arun``):
        unset -> omit the key; explicit None -> send null;
        falsy 0/'' -> send. Enforced by the sentinel.
    """
    fields: dict[str, Any] = {}
    props = (input_schema or {}).get("properties") or {}
    required = set((input_schema or {}).get("required") or [])
    for prop_name, prop_spec in props.items():
        py_type: Any = Any
        prop_spec_dict: dict[str, Any] = {}
        if isinstance(prop_spec, dict):
            prop_spec_dict = prop_spec
            jt = prop_spec.get("type")
            if isinstance(jt, str):
                py_type = _JSON_TO_PY.get(jt, Any)
            description = prop_spec.get("description") or ""
        else:
            description = ""
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            # At runtime the sentinel distinguishes "not provided" from a
            # deliberate None; at validation time it is acceptable thanks to
            # ``_UnsetType.__get_pydantic_core_schema__``.
            #
            # For the LLM-bound JSON schema we replace the field schema
            # with the original ``input_schema[prop_name]``: the server's
            # type, default (or its absence), description — all preserved.
            field_json = dict(prop_spec_dict)

            def _override(_schema: dict[str, Any], _src: dict[str, Any] = field_json) -> None:
                _schema.clear()
                _schema.update(_src)

            # ``Any | _UnsetType`` degenerates into ``Any`` and loses the
            # server's type in the schema — stitch the union only when py_type is set.
            field_type: Any = py_type if py_type is Any else (py_type | _UnsetType)
            fields[prop_name] = (
                field_type,
                Field(
                    default=_UNSET,
                    description=description,
                    json_schema_extra=_override,
                ),
            )

    model_name = f"MCPArgs_{server_name}_{tool_name}"
    # ``extra='allow'`` — MCP servers may also allow other arguments.
    model_cls = create_model(  # type: ignore[call-overload]
        model_name,
        __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
        **fields,
    )
    return model_cls


def _flatten_call_result(result: Any) -> str:
    """Reduce an mcp.types.CallToolResult to a plain string for ToolMessage.

    Real shape (mcp 1.x): ``CallToolResult(content=[TextContent|...], isError: bool)``.
    Our test double mirrors that with ``SimpleNamespace``.
    """
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        # The common shape: ``block.type == 'text'`` with ``block.text``.
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        # Structured / image / resource blocks → JSON dump of the dict form.
        if hasattr(block, "model_dump"):
            parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
        elif isinstance(block, dict):
            parts.append(json.dumps(block, ensure_ascii=False))
        else:
            parts.append(str(block))
    out = "\n".join(parts) if parts else ""
    # If the single block carried a JSON-serialisable object (dict/list/etc.)
    # the tests want it dumped — so re-dump non-string payloads here.
    if len(parts) == 1 and not isinstance(content[0].text, str):  # type: ignore[union-attr]
        return json.dumps(content[0].text, ensure_ascii=False)  # type: ignore[union-attr]
    return out


class MCPToolAdapter(BaseTool):
    """Wrapper around a single MCP tool as a LangChain BaseTool."""

    # pydantic v2 on BaseTool — public fields (defaults to satisfy strict override checks)  # noqa: E501
    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] | dict[str, Any] | None = None

    # private: session + bare tool name + canonical (for UI/wire)
    _session: Any = None
    _bare_tool_name: str = ""
    _canonical_name: str = ""

    def __init__(
        self,
        *,
        session: Any,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        mangler: MCPToolNameMangler | None = None,
    ) -> None:
        canonical = f"{server_name}.{tool_name}"
        # If the caller passed a shared per-agent mangler, register the
        # canonical name in it (collisions between different tools of one
        # agent are caught there too). Otherwise create a local one so the
        # adapter stays self-contained (stand-alone tests).
        m = mangler if mangler is not None else MCPToolNameMangler()
        mangled_name = m.add(canonical)
        args_model = _schema_to_pydantic(server_name, tool_name, input_schema or {})
        super().__init__(
            name=mangled_name,
            description=description or f"MCP tool {canonical}",
            args_schema=args_model,
        )
        # bypass pydantic's frozen init for private fields
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_bare_tool_name", tool_name)
        object.__setattr__(self, "_canonical_name", canonical)

    @property
    def canonical_name(self) -> str:
        """Canonical MCP name ``<server>.<tool>`` (with dots/slashes)
        for the UI/catalog. Source of truth for display; ``name`` (mangled)
        is used for routing inside the graph and by providers."""
        return self._canonical_name

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        # All calls from ToolNode go through the async path (LangGraph will
        # pick ainvoke for a tool with _arun). The sync path is kept only as
        # a safety net.
        raise NotImplementedError(
            "MCPToolAdapter is async-only; the graph should call _arun()."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        # LangChain sometimes passes config as the first positional arg; we
        # ignore it (validation already happened in args_schema, the rest is
        # up to the MCP server).
        sess = self._session
        # Drop unset optional arguments: the JSON Schema of many MCP
        # tools declares them strictly typed (``{"type": "integer",
        # "default": -1}``), so an explicit ``null`` is a validation
        # error, while omitting the key yields the server default.
        call_args = {k: v for k, v in kwargs.items() if not isinstance(v, _UnsetType)}
        result = await sess.call_tool(self._bare_tool_name, call_args)
        if getattr(result, "isError", False):
            text = _flatten_call_result(result) or "MCP tool returned isError=True"
            raise RuntimeError(text)
        return _flatten_call_result(result)
