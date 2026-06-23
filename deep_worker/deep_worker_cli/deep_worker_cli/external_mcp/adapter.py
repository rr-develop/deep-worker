"""MCPToolAdapter — turns an MCP tool record into an ordinary LangChain BaseTool.

The adapter is the only place where we make the MCP call "by hand". Further
down the graph the tool looks like any other ``BaseTool``: it is picked up by
``SafeToolCallMiddleware`` (exceptions → ToolMessage(status='error')),
``BoundedToolCallMiddleware`` (2-min wait → job-handle), and ordinary HITL
``interrupt_on``.

See DESIGN_EXTERNAL_MCP_TOOLS.md §3.3.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

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
    are the source of truth for argument validity; we only need a
    schema rich enough for LangChain's tool-call machinery to bind. So
    every property is typed loosely (its JSON-type or ``Any``), and
    nothing extra is rejected.
    """
    fields: dict[str, Any] = {}
    props = (input_schema or {}).get("properties") or {}
    required = set((input_schema or {}).get("required") or [])
    for prop_name, prop_spec in props.items():
        py_type: Any = Any
        if isinstance(prop_spec, dict):
            jt = prop_spec.get("type")
            if isinstance(jt, str):
                py_type = _JSON_TO_PY.get(jt, Any)
            description = prop_spec.get("description") or ""
        else:
            description = ""
        default = ... if prop_name in required else None
        fields[prop_name] = (py_type, Field(default=default, description=description))

    model_name = f"MCPArgs_{server_name}_{tool_name}"
    # ``extra='allow'`` — MCP servers may permit additional arguments.
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

    # pydantic v2 on BaseTool — public fields
    name: str
    description: str
    args_schema: type[BaseModel]

    # private: session + bare tool name
    _session: Any = None
    _bare_tool_name: str = ""

    def __init__(
        self,
        *,
        session: Any,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        args_model = _schema_to_pydantic(server_name, tool_name, input_schema or {})
        super().__init__(
            name=f"mcp__{server_name}__{tool_name}",
            description=description or f"MCP tool {server_name}.{tool_name}",
            args_schema=args_model,
        )
        # bypass pydantic-frozen-init for private fields
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_bare_tool_name", tool_name)

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        # All calls from ToolNode go the async way (LangGraph will choose
        # ainvoke for a tool with _arun). The sync path is kept only as a
        # safety net.
        raise NotImplementedError(
            "MCPToolAdapter is async-only; the graph should call _arun()."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        # LangChain sometimes passes config as the first positional argument;
        # we ignore it (validation already happened in args_schema, the rest
        # is up to the MCP server).
        sess = self._session
        result = await sess.call_tool(self._bare_tool_name, kwargs)
        if getattr(result, "isError", False):
            text = _flatten_call_result(result) or "MCP tool returned isError=True"
            raise RuntimeError(text)
        return _flatten_call_result(result)
