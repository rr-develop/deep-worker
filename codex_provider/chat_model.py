"""CodexChatModel — LangChain BaseChatModel that drives Codex CLI.

This module is the public surface of the Codex provider. Same shape as
``claude_code_provider/chat_model.py:ClaudeCodeChatModel`` — only
``ainvoke`` is real; ``_generate`` raises (async-only); LangChain
streaming methods are not implemented.

See ``docs/DESIGN_CODEX_PROVIDER.md`` for the design rationale.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Callable, Sequence

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, PrivateAttr

from codex_provider.client_state import (
    CodexClientState,
    get_global_client_state,
)
from codex_provider.custom_types import MCPResult, ToolMetadata
from codex_provider.exec_runner import (
    AgentMessageItem,
    CodexExecRunner,
    ExecArgs,
    ExecHandle,
    GenericItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from codex_provider.proxy_mcp_server import (
    ProxyMCPServer,
    get_global_proxy_mcp_server,
)


# Codex built-in deny-list (Q5). At minimum the Codex counterparts of
# every default deep-worker tool, so Codex never bypasses our tools.
# Format: name → ``--config tools.<name>=false``.
#
# This list is intentionally conservative — we'd rather over-disable
# than under-disable. A future Codex addition will not break us because
# unknown config keys are silently ignored by the CLI.
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = (
    # Filesystem
    "read",
    "write",
    "edit",
    "apply_patch",
    "glob",
    "grep",
    # Shell
    "shell",
    "exec",
    "bash",
    # Web
    "web_search",
    "web_fetch",
    "fetch",
    # Agent infra
    "memory",
    "todo_list",
    "task",
)


class CodexChatModel(BaseChatModel):
    """LangChain chat-model wrapper around Codex CLI.

    Provider name in deep-worker: ``"codex"``. Construction is cheap and
    side-effect-free; the proxy MCP server is started lazily on the
    first ``ainvoke`` (or on the first ``bind_tools`` if it carries
    tools).
    """

    # Pydantic config identical to claude-code:
    # - arbitrary_types_allowed for non-Pydantic privates
    # - extra='allow' so LangChain can pass through model/max_tokens/etc.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _client_state: Any = PrivateAttr(default=None)
    _exec_runner: Any = PrivateAttr(default=None)
    _proxy_mcp: Any = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    _working_dir: str | None = PrivateAttr(default=None)
    _model_name: str | None = PrivateAttr(default=None)
    _disallowed_tools: tuple[str, ...] = PrivateAttr(default=DEFAULT_DISALLOWED_TOOLS)
    _sandbox_mode: str = PrivateAttr(default="workspace-write")
    _approval_policy: str = PrivateAttr(default="never")
    _proxy_started: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------
    # Pydantic schema overrides — same surgery as claude_code_provider:
    # LangGraph pickles models during checkpointing, and BaseChatModel
    # carries callable fields that can't be JSON-serialised. Strip them.
    # ------------------------------------------------------------------
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        try:
            parent = handler(core_schema)
        except Exception:
            parent = {"type": "object", "properties": {}}
        props: dict[str, Any] = parent.get("properties", {}) or {}
        parent["properties"] = {
            k: v for k, v in props.items()
            if k not in {"custom_get_token_ids", "callbacks", "rate_limiter"}
        }
        parent["title"] = cls.__name__
        parent["description"] = "Codex chat model for LangChain"
        return parent

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude: set[str] = {"custom_get_token_ids", "callbacks", "rate_limiter"}
        user_exclude = kwargs.get("exclude")
        if isinstance(user_exclude, set):
            kwargs["exclude"] = exclude | user_exclude
        elif user_exclude is None:
            kwargs["exclude"] = exclude
        return super().model_dump(**kwargs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        sandbox_mode: str = "workspace-write",
        approval_policy: str = "never",
        disallowed_tools: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseChatModel has a Callable field that breaks JSON-schema
        # generation; set it to None up front.
        kwargs["custom_get_token_ids"] = None
        super().__init__(**kwargs)

        self._client_state = get_global_client_state()
        self._exec_runner = CodexExecRunner()
        self._tools_metadata = []
        self._model_name = model
        self._sandbox_mode = sandbox_mode
        self._approval_policy = approval_policy
        if disallowed_tools is not None:
            self._disallowed_tools = tuple(disallowed_tools)

    @property
    def _llm_type(self) -> str:
        return "codex"

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "CodexChatModel":
        """Extract MCP-compatible metadata from LangChain tools and bind it.

        Returns self with the metadata populated (LangChain pattern
        allows returning self or a clone; claude-code returns a clone,
        we return self to avoid duplicating the proxy MCP server).
        The proxy MCP server is materialised lazily on first ainvoke
        so that bind_tools stays side-effect free.
        """
        tools_metadata: list[ToolMetadata] = []
        for item in tools:
            if not isinstance(item, BaseTool):
                continue
            schema = self._extract_tool_schema(item)
            tools_metadata.append({
                "name": item.name,
                "description": item.description or "",
                "input_schema": schema,
            })
        self._tools_metadata = tools_metadata
        # Drop any previously-started proxy — its tool set is stale.
        # Actual restart happens lazily on next ainvoke.
        return self

    @staticmethod
    def _extract_tool_schema(tool: BaseTool) -> dict[str, Any]:
        """Same algorithm as claude_code_provider.chat_model.bind_tools."""
        schema: dict[str, Any] = {}
        if tool.args_schema:
            try:
                if hasattr(tool.args_schema, "model_json_schema"):
                    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
                elif hasattr(tool.args_schema, "schema"):
                    schema = tool.args_schema.schema()  # type: ignore[union-attr,deprecated]
            except Exception:
                schema = {}

        if not schema.get("properties"):
            func = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
            if func:
                try:
                    sig = inspect.signature(func)
                    props: dict[str, Any] = {}
                    required: list[str] = []
                    for pname, param in sig.parameters.items():
                        if pname in ("runtime", "self", "kwargs"):
                            continue
                        ann = param.annotation
                        if ann is int:
                            ptype = "integer"
                        elif ann is bool:
                            ptype = "boolean"
                        elif ann is float:
                            ptype = "number"
                        else:
                            ptype = "string"
                        prop: dict[str, Any] = {
                            "type": ptype,
                            "title": pname.replace("_", " ").title(),
                        }
                        if hasattr(ann, "__metadata__"):
                            for meta in ann.__metadata__:
                                if isinstance(meta, str):
                                    prop["description"] = meta
                                    break
                        if param.default != inspect.Parameter.empty:
                            prop["default"] = param.default
                        else:
                            required.append(pname)
                        props[pname] = prop
                    if props:
                        schema = {
                            "type": "object",
                            "properties": props,
                            "required": required,
                        }
                except (ValueError, TypeError):
                    pass
        return schema

    # ------------------------------------------------------------------
    # Async-only — _generate raises, _agenerate not provided, ainvoke is real.
    # ------------------------------------------------------------------
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "CodexChatModel only supports async operations. Use ainvoke()."
        )

    async def ainvoke(  # type: ignore[override]
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Drive one Codex turn (or resume one mid-turn after a tool call)."""
        from langchain_core.prompt_values import PromptValue
        from langchain_core.runnables.config import ensure_config

        # 1. Normalize input
        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)  # type: ignore[arg-type]

        # 2. Mandatory agent_thread_id (the deep-worker contract).
        config = ensure_config(config)
        cfg = config.get("configurable") or {}
        if "thread_id" not in cfg:
            raise ValueError(
                "CodexChatModel requires explicit thread_id in config — "
                "this is deep-worker's agent name. Each caller (main "
                "agent, TaskManager, subagent) must own its thread_id. "
                "Implicit defaults are forbidden."
            )
        agent_thread_id: str = cfg["thread_id"]

        # 3. Per-agent_thread_id state
        state = self._client_state.get_or_create(agent_thread_id)

        # 4. Summarization detection — drop Codex session, keep agent_thread_id.
        if state.sent_count > len(messages):
            self._client_state.record_summarization(agent_thread_id)
            # current_exec, if any, was for the OLD session — drop it too
            if state.current_exec is not None:
                with _suppress_errors():
                    await state.current_exec.aclose()
            state.current_exec = None
            # record_summarization mutated state.codex_session_uuid and
            # state.sent_count in place; re-fetch for clarity:
            state = self._client_state.get_or_create(agent_thread_id)

        # 5. System prompt (first SystemMessage in messages)
        system_prompt: str | None = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
                break

        # 6. Ensure proxy MCP server is started (lazy, idempotent)
        await self._ensure_proxy_started()

        # 7. Tail delta only — incremental mirror, never full rebuild
        new_messages = messages[state.sent_count:]
        user_text_parts: list[str] = []
        tool_resolved = False
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                self._proxy_mcp.resolve_call(
                    msg.tool_call_id,
                    {"content": _build_mcp_content(msg.content)},
                )
                tool_resolved = True
            elif isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                user_text_parts.append(content)
            elif isinstance(msg, AIMessage):
                # already in Codex's session — don't re-send
                pass
            elif isinstance(msg, SystemMessage):
                # forwarded via base-instructions, not as a user message
                pass

        # 8. Update sent_count BEFORE awaiting response (mirror claude-code)
        state.sent_count = len(messages)

        # 9. Mid-turn (we're resolving a previously-emitted tool_call)
        #    vs new turn (spawn fresh codex exec).
        if tool_resolved and state.current_exec is not None and not user_text_parts:
            events_iter = state.current_exec.events
            handle: ExecHandle | Any = state.current_exec
        else:
            # If there's an old current_exec from a previous interrupted
            # turn (e.g. main user message arrived before tool resolution),
            # close it cleanly to avoid an orphaned subprocess.
            if state.current_exec is not None:
                with _suppress_errors():
                    await state.current_exec.aclose()
                state.current_exec = None

            prompt = "\n\n".join(user_text_parts) if user_text_parts else ""
            args = ExecArgs(
                prompt=prompt,
                session_uuid=state.codex_session_uuid,
                working_directory=self._working_dir,
                sandbox_mode=self._sandbox_mode,
                approval_policy=self._approval_policy,
                model=self._model_name,
                base_instructions=system_prompt,
                disallowed_tools=self._disallowed_tools,
            )
            handle = await self._exec_runner.spawn(args)
            state.current_exec = handle
            events_iter = handle.events

        # 10. Consume events — event-driven, no polling.
        final_text = ""
        pending_tool_calls: list[dict[str, Any]] = []
        turn_done = False
        try:
            async for event in events_iter:
                if isinstance(event, ThreadStartedEvent):
                    if state.codex_session_uuid is None:
                        state.codex_session_uuid = event.thread_id
                elif isinstance(event, ItemStartedEvent):
                    # ``item.started`` with ``status="in_progress"`` for an
                    # ``mcp_tool_call`` on OUR proxy is the earliest signal
                    # that Codex is calling our tool. We wait (event-driven)
                    # for the in-process HTTP handler to actually receive
                    # the ``tools/call`` request and park its Future —
                    # those are two independent channels (stdout vs HTTP)
                    # and either can land first.
                    #
                    # We do NOT touch other servers' tool calls — Codex
                    # handles them internally.
                    it = event.item
                    if (
                        isinstance(it, McpToolCallItem)
                        and it.server == self._proxy_mcp.server_name
                        and it.status == "in_progress"
                    ):
                        # Wait up to 30 s for the matching tools/call to
                        # land at our HTTP handler. Each tick is ``await
                        # asyncio.sleep(0)`` which is the standard
                        # "yield control" idiom in asyncio — NOT polling
                        # in the CPU-busy sense; the loop wakes us only
                        # when something else runs. We bound the wait so
                        # a never-arriving call doesn't deadlock the turn.
                        deadline = asyncio.get_event_loop().time() + 30.0
                        while not self._proxy_mcp.has_pending_calls():
                            if asyncio.get_event_loop().time() > deadline:
                                break
                            # Sleep briefly so other tasks (the HTTP
                            # handler) get scheduled. 5 ms is a balance:
                            # short enough to be responsive, long enough
                            # to avoid burning CPU.
                            await asyncio.sleep(0.005)
                        if self._proxy_mcp.has_pending_calls():
                            pending_tool_calls = [
                                {"id": pc["id"], "name": pc["name"], "args": pc["args"]}
                                for pc in self._proxy_mcp.get_pending_calls()
                            ]
                            break
                elif isinstance(event, ItemCompletedEvent):
                    it = event.item
                    if isinstance(it, AgentMessageItem):
                        final_text = it.text
                    elif isinstance(it, McpToolCallItem):
                        # Tool call already completed (resolved by us or
                        # cancelled/failed by Codex). Nothing more for us
                        # to do here — pending_calls was cleared when the
                        # handler returned. If the call FAILED, Codex
                        # likely emits a turn.failed or its agent_message
                        # explaining the failure next; we surface that
                        # to the user.
                        pass
                    elif isinstance(it, GenericItem):
                        # reasoning / file_change / web_search / etc.
                        # No deep-worker action; ignore.
                        pass
                elif isinstance(event, TurnCompletedEvent):
                    turn_done = True
                    break
                elif isinstance(event, TurnFailedEvent):
                    turn_done = True
                    raise RuntimeError(f"Codex turn failed: {event.error_message}")
        finally:
            if turn_done:
                with _suppress_errors():
                    await handle.aclose()
                state.current_exec = None

        if pending_tool_calls:
            return AIMessage(content=final_text, tool_calls=pending_tool_calls)
        return AIMessage(content=final_text)

    # ------------------------------------------------------------------
    # Proxy MCP lifecycle
    # ------------------------------------------------------------------
    async def _ensure_proxy_started(self) -> None:
        # Process-global singleton (one proxy MCP server per agent
        # process). Tests may inject a custom ``_proxy_mcp`` — don't
        # clobber that.
        #
        # Skip materialisation entirely if we have no tools to expose:
        # Codex caches ``tools/list`` per MCP-server registration, so
        # if we register an empty server and *later* call ``bind_tools``
        # with tools, Codex won't know about them. By deferring
        # registration until we actually have tools, the first
        # ``codex exec`` after ``bind_tools`` sees the right list.
        if self._proxy_mcp is None:
            if not self._tools_metadata:
                return
            self._proxy_mcp = get_global_proxy_mcp_server(self._tools_metadata)
        else:
            if hasattr(self._proxy_mcp, "update_tools"):
                try:
                    self._proxy_mcp.update_tools(self._tools_metadata)
                except Exception:
                    pass
        if not self._proxy_started:
            await self._proxy_mcp.start()
            await _register_proxy_with_codex(
                self._proxy_mcp.server_name, self._proxy_mcp.port
            )
            self._proxy_started = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_mcp_content(content: Any) -> list[dict[str, Any]]:
    """Convert a LangChain ToolMessage content into MCP content blocks.

    Same shape as ``claude_code_provider.chat_model._build_mcp_content``.
    """
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks: list[dict[str, Any]] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            blocks.append({"type": "text", "text": str(b.get("text", ""))})
        else:
            blocks.append({"type": "text", "text": str(b)})
    return blocks


class _suppress_errors:
    """Context manager that swallows any exception during cleanup paths.

    Used in the few places where we must drain a generator / kill a
    subprocess and any exception there is strictly worse than swallowing
    it (the agent must continue serving other turns).
    """

    def __enter__(self) -> "_suppress_errors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True  # suppress

    async def __aenter__(self) -> "_suppress_errors":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True


async def _register_proxy_with_codex(server_name: str, port: int) -> None:
    """Run ``codex mcp add <name> --url http://127.0.0.1:<port>``.

    Idempotent in spirit: we first try ``codex mcp remove <name>`` (silent
    on failure) to clear any stale entry from a previous crashed agent,
    then add fresh.
    """
    # Defensive: refuse to register garbage. Codex demands server_name
    # match [A-Za-z0-9_-]; an integer port. If a test mistakenly passed
    # a MagicMock, fail here instead of polluting ~/.codex/config.toml.
    import re
    if not isinstance(server_name, str) or not re.match(r"^[A-Za-z0-9_-]+$", server_name):
        raise ValueError(f"invalid MCP server_name: {server_name!r}")
    if not isinstance(port, int) or port <= 0:
        raise ValueError(f"invalid MCP port: {port!r}")
    url = f"http://127.0.0.1:{port}/mcp"
    # Remove first (silent)
    rm = await asyncio.create_subprocess_exec(
        "codex", "mcp", "remove", server_name,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await rm.wait()
    add = await asyncio.create_subprocess_exec(
        "codex", "mcp", "add", server_name, "--url", url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await add.communicate()
    if add.returncode != 0:
        raise RuntimeError(
            f"`codex mcp add {server_name} --url {url}` failed "
            f"(exit {add.returncode}): {err.decode(errors='replace')}"
        )
