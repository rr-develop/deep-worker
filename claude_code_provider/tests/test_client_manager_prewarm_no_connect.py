"""Regression: ``_prewarm_client_manager`` MUST NOT open the Claude CLI
session — otherwise history-restore on the first ``ainvoke`` is silently
skipped and the agent forgets its conversation.

Background — a related internal incident analysis (not included here):
the agent "does not see its own history"
==================================================================

Commit ``9c58d31`` introduced ``agent_process._prewarm_client_manager``
to deterministically seed the ``ClientManager`` cache with the FINAL
tool inventory before any ``ainvoke`` could run (a related internal
incident fix, not included here). The prewarm called
``client_manager.get_or_create_client(thread_id, tools_metadata,
working_dir)`` — i.e. it actually **opened a Claude CLI subprocess and
an MCP session** with ``resume_session_id=None``.

That side-effect collided with the history-restore design
(``DESIGN_HISTORY_RESTORE_ON_RESTART.md``):

  * ``ClaudeCodeChatModel.ainvoke`` only runs the restore block when
    ``not client_manager.has_client(thread_id)`` — i.e. on the very
    first turn after a process (re)start.
  * After prewarm, ``has_client(thread_id) == True`` BEFORE the first
    ``ainvoke``. The restore block was therefore NEVER reached on
    restart: ``resume_session_id`` stayed ``None``, Claude was given a
    fresh empty CLI session, and the send loop assumed the 191 already-
    "sent" messages were in that session when they were not.

Result: the agent appeared to "forget" its conversation on every
restart. Live evidence (from a related internal incident analysis, not
included here): the persisted ``claude_session_state.json`` had
``session_uuid``=016a6bfd-... with ``sent_count``=191; the matching CLI
transcript JSONL exists on disk and ``resolve_restore_session(...)``
returns ``016a6bfd-...`` when invoked manually — but the live agent's
``MCP_DIAG connect_done`` lines show ``resume=None`` for every fresh
turn on the main thread, because the restore block never ran.

Contract enforced here
======================

1. ``ClientManager.prepare_proxy_only(thread_id, tools_metadata,
   working_dir)`` reserves a cache slot **with a ready ProxyMCPServer**
   but DOES NOT call ``ClaudeSDKClient.connect()``. No Claude CLI
   subprocess is spawned by this call.

2. After ``prepare_proxy_only``, ``has_client(thread_id) == False``.
   This keeps the "first turn after (re)start" discriminator in
   ``ClaudeCodeChatModel.ainvoke`` honest: history-restore runs as if
   the cache were empty.

3. The first ``get_or_create_client(thread_id, tools_metadata, ...,
   resume_session_id=X)`` after ``prepare_proxy_only`` ADOPTS the
   reserved ``ProxyMCPServer`` (when the tool set matches) and creates
   the CLI client with the requested ``resume_session_id``. No
   throw-away connect/disconnect, no rotate.

4. If the tool set does NOT match, the reserved proxy is discarded and
   a fresh one is built — same shape as the existing
   ``_cache_entry_matches_tools`` rotate path, just on the
   not-yet-connected slot.

⚠ DO NOT undo this by making prewarm call ``get_or_create_client``
again. That re-introduces the regression — see a related internal
incident analysis (not included here).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from claude_code_provider.client_manager import ClientManager
from claude_code_provider.custom_types import ToolMetadata


def _md(name: str) -> ToolMetadata:
    return {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


@pytest.mark.asyncio
async def test_prepare_proxy_only_does_not_spawn_cli_subprocess() -> None:
    """``prepare_proxy_only`` must NOT instantiate or connect a
    ``ClaudeSDKClient``. The reserved proxy must be the one returned to
    the next ``get_or_create_client`` so the cache slot survives intact."""
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    tools = [_md("alpha"), _md("beta")]

    # If the SDK client were spawned here, this patch would be touched.
    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient"
    ) as sdk_client_cls:
        sdk_client_cls.return_value = Mock()
        await manager.prepare_proxy_only(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
        )

    assert sdk_client_cls.call_count == 0, (
        "prepare_proxy_only must NOT instantiate ClaudeSDKClient — that "
        "spawns a Claude CLI subprocess and a fresh empty MCP session "
        "before history-restore runs. See a related internal incident "
        "analysis (not included here)."
    )


@pytest.mark.asyncio
async def test_has_client_false_after_prepare_proxy_only() -> None:
    """``has_client`` must stay False until a real CLI client is
    connected — this is the discriminator history-restore relies on."""
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    tools = [_md("alpha")]
    with patch("claude_code_provider.client_manager.ClaudeSDKClient"):
        await manager.prepare_proxy_only(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
        )

    assert manager.has_client("t1") is False, (
        "has_client(thread_id) must be False after prepare_proxy_only: "
        "ainvoke's restore block runs on `not has_client(thread_id)` and "
        "MUST run on the first real turn so the resume_session_id can be "
        "computed BEFORE the CLI subprocess is launched."
    )


@pytest.mark.asyncio
async def test_first_get_or_create_after_prewarm_uses_resume_session_id() -> None:
    """The first real ``get_or_create_client`` after prewarm MUST honour
    ``resume_session_id`` — i.e. it builds a CLI client with the
    requested resume, not with ``resume=None``."""
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    client_mock = Mock()
    client_mock.connect = AsyncMock()
    client_mock.disconnect = AsyncMock()

    tools = [_md("alpha"), _md("beta")]

    # Prewarm (no SDK client spawn expected).
    with patch("claude_code_provider.client_manager.ClaudeSDKClient"):
        await manager.prepare_proxy_only(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
        )

    # Now the real first turn: ainvoke resolves a resume id and calls
    # get_or_create_client with it. The CLI client MUST be created with
    # ``resume`` set to that id.
    captured_options: dict[str, object] = {}

    def _capture_client(options: ClaudeAgentOptions) -> Mock:
        captured_options["resume"] = options.resume
        captured_options["fork_session"] = options.fork_session
        return client_mock

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        side_effect=_capture_client,
    ):
        _client, _proxy = await manager.get_or_create_client(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
            resume_session_id="resume-uuid-xyz",
        )

    assert captured_options.get("resume") == "resume-uuid-xyz", (
        "First get_or_create_client after prepare_proxy_only must "
        "instantiate ClaudeSDKClient with options.resume=<the requested "
        "uuid>. Anything else means history-restore on restart is being "
        "lost — see a related internal incident analysis (not included "
        "here)."
    )
    assert captured_options.get("fork_session") is True
    client_mock.connect.assert_awaited_once()
    # has_client now becomes True (real CLI client is up).
    assert manager.has_client("t1") is True


@pytest.mark.asyncio
async def test_proxy_from_prewarm_is_reused_when_tools_match() -> None:
    """The ProxyMCPServer reserved by ``prepare_proxy_only`` must be the
    SAME instance returned by the subsequent ``get_or_create_client``
    when the tool set matches — no throw-away allocation, no duplicate
    MCP server seen by the SDK."""
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    tools = [_md("alpha"), _md("beta")]

    with patch("claude_code_provider.client_manager.ClaudeSDKClient"):
        await manager.prepare_proxy_only(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
        )
    reserved_proxy = manager._proxy_only["t1"]  # type: ignore[attr-defined]

    client_mock = Mock()
    client_mock.connect = AsyncMock()
    client_mock.disconnect = AsyncMock()

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        return_value=client_mock,
    ):
        _client, proxy = await manager.get_or_create_client(
            thread_id="t1",
            tools_metadata=tools,
            working_dir="/tmp",
        )

    assert proxy is reserved_proxy, (
        "Reserved ProxyMCPServer must be adopted by the first real "
        "get_or_create_client (same instance, identity-checked)."
    )
    # Reservation slot must be cleared after adoption.
    assert "t1" not in manager._proxy_only  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_proxy_from_prewarm_is_discarded_on_tool_mismatch() -> None:
    """If the tool set requested by ``get_or_create_client`` differs from
    the prewarm-reserved one, the reservation is discarded and a fresh
    proxy is built — same outcome as the existing rotate-on-mismatch
    path for connected entries (see ``_cache_entry_matches_tools``)."""
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    prewarm_tools = [_md("alpha")]
    real_tools = [_md("alpha"), _md("beta"), _md("gamma")]

    with patch("claude_code_provider.client_manager.ClaudeSDKClient"):
        await manager.prepare_proxy_only(
            thread_id="t1",
            tools_metadata=prewarm_tools,
            working_dir="/tmp",
        )
    reserved_proxy = manager._proxy_only["t1"]  # type: ignore[attr-defined]

    client_mock = Mock()
    client_mock.connect = AsyncMock()
    client_mock.disconnect = AsyncMock()

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        return_value=client_mock,
    ):
        _client, proxy = await manager.get_or_create_client(
            thread_id="t1",
            tools_metadata=real_tools,
            working_dir="/tmp",
        )

    assert proxy is not reserved_proxy, (
        "Tool-set mismatch must discard the reserved proxy — same "
        "invariant as the connected rotate-on-mismatch path."
    )
    assert set(proxy.tool_names) == {"alpha", "beta", "gamma"}
    assert "t1" not in manager._proxy_only  # type: ignore[attr-defined]
