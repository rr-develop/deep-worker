"""Unit tests for the send_message_to_agent inter-agent messaging tool.

See design document:
    deep_worker_cli/docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md

Architectural invariants verified here:

1. The sender name is injected by the tool factory from the sending
   agent's own identity, NOT taken from tool arguments. The LLM cannot
   forge it.

2. The delivered message has the exact shape:
       agent:<sender>:> <message>
   Example for sender="test_1", message="2+2":  "agent:test_1:> 2+2"

3. Failures (agent not found, connection errors, validation errors) are
   returned as structured dicts with a non-raising contract — the LLM can
   reason about them.

4. Self-send is rejected by default with error code "self_send_forbidden".

5. The call is fire-and-forget: the tool does not await the target
   agent's reply. We only verify the target's invoke_agent() was called
   with the prefixed message.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# NOTE: These imports will fail until the feature is implemented.
# That is expected during the TDD red phase.
from deep_worker_cli.tools import make_send_message_to_agent_tool


# ===========================================================================
# Helpers
# ===========================================================================


def _get_tool_coroutine(tool: Any) -> Any:
    """Return the async callable underlying a LangChain tool.

    LangChain's @tool-decorated callables can be invoked via .ainvoke(args)
    or via their underlying function (tool.coroutine / tool.func). We
    prefer ainvoke() because it matches how LangGraph actually calls
    tools at runtime.
    """
    return tool


async def _ainvoke(tool: Any, **kwargs: Any) -> Any:
    """Invoke a LangChain tool asynchronously with the given arguments."""
    return await tool.ainvoke(kwargs)


# ===========================================================================
# Factory contract
# ===========================================================================


class TestFactory:
    """Tests for make_send_message_to_agent_tool() factory semantics."""

    def test_factory_returns_tool_with_expected_name(self) -> None:
        tool = make_send_message_to_agent_tool("alpha")
        # LangChain BaseTool exposes .name
        assert tool.name == "send_message_to_agent"

    def test_factory_returns_distinct_tools_per_sender(self) -> None:
        """Two agents must get tools with distinct bound sender names."""
        tool_a = make_send_message_to_agent_tool("alpha")
        tool_b = make_send_message_to_agent_tool("beta")
        # They are different objects (factory produces fresh instances)
        assert tool_a is not tool_b

    def test_sender_name_cannot_be_passed_as_argument(self) -> None:
        """The tool's schema must NOT expose a 'sender' parameter.

        This is the core security invariant: the LLM must not be able to
        forge the sender. The tool's arguments are exactly agent_name
        and message — nothing else.
        """
        tool = make_send_message_to_agent_tool("alpha")
        schema = tool.args_schema
        # args_schema may be a pydantic model or a dict-like; get field names
        if hasattr(schema, "model_fields"):
            field_names = set(schema.model_fields.keys())
        else:  # pragma: no cover - fallback for older pydantic
            field_names = set(schema.__fields__.keys())  # type: ignore[attr-defined]

        assert "sender" not in field_names
        assert "sender_name" not in field_names
        assert "from_agent" not in field_names
        # The two legitimate arguments MUST be present
        assert "agent_name" in field_names
        assert "message" in field_names


# ===========================================================================
# Message prefixing (the core requirement)
# ===========================================================================


class TestMessagePrefixing:
    """Verify agent:<sender>:> <message> is constructed correctly."""

    async def test_message_prefixed_with_sender_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sender='test_1', message='2+2' -> delivered='agent:test_1:> 2+2'."""
        tool = make_send_message_to_agent_tool("test_1")

        delivered: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, name: str) -> None:
                delivered["target"] = name

            async def connect(self) -> None:
                return None

            async def invoke_agent(
                self, message: str, stream: bool = False
            ) -> dict[str, Any]:
                delivered["message"] = message
                delivered["stream"] = stream
                return {"success": True}

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        result = await _ainvoke(tool, agent_name="test_2", message="2+2")

        assert result["success"] is True
        assert result["target_agent"] == "test_2"
        assert result["delivered_content"] == "agent:test_1:> 2+2"
        assert delivered["target"] == "test_2"
        assert delivered["message"] == "agent:test_1:> 2+2"

    async def test_message_fidelity_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leading/trailing whitespace and newlines inside the message
        must not be stripped, mangled, or escaped — we keep fidelity
        with the TaskManager:> convention."""
        tool = make_send_message_to_agent_tool("alpha")

        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                return None

            async def invoke_agent(
                self, message: str, stream: bool = False  # noqa: ARG002
            ) -> dict[str, Any]:
                captured["message"] = message
                return {"success": True}

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        raw = "  line1\n\nline2 with spaces  "
        await _ainvoke(tool, agent_name="beta", message=raw)
        assert captured["message"] == f"agent:alpha:> {raw}"

    async def test_factory_bakes_sender_closure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two tools from two different factory calls must use their
        own bound sender name, not share global state."""
        tool_a = make_send_message_to_agent_tool("alpha")
        tool_b = make_send_message_to_agent_tool("beta")

        captured_msgs: list[str] = []

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                return None

            async def invoke_agent(
                self, message: str, stream: bool = False  # noqa: ARG002
            ) -> dict[str, Any]:
                captured_msgs.append(message)
                return {"success": True}

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        await _ainvoke(tool_a, agent_name="gamma", message="hi")
        await _ainvoke(tool_b, agent_name="gamma", message="hi")

        assert captured_msgs[0] == "agent:alpha:> hi"
        assert captured_msgs[1] == "agent:beta:> hi"


# ===========================================================================
# Validation and policy
# ===========================================================================


class TestValidation:
    """Argument validation and policy checks."""

    async def test_empty_agent_name_rejected(self) -> None:
        tool = make_send_message_to_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="", message="hi")
        assert result["success"] is False
        assert result["error"] == "invalid_argument"

    async def test_empty_message_rejected(self) -> None:
        tool = make_send_message_to_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta", message="")
        assert result["success"] is False
        assert result["error"] == "invalid_argument"

    async def test_whitespace_only_message_rejected(self) -> None:
        tool = make_send_message_to_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta", message="   \n\t")
        assert result["success"] is False
        assert result["error"] == "invalid_argument"

    async def test_self_send_rejected(self) -> None:
        tool = make_send_message_to_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="alpha", message="hi")
        assert result["success"] is False
        assert result["error"] == "self_send_forbidden"


# ===========================================================================
# Registry / connectivity failures
# ===========================================================================


class TestConnectivityErrors:
    """Structured error returns for transport failures."""

    async def test_agent_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If AgentClient.connect() raises AgentNotFoundError, the tool
        returns agent_not_found — NOT an exception."""
        from deep_worker_cli.exceptions import AgentNotFoundError

        tool = make_send_message_to_agent_tool("alpha")

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                raise AgentNotFoundError("Agent 'ghost' not found")

            async def invoke_agent(
                self, message: str, stream: bool = False
            ) -> dict[str, Any]:  # pragma: no cover - never reached
                return {"success": True}

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        result = await _ainvoke(tool, agent_name="ghost", message="hi")
        assert result["success"] is False
        assert result["error"] == "agent_not_found"
        # Human-readable message is present and mentions the target name
        assert "ghost" in result["message"]

    async def test_connection_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic IPCConnectionError -> connection_failed."""
        from deep_worker_cli.exceptions import IPCConnectionError

        tool = make_send_message_to_agent_tool("alpha")

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                raise IPCConnectionError("connection refused")

            async def invoke_agent(
                self, message: str, stream: bool = False
            ) -> dict[str, Any]:  # pragma: no cover - never reached
                return {"success": True}

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        result = await _ainvoke(tool, agent_name="beta", message="hi")
        assert result["success"] is False
        assert result["error"] == "connection_failed"

    async def test_unexpected_exception_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any unexpected exception from the client is converted into a
        structured error — never propagated to the LLM."""
        tool = make_send_message_to_agent_tool("alpha")

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                return None

            async def invoke_agent(
                self, message: str, stream: bool = False  # noqa: ARG002
            ) -> dict[str, Any]:
                raise RuntimeError("boom")

            async def disconnect(self) -> None:
                return None

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        result = await _ainvoke(tool, agent_name="beta", message="hi")
        assert result["success"] is False
        # The contract allows connection_failed as the catch-all for
        # unexpected transport exceptions.
        assert result["error"] == "connection_failed"
        assert "boom" in result["message"]


# ===========================================================================
# Fire-and-forget behavior
# ===========================================================================


class TestFireAndForget:
    """The tool must not wait for the target's assistant reply."""

    async def test_does_not_wait_for_target_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """invoke_agent() is called, and its return value (if any) is
        not interpreted as 'the reply'. The tool only confirms delivery.
        """
        tool = make_send_message_to_agent_tool("alpha")

        call_order: list[str] = []

        class _FakeClient:
            def __init__(self, name: str) -> None:  # noqa: ARG002
                pass

            async def connect(self) -> None:
                call_order.append("connect")

            async def invoke_agent(
                self, message: str, stream: bool = False  # noqa: ARG002
            ) -> dict[str, Any]:
                call_order.append("invoke")
                # Simulate agent accepting the message but producing a
                # long reply that we must NOT wait for in this test.
                return {
                    "success": True,
                    "response": {
                        "role": "assistant",
                        "content": "(streaming...)",
                    },
                }

            async def disconnect(self) -> None:
                call_order.append("disconnect")

        monkeypatch.setattr(
            "deep_worker_cli.tools.AgentClient", _FakeClient
        )

        result = await _ainvoke(tool, agent_name="beta", message="ping")
        assert result["success"] is True
        # Disconnect MUST still be called so we don't leak connections.
        assert call_order == ["connect", "invoke", "disconnect"]
        # The tool does NOT expose the target's reply content to the LLM.
        # (A separate future ask_agent tool would do that.)
        assert "response" not in result


# ===========================================================================
# Wiring (agent_process.py integration)
# ===========================================================================


class TestWiringIntoAgentProcess:
    """Verify the tool is exported and wireable.

    Full integration (the web_tools list assembly) is tested via the
    agent_process unit tests; here we just ensure the factory is
    importable from the expected path.
    """

    def test_factory_importable_from_tools_module(self) -> None:
        from deep_worker_cli.tools import make_send_message_to_agent_tool as f

        assert callable(f)


# ===========================================================================
# HITL approval wiring (agent.py)
# ===========================================================================


class TestHITLWiring:
    """send_message_to_agent must be gated by the standard approval
    system (same as execute/write_file/etc). Under auto_approve=True
    (the -sp flag), it runs without prompts — same as every other
    gated tool — this is handled by create_cli_agent() passing
    interrupt_on={}."""

    def test_interrupt_on_config_includes_send_message_to_agent(self) -> None:
        from deep_worker_cli.agent import _add_interrupt_on

        config = _add_interrupt_on()
        assert "send_message_to_agent" in config
        entry = config["send_message_to_agent"]
        # Same shape as the other entries
        assert "allowed_decisions" in entry
        assert set(entry["allowed_decisions"]) == {"approve", "reject"}
        assert callable(entry["description"])

    def test_interrupt_description_includes_target_and_preview(self) -> None:
        """The description formatter must render the target agent name
        and a preview of the message, mirroring the other tool formatters."""
        from deep_worker_cli.agent import _add_interrupt_on

        config = _add_interrupt_on()
        description_fn = config["send_message_to_agent"]["description"]

        tool_call = {
            "name": "send_message_to_agent",
            "args": {
                "agent_name": "beta",
                "message": "please review my report",
            },
            "id": "call_1",
            "type": "tool_call",
        }
        text = description_fn(tool_call, MagicMock(), MagicMock())

        assert isinstance(text, str)
        assert "beta" in text
        assert "please review my report" in text
