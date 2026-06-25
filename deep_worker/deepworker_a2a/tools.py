"""LLM tools for outbound A2A interactions (``a2a_*``).

These are the agent-facing bridges that let an LLM initiate A2A protocol calls
to other A2A-compliant agents (the LLM cannot open HTTP sockets itself). They
mirror the A2A methods 1:1 and are thin wrappers over
:class:`deepworker_a2a.client.A2AOutboundClient`.

Naming note: the *tools* are ``a2a_*`` (tool names, not the Python package),
distinct from the internal ``send_message_to_agent`` RPyC channel.

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§2.2, §3.2).
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from deepworker_a2a.client import A2AOutboundClient


class _DiscoverArgs(BaseModel):
    target: str = Field(
        ...,
        description=(
            "Target A2A agent: either an http(s) URL of an A2A agent, or the "
            "registry name of another running deep-worker agent (its A2A "
            "endpoint is looked up automatically)."
        ),
    )


class _SendArgs(BaseModel):
    target: str = Field(
        ...,
        description=(
            "Target A2A agent: an http(s) URL, or the registry name of "
            "another running deep-worker agent."
        ),
    )
    text: str = Field(..., description="The message text to send to the agent.")
    context_id: str | None = Field(
        None,
        description=(
            "Optional A2A context id to continue a previous multi-turn "
            "conversation with the same agent. Omit to start a new one."
        ),
    )


class _TaskArgs(BaseModel):
    target: str = Field(
        ...,
        description="Target A2A agent: an http(s) URL or a deep-worker agent name.",
    )
    task_id: str = Field(..., description="The A2A task id to query / cancel.")


def make_a2a_tools(registry: Any | None = None) -> list[BaseTool]:
    """Build the outbound A2A LLM tools.

    Args:
        registry: Optional ``ProcessRegistry`` for resolving agent-name targets.
            Defaults to the standard registry inside the client.

    Returns:
        ``[a2a_discover, a2a_send_message, a2a_send_message_streaming,
           a2a_get_task, a2a_cancel_task]``.
    """
    client = A2AOutboundClient(registry=registry)

    @tool("a2a_discover", args_schema=_DiscoverArgs)
    async def a2a_discover(target: str) -> dict[str, Any]:
        """Fetch the AgentCard of an A2A agent and return a structured summary
        (url, name, description, version, capabilities, skills). Use this to
        learn what a remote A2A agent can do before sending it work."""
        return await client.discover(target)

    @tool("a2a_send_message", args_schema=_SendArgs)
    async def a2a_send_message(
        target: str, text: str, context_id: str | None = None
    ) -> dict[str, Any]:
        """Send a message to an A2A agent and wait for the final result.

        Returns a dict ``{success, answer, task_id, context_id, state}``.
        Pass ``context_id`` (from a prior call) to continue a multi-turn
        conversation with the same agent. Use ONLY when the user asked you to
        delegate work to that agent — do not initiate on your own."""
        return await client.send(target, text, context_id=context_id)

    @tool("a2a_send_message_streaming", args_schema=_SendArgs)
    async def a2a_send_message_streaming(
        target: str, text: str, context_id: str | None = None
    ) -> dict[str, Any]:
        """Like ``a2a_send_message`` but streams intermediate output. Returns
        ``{success, answer, chunks, task_id, context_id, state}`` where
        ``chunks`` is the ordered list of streamed artifact pieces."""
        return await client.send_streaming(target, text, context_id=context_id)

    @tool("a2a_get_task", args_schema=_TaskArgs)
    async def a2a_get_task(target: str, task_id: str) -> dict[str, Any]:
        """Get the current state/result of a previously created A2A task.
        Returns ``{success, task_id, context_id, state}``."""
        return await client.get_task(target, task_id)

    @tool("a2a_cancel_task", args_schema=_TaskArgs)
    async def a2a_cancel_task(target: str, task_id: str) -> dict[str, Any]:
        """Request cancellation of an in-progress A2A task on a remote agent.
        Returns ``{success, task_id, state}``."""
        return await client.cancel_task(target, task_id)

    return [
        a2a_discover,
        a2a_send_message,
        a2a_send_message_streaming,
        a2a_get_task,
        a2a_cancel_task,
    ]
