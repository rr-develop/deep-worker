"""AgentCard construction for deep-worker A2A support.

Builds an A2A ``AgentCard`` (a2a-sdk 1.1.0, protobuf types) from the agent's
identity, advertising the JSON-RPC interface at the agent's localhost URL and
declaring streaming capability (we support ``message/stream`` via SSE).

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§3.1, §5.1).

The ``a2a-sdk`` import lives here (not at package import time) so the top-level
``deepworker_a2a`` package import never hard-fails when the optional dependency
is absent.
"""

from __future__ import annotations

from typing import Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol

DEFAULT_INPUT_MODES = ["text/plain"]
DEFAULT_OUTPUT_MODES = ["text/plain"]


def _default_skill(agent_name: str) -> AgentSkill:
    """The baseline general-purpose skill every deep-worker agent exposes.

    Later phases may map the agent's ``~/.deep_worker/agents/<name>/skills`` into
    additional ``AgentSkill`` entries.
    """
    return AgentSkill(
        id="deepworker.general",
        name="General deep-worker agent",
        description=(
            f"General-purpose deep-worker agent '{agent_name}': coding, "
            f"research, analysis and multi-step tasks via natural-language "
            f"requests."
        ),
        tags=["general", "deep-worker", "agent"],
        examples=["Summarise this repository", "Fix the failing test"],
    )


def build_agent_card(
    agent_name: str,
    url: str,
    *,
    description: str | None = None,
    version: str = "1.0.0",
    skills: list[AgentSkill] | None = None,
    streaming: bool = True,
) -> AgentCard:
    """Build an A2A ``AgentCard`` for ``agent_name`` served at ``url``.

    Args:
        agent_name: The deep-worker agent name (becomes the card ``name``).
        url: The localhost URL the A2A JSON-RPC endpoint is reachable at,
            e.g. ``http://127.0.0.1:9999/``.
        description: Human-readable description. Defaulted from the name.
        version: Card version string.
        skills: Optional explicit skills; defaults to a single general skill.
        streaming: Whether to advertise streaming (message/stream / SSE).

    Returns:
        A protobuf ``AgentCard`` ready to serve at ``/.well-known/agent-card.json``.
    """
    return AgentCard(
        name=agent_name,
        description=description or f"deep-worker agent {agent_name}",
        version=version,
        supported_interfaces=[
            AgentInterface(
                protocol_binding=TransportProtocol.JSONRPC.value,
                url=url,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=False,
        ),
        default_input_modes=list(DEFAULT_INPUT_MODES),
        default_output_modes=list(DEFAULT_OUTPUT_MODES),
        skills=skills if skills is not None else [_default_skill(agent_name)],
    )


def agent_card_to_summary(card: AgentCard) -> dict[str, Any]:
    """Reduce an ``AgentCard`` to a plain, JSON-friendly dict.

    Used by the ``a2a_discover`` tool so the LLM gets a structured summary
    (url, capabilities, skills) rather than a raw protobuf message (per the
    reviewer's note). Never raises on a well-formed card.
    """
    # Pick a URL to advertise: prefer the JSON-RPC interface, else the first.
    url = ""
    interfaces = list(card.supported_interfaces)
    for iface in interfaces:
        if iface.protocol_binding == TransportProtocol.JSONRPC.value:
            url = iface.url
            break
    if not url and interfaces:
        url = interfaces[0].url

    return {
        "name": card.name,
        "description": card.description,
        "version": card.version,
        "url": url,
        "capabilities": {
            "streaming": bool(card.capabilities.streaming),
            "push_notifications": bool(card.capabilities.push_notifications),
        },
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "tags": list(s.tags),
            }
            for s in card.skills
        ],
    }
