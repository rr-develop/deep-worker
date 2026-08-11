"""Unit tests for ``deepworker_a2a.card.build_agent_card`` and
``agent_card_to_summary`` (used by the ``a2a_discover`` tool).

We assert the card reflects the agent identity, advertises the JSONRPC
interface at the configured localhost URL, and declares streaming capability.
"""

import pytest

from deep_worker.providers.deepworker_a2a.card import build_agent_card, agent_card_to_summary


def test_build_card_basic_identity():
    card = build_agent_card(
        agent_name="agentA",
        url="http://127.0.0.1:9999/",
        description="deep-worker agent agentA",
        version="1.0.0",
    )
    assert card.name == "agentA"
    assert card.description == "deep-worker agent agentA"
    assert card.version == "1.0.0"
    # streaming must be advertised (we support message/stream via SSE)
    assert card.capabilities.streaming is True


def test_build_card_advertises_jsonrpc_interface_with_url():
    from a2a.utils.constants import TransportProtocol

    card = build_agent_card(agent_name="agentA", url="http://127.0.0.1:9999/")
    bindings = {i.protocol_binding: i.url for i in card.supported_interfaces}
    assert TransportProtocol.JSONRPC.value in bindings
    assert bindings[TransportProtocol.JSONRPC.value] == "http://127.0.0.1:9999/"


def test_build_card_has_at_least_one_skill():
    card = build_agent_card(agent_name="agentA", url="http://127.0.0.1:9999/")
    assert len(card.skills) >= 1
    assert card.skills[0].id


def test_card_to_summary_is_plain_dict():
    card = build_agent_card(
        agent_name="agentA",
        url="http://127.0.0.1:9999/",
        description="d",
        version="2.0.0",
    )
    summary = agent_card_to_summary(card)
    assert isinstance(summary, dict)
    assert summary["name"] == "agentA"
    assert summary["version"] == "2.0.0"
    assert summary["url"] == "http://127.0.0.1:9999/"
    assert summary["capabilities"]["streaming"] is True
    assert isinstance(summary["skills"], list)
    assert summary["skills"][0]["id"]
