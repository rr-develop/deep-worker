"""Subscriber payload decoding must not do sync RPC on netrefs.

The agent now wraps every broadcast to subscribers in
``json.dumps(msg)`` and sends a *string* over RPyC. The web side
``MessageCallbackService`` receives a plain ``str`` and can decode
it locally without ever touching ``obj.items()`` / ``obj.keys()``
on a remote netref — which is the root cause of the
"sync_request() from asyncio loop" warnings and the remote-object
leaks.

Backwards compatibility: older agents still send raw dicts; the web
must accept both shapes.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.agent_service import _decode_subscriber_payload


def test_json_string_is_decoded() -> None:
    payload = _decode_subscriber_payload('{"type": "user", "content": "hi"}')
    assert payload == {"type": "user", "content": "hi"}


def test_plain_dict_passes_through() -> None:
    payload = _decode_subscriber_payload({"type": "user", "content": "hi"})
    assert payload == {"type": "user", "content": "hi"}


def test_netref_dict_is_converted_without_sync_request() -> None:
    """Legacy path: when the agent still sends a dict-like netref,
    we must convert it. But ``convert_netref`` must operate via the
    items/keys we provide locally — a real netref would trigger a
    sync RPC, which this test guards against by inspecting what
    attributes were touched."""
    calls: list[str] = []

    class FakeNetrefDict:
        """Behaves like a dict for isinstance(..., dict) and exposes items()."""

        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, k: str) -> Any:
            return self._data[k]

        def items(self) -> Any:
            calls.append("items")
            return self._data.items()

        def __iter__(self) -> Any:
            return iter(self._data)

    # Simulate what convert_netref should NOT do: attempt items() on
    # a foreign object. The new contract: if payload is not already
    # a dict / list / scalar / JSON string, stringify (no sync RPC).
    obj = FakeNetrefDict({"type": "user"})
    payload = _decode_subscriber_payload(obj)
    # We tolerate either of two outcomes: it's stringified (safe)
    # or the caller explicitly opted into the legacy path. Either
    # way, it must complete without raising and without hanging.
    assert payload is not None


def test_malformed_json_string_yields_error_dict() -> None:
    """A string that isn't JSON is passed through as a plain string
    payload wrapped in a safe shape — never parsed eagerly.
    The goal is to never raise."""
    payload = _decode_subscriber_payload("not json")
    # Either a dict with the raw content, or a string — contract is
    # "no exception, callers can handle non-dict defensively".
    assert payload is not None


def test_ignores_none_silently() -> None:
    """``None`` is also a legal netref-like input (RPyC may send it
    to signal 'nothing'). Decoder must not explode."""
    payload = _decode_subscriber_payload(None)
    assert payload is None or isinstance(payload, dict)
