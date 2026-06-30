"""Shared fixtures/helpers for the rpyc_service subscriber tests.

Why this exists
===============
Production no longer calls ``await subscriber.on_message(payload)``
directly. Both ``AgentService._notify_subscribers`` and the initial
snapshot push in ``exposed_subscribe_messages`` route the call through
``rpyc.core.netref.asyncreq``::

    asyncreq(subscriber, 8, "on_message", (payload_str,), ())

and ``asyncreq`` (see ``rpyc_async``'s ``rpyc/core/netref.py``) does::

    conn = object.__getattribute__(proxy, "____conn__")
    return conn.async_request(handler, proxy, *args)

It reaches ``____conn__`` via ``object.__getattribute__`` — bypassing
``MagicMock.__getattr__`` entirely. A bare ``MagicMock()`` has no real
``____conn__`` entry in its ``__dict__`` (the dunder is on the
blacklist), so ``object.__getattribute__(mock, "____conn__")`` raises
``AttributeError`` and the subscriber is evicted before the payload is
ever delivered — which is exactly why these tests used to fail.

``make_netref_subscriber`` builds a MagicMock whose ``____conn__`` is a
real object placed in ``__dict__`` via ``object.__setattr__``. Its
``async_request`` mirrors the real signature
``async_request(handler, proxy, name, args, kwargs)`` and, for the
``on_message`` HANDLE_CALLATTR (8) call, delegates to the supplied
``on_message`` coroutine. The returned proxy also carries a real
``.on_message`` attribute so tests that assert ``await_count`` /
``await_args`` on an ``AsyncMock`` keep working unchanged.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import pytest


class _FakeConn:
    """Stand-in for a netref's ``____conn__``.

    Implements just enough of the rpyc_async connection contract for
    ``asyncreq(proxy, 8, "on_message", args, kwargs)`` to resolve to an
    awaitable that ultimately invokes the test's ``on_message``.
    """

    def __init__(self, on_message: Callable[..., Any]) -> None:
        self._on_message = on_message

    def async_request(
        self,
        handler: int,
        _proxy: Any,
        name: str,
        args: tuple[Any, ...],
        kwargs: tuple[Any, ...] | dict[str, Any],
    ) -> Any:
        # 8 == consts.HANDLE_CALLATTR; production only ever dispatches
        # ``on_message`` through this path.
        if handler == 8 and name == "on_message":
            call_kwargs = dict(kwargs) if kwargs else {}
            # Return the awaitable so ``await asyncreq(...)`` works and so
            # fire_and_forget_async can schedule it.
            return self._on_message(*args, **call_kwargs)
        raise NotImplementedError(
            f"unexpected async_request handler={handler!r} name={name!r}"
        )


def make_netref_subscriber(on_message: Callable[..., Any]) -> MagicMock:
    """Build a MagicMock subscriber that behaves like an rpyc_async netref.

    ``on_message`` is the coroutine function invoked when production
    dispatches ``asyncreq(subscriber, 8, "on_message", (payload,), ())``.
    It is also exposed as ``subscriber.on_message`` so existing
    ``AsyncMock``-style assertions (``await_count`` / ``await_args``)
    keep working when callers pass an ``AsyncMock``.
    """
    subscriber = MagicMock()
    subscriber.on_message = on_message
    # ``____conn__`` must live in ``__dict__`` so the netref's
    # ``object.__getattribute__(proxy, "____conn__")`` finds it without
    # tripping MagicMock's dunder blacklist.
    object.__setattr__(subscriber, "____conn__", _FakeConn(on_message))
    return subscriber


@pytest.fixture
def netref_subscriber() -> Callable[[Callable[..., Any]], MagicMock]:
    """Fixture exposing :func:`make_netref_subscriber` as a factory.

    Conftest helpers cannot be imported via ``from .conftest import ...``
    when the test directory has no ``__init__.py`` (relative import has no
    parent package). A fixture is the conftest-native way to share the
    factory across the rpyc_service test modules.
    """
    return make_netref_subscriber
