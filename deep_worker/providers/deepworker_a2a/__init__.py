"""A2A (Agent2Agent) protocol support for deep-worker agents.

This package implements full A2A protocol support as a self-contained,
optional layer on top of the existing agent turn engine. It is independent
of the ``send_message_to_agent`` RPyC channel.

Two ends of the protocol:

* **Inbound** — each agent can run an A2A HTTP server (Starlette ASGI on the
  agent's existing asyncio event loop, localhost-only by default) that exposes
  an ``AgentCard`` on ``/.well-known/agent-card.json`` and routes A2A JSON-RPC
  methods (``message/send``, ``message/stream``, ``tasks/get``,
  ``tasks/cancel``) to :class:`DeepWorkerAgentExecutor`, which reuses the
  agent's normal ``_handle_invoke`` turn engine.
* **Outbound** — agents get ``a2a_*`` tools to initiate A2A interactions with
  other A2A-compliant agents.

The import name is ``deepworker_a2a`` (NOT ``a2a``) on purpose: the official
``a2a-sdk`` already occupies the top-level ``a2a`` import namespace, and the
deep-worker launcher puts the ``deep_worker`` package root on ``PYTHONPATH``.
A package literally named ``a2a`` here would shadow the installed SDK and
break it.
See ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§5.1).

The heavy SDK imports (``a2a-sdk``) live inside the submodules, not here, so
that importing this package never hard-fails when the optional dependency is
absent (see :class:`deepworker_a2a.config.A2AConfig` — disabled by default).
"""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import A2AConfig  # noqa: F401

_LAZY = {
    "A2AConfig": "config",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(f".{sub}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
