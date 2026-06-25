"""Configuration for deep-worker A2A protocol support.

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§5, §6).

Invariants enforced here (user requirements / project policy):

* **Disabled by default** — ``enabled`` is ``False`` unless explicitly turned
  on. A failed ``a2a-sdk`` import must never be fatal for the agent (handled by
  the server bootstrap, which only runs when ``enabled`` is True).
* **localhost-only by default** — ``host`` defaults to ``127.0.0.1`` and any
  non-loopback bind address is REFUSED unless ``allow_external=True`` is
  explicitly provided. Nothing is exposed outside localhost by accident.
* **Future cross-host friendly** — the bind address, port and transport are
  plain config fields, so switching to cross-host later is a configuration
  change, not a code change.

No ``a2a-sdk`` import happens in this module, so importing it is always safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


@dataclass
class A2AConfig:
    """Runtime configuration for an agent's A2A endpoints.

    Attributes:
        enabled: Whether the A2A server is started for this agent. Default
            ``False`` (opt-in).
        host: Bind address for the A2A HTTP server. Default ``127.0.0.1``.
            A non-loopback value requires ``allow_external=True``.
        port: TCP port. ``0`` (default) means "let the OS pick a free port";
            the actual port is read back from the bound server and written to
            the process registry.
        transport: A2A transport. Only ``"jsonrpc"`` is supported for now.
        allow_external: Explicit opt-in required to bind a non-loopback host.
            Guards against accidentally exposing the agent outside localhost.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 0
    transport: str = "jsonrpc"
    allow_external: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("A2AConfig.host must be a non-empty string")
        if not _is_loopback(self.host) and not self.allow_external:
            raise ValueError(
                f"Refusing non-loopback A2A bind host {self.host!r} without "
                f"allow_external=True. A2A is localhost-only by default; set "
                f"allow_external=True explicitly to expose it outside localhost."
            )
        if self.transport != "jsonrpc":
            raise ValueError(
                f"Unsupported A2A transport {self.transport!r} (only 'jsonrpc')"
            )
        if not isinstance(self.port, int) or self.port < 0 or self.port > 65535:
            raise ValueError(f"A2AConfig.port out of range: {self.port!r}")

    def advertised_url(self, host: str | None = None, port: int | None = None) -> str:
        """Return the AgentCard ``url`` for this agent's A2A endpoint.

        Args:
            host: Override host to advertise (e.g. the actual bound host).
            port: Override port to advertise (e.g. the actual bound port when
                ``self.port == 0`` and the OS assigned one).
        """
        h = host or self.host
        p = port if port is not None else self.port
        return f"http://{h}:{p}/"

    @classmethod
    def from_env(cls, agent_name: str) -> A2AConfig:
        """Build config from the ``DEEP_WORKER_A2A`` environment variables.

        Recognised variables (all optional):
            DEEP_WORKER_A2A_ENABLED         -> enabled (truthy: 1/true/yes/on)
            DEEP_WORKER_A2A_HOST            -> host (default 127.0.0.1)
            DEEP_WORKER_A2A_PORT            -> port (default 0)
            DEEP_WORKER_A2A_ALLOW_EXTERNAL  -> allow_external (truthy)

        ``agent_name`` is accepted for forward compatibility (per-agent config
        sources) and currently unused.
        """

        def _truthy(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        host = os.environ.get("DEEP_WORKER_A2A_HOST", "127.0.0.1").strip() or "127.0.0.1"

        port_raw = os.environ.get("DEEP_WORKER_A2A_PORT", "0").strip() or "0"
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(
                f"DEEP_WORKER_A2A_PORT must be an integer, got {port_raw!r}"
            ) from exc

        return cls(
            enabled=_truthy("DEEP_WORKER_A2A_ENABLED"),
            host=host,
            port=port,
            transport="jsonrpc",
            allow_external=_truthy("DEEP_WORKER_A2A_ALLOW_EXTERNAL"),
        )
