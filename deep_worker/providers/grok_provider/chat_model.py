"""GrokChatModel — the xAI Grok CLI over the Agent Client Protocol.

Thin subclass of the shared :class:`AcpChatModel`, exactly like
``ClaudeCodeAcpChatModel``: only the profile differs (``grok agent stdio``
command, ``_x.ai/`` vendor prefix, ``CallMcpTool`` dispatch, grok built-ins,
``allow-*``/``deny-*`` permission kinds). Everything else — the ACP client, the
proxy MCP server, session state, tool passthrough, bounded turn state, and the
``SummarizerPort`` — is inherited.

Before 2026-08-24 this file was ~1550 lines: a private fork of the ACP layer,
made before that layer existed. The fork was a strict SUBSET of the shared
implementation — it exposed no method and no private attribute that
``AcpChatModel`` lacked, while missing 29 of its methods — so replacing it is a
pure gain. What it had been missing:

  * ``asummarize`` / ``areset_after_summary``. Combined with grok being
    misclassified as *stateless*, summarization skipped the session-stateful
    guard and silently dumped the entire history into a fresh grok session —
    full re-tokenization, every time.
  * the ``session_new_validator`` HOOK itself (the 2026-08-15 incident's
    model-pin gate) on both ``session/new`` and ``session/load``. The
    MECHANISM is inherited; grok deliberately does NOT wire a validator into
    it, because grok takes its model from its own CLI configuration rather
    than from an operator pin — there is nothing to validate, and a validator
    here would be dead code. claude-code wires one because its pin travels
    through the environment and the adapter can silently drop it.
  * idempotent ``initialize`` / ``authenticate``.
  * shared-client pooling: the agent model and the TaskManager model each
    spawned their own ``grok`` subprocess.
  * EOF recovery — a per-model ``_acp_started`` flag latched True and wedged
    the model permanently once the CLI died, where the shared client
    transparently re-spawns and re-handshakes on next use.
  * fail-closed permissions (see ``profile.build_profile``).

See docs/DESIGN_GROK_PROVIDER.md.
"""
from __future__ import annotations

from typing import Any

from ..acp.chat_model import AcpChatModel
from .profile import build_profile


class GrokChatModel(AcpChatModel):
    """LangChain chat-model wrapper around the grok CLI via ACP.

    Provider name in deep-worker: ``"grok"``. Construction is cheap and
    side-effect-free; the proxy MCP server and the shared ACP client start
    lazily on the first ``ainvoke``.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        working_dir: str | None = None,
        *,
        auto_approve: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            profile=build_profile(
                model=model,
                auto_approve=auto_approve,
            ),
            model=model,
            max_tokens=max_tokens,
            working_dir=working_dir,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        # Deliberately NOT inherited. The shared layer reports "acp", but this
        # string is grok's provider identity for the hot/cold classifier, the
        # token-stats records and the availability tracker, all of which key on
        # the provider name. It must stay "grok".
        return "grok"
