"""ClaudeCodeAcpChatModel — claude-code-sdk over the Agent Client Protocol.

Thin subclass of the shared :class:`AcpChatModel`: only the profile differs
(claude-agent-acp adapter, ``_claude/`` vendor prefix, MCP-name-prefix
dispatch, claude built-ins, reject-based permission deny). Everything else —
the ACP client, proxy MCP server, session state, tool passthrough, bounded
state — is inherited from the shared layer.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..acp.chat_model import AcpChatModel
from .profile import build_profile, preflight_async, read_session_model

logger = logging.getLogger(__name__)


class ClaudeCodeAcpChatModel(AcpChatModel):
    """LangChain chat-model wrapper around Claude Code via claude-agent-acp.

    Provider name in deep-worker: ``"claude-code"``. Construction runs
    the §5.2 adapter preflight from TRUSTED metadata only (adapter entry +
    pinned package.json version — no node spawn, review item 7) and is
    otherwise cheap and side-effect-free; the live node/adapter probe and
    the proxy MCP server / ACP client start lazily on the first ``ainvoke``.
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
        # ``model`` MUST reach the profile: it is pinned into the CLI's
        # environment there (ANTHROPIC_MODEL). Omitting it here was the
        # owner-reported defect — the name was stored on the model object and
        # reported to token stats while the CLI silently ran its own default.
        profile = build_profile(
            model=model,
            auto_approve=auto_approve,
        )
        super().__init__(
            profile=profile,
            model=model,
            max_tokens=max_tokens,
            working_dir=working_dir,
            **kwargs,
        )

    async def _ensure_acp_started(self) -> None:
        # Review item 7: run the §5.2 preflight ASYNC (node >= 22 + live
        # adapter --version smoke) inside the event loop before the first
        # handshake — never a sync subprocess during async agent init.
        await preflight_async()
        await super()._ensure_acp_started()

    async def _finalize_response(
        self,
        response: dict,
        text_acc: list[str],
        state: Any | None = None,
    ) -> AIMessage:
        """Turn boundary: the base reports the model, then we fill the gap.

        Overridden here rather than in ``_maybe_report_model`` because the
        session id lives on ``state``, which only this signature carries.
        """
        message = await super()._finalize_response(response, text_acc, state)
        await self._report_transcript_model(state)
        return message

    async def _report_transcript_model(self, state: Any | None) -> None:
        """Report the model this session is really running.

        The base implementation reads ``_meta.modelId`` off the
        ``session/prompt`` reply. claude-agent-acp never sends it, so for
        this provider the base is a no-op: ``_resolved_model_name`` stayed
        empty, token stats recorded ``""`` and the web UI showed
        ``unknown`` for every claude-code agent. Agents launched with
        ``-m opus`` were worse than useless — they reported the ALIAS
        ``opus`` while the session ran ``claude-opus-5``.

        The CLI writes its own transcript for this very session, and every
        assistant record in it carries the model the API answered with,
        beside that request's real usage. It is a report of what happened
        rather than a declaration of intent, and it is free: the file is
        already on disk, written by the process we are already running.

        Called at each turn boundary, so a fresh session picks the name up
        with its first reply and a resumed one already has it. Runs after
        the base: if a future adapter does start sending ``modelId``, the
        protocol answer is already in place and this only confirms it.

        Reporting a model is display metadata; completing the turn is the
        job. No failure here may propagate — losing a whole turn over a
        cosmetic field would be a far worse bug than the one being fixed.
        Cancellation is unaffected: ``CancelledError`` derives from
        ``BaseException``, so ``except Exception`` lets it through and a
        cancelled turn stays cancelled.

        The ACP session id names the transcript file: live-verified, every
        ``sessionId`` from ``session/new`` has a matching ``<id>.jsonl``
        under the CLI's projects directory, and the adapter resumes by
        that same id.
        """
        session_id = getattr(state, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            return
        try:
            # Small local file, but read off the event loop so a slow disk
            # cannot stall the turn.
            model = await asyncio.to_thread(read_session_model, session_id)
            if model and model != self._resolved_model_name:
                await self._record_resolved_model(model)
        except Exception as exc:  # noqa: BLE001 — never fatal to a turn
            logger.warning(
                "[claude-code] could not report the session model "
                "(%s: %s); the turn completes regardless",
                type(exc).__name__, exc,
            )
