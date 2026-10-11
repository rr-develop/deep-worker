"""The grok :class:`AcpProfile` — grok's differences from the shared ACP layer, as data.

Grok speaks ACP natively (``grok agent stdio``). Until 2026-08-24 it did so
through a private FORK of the ACP layer: nine same-named modules, ~2000 lines,
of which ``proxy_mcp_server.py`` (608 lines) differed from its ACP twin ONLY in
comments. The fork predates the shared layer, so it never received any of the
fixes the shared layer has landed since:

  * ``initialize``/``authenticate`` were not idempotent (re-handshake on reuse);
  * no ``session_new_validator`` HOOK at all — the shared layer's post-session
    gate (the 2026-08-15 model-pin incident) could not run on ``session/new``
    or ``session/load``. NOTE: the hook is now inherited but this profile
    leaves it unset ON PURPOSE — see ``build_profile`` below;
  * a per-model ``_acp_started`` flag that wedges the model permanently after a
    CLI EOF (the shared client recovers by re-handshaking on next use);
  * no shared client pooling: the agent model and the TaskManager model each
    spawned their own ``grok`` subprocess;
  * ``asyncio.get_event_loop()`` (deprecated) instead of ``get_running_loop()``;
  * and no ``SummarizerPort`` at all, which combined with grok's misclassification
    as *stateless* meant summarization silently dumped the full history into a
    fresh grok session.

Structural comparison before the merge: grok's chat-model exposed NO method and
NO private attribute that ACP lacked — a strict subset, ACP being grok plus 29
methods. So this file is the entire remaining delta.
"""

from __future__ import annotations

from ..acp.profile import AcpProfile, PermissionPolicy
from .acp_types import GROK_COLLIDING_BUILTINS, KNOWN_GROK_BUILTINS

# grok's proprietary notification namespace — pure noise for our purposes.
VENDOR_PREFIX = "_x.ai/"

# Name surfaced for a grok built-in we do not know at design time, and the args
# key carrying its real title. PINNED (not derived from the provider id) — see
# the profile fields below for why these two strings are a public contract.
GENERIC_BUILTIN_NAME = "GrokBuiltin"
ORIGINAL_TITLE_ARG_KEY = "_grok_original_title"

# No ``extra_limit_markers``: the fork's marker tuple was compared pattern by
# pattern against the shared layer's generic set and is a strict subset — every
# grok pattern ("subscription limit", "hit your ... limit", "payment required",
# r"\b402\b", the credit-exhaustion family) is already matched there. Restating
# them would be dead configuration that silently rots out of sync.


def build_profile(
    model: str | None = None,
    *,
    auto_approve: bool = False,
) -> AcpProfile:
    """Build the grok profile.

    ``model`` is accepted for symmetry with the claude-code profile builder.
    grok takes its model from its own CLI configuration rather than from an
    ACP field or an environment variable, so a pin is not forwarded here; see
    the model-resolution path in the chat model.

    For the same reason ``session_new_validator`` is left UNSET (defaulting to
    None) — deliberately, not by omission. That hook exists to catch an
    adapter silently dropping an operator's model pin (2026-08-15); with no
    pin travelling to the CLI there is nothing to validate, so a validator
    here would be dead code. claude-code wires one because its pin does travel
    (via ANTHROPIC_MODEL) and can be dropped without any error.

    ``auto_approve`` mirrors deep-worker's ``-sp`` operator flag: with it,
    EVERY permission request is approved (no tool list to maintain); with
    the default ``False`` NOTHING is approved.
    """
    # Fail-closed by default; blanket approval under -sp (DESIGN §4.6).
    #
    # History, so nobody re-derives the wrong lesson. The forked client
    # approved every request UNCONDITIONALLY — no -sp gate at all — which is
    # why grok agents ran fully autonomous however they were launched. The
    # shared policy fixed the gate but arrived with an EMPTY allowlist and,
    # via config.py, without the flag: the net effect was that grok approved
    # NOTHING and every shell turn was killed by the CLI for a rejected
    # permission (2026-08-24). The correct semantic is neither of those two:
    # honour -sp, and when it is set approve everything the CLI asks about.
    policy = PermissionPolicy(
        auto_approve=auto_approve,
        # grok offers ``allow-*`` / ``deny-*`` option kinds.
        allow_kind_prefix="allow",
        deny_kind_prefix="deny",
    )

    return AcpProfile(
        provider="grok",
        cmd=("grok", "agent", "stdio"),
        vendor_prefix=VENDOR_PREFIX,
        # grok wraps MCP tool calls in a dispatch built-in:
        # CallMcpTool{server, toolName, arguments} on ≤1.0.4 and
        # use_tool{tool_name="<server>__<tool>", tool_input} on 1.0.5+
        # (a related internal incident analysis, not included here). Recognition
        # is title-agnostic (acp_types.proxy_dispatch_inner); this field
        # documents the ≤1.0.4 wrapper name for checkpointed histories.
        mcp_dispatch_tool="CallMcpTool",
        mcp_dispatch_style="dispatch_builtin",
        known_builtins=KNOWN_GROK_BUILTINS,
        # 1.0.5 snake_case natives that reuse deep-worker tool names — see
        # GROK_COLLIDING_BUILTINS for why they are segregated.
        colliding_builtins=GROK_COLLIDING_BUILTINS,
        session_load_supported=True,
        # grok authenticates with methodId=cached_token after initialize.
        requires_authenticate=True,
        # grok puts the real tool name in ``title``.
        tool_name_from_update="from_title",
        # PINNED, not derived. The shared layer would compute "grokBuiltin"
        # from the provider id, but ``bounded_tool_call`` matches the literal
        # "GrokBuiltin" (and reads "_grok_original_title") to give image
        # generation the 600s bound rather than the default 120s. A silent
        # rename would not fail anywhere — imagine calls would just begin
        # timing out. These strings are also already written into every
        # checkpointed grok history.
        generic_builtin_name=GENERIC_BUILTIN_NAME,
        original_title_arg_key=ORIGINAL_TITLE_ARG_KEY,
        permission_policy=policy,
        # grok sends no usage_update and reports ``_meta`` only with the
        # prompt reply, so the per-call number comes from its own telemetry
        # log (see acp/context_watch.py).
        live_context_source="grok_unified_log",
        adapter="grok-cli",
        session_capability="load-v1",
    )
