"""Custom exceptions for deep_worker."""


class DeepWorkerError(Exception):
    """Base exception for all deep_worker errors."""


class AgentAlreadyExistsError(DeepWorkerError):
    """Raised when trying to register an agent with a name that already exists."""


class AgentNotFoundError(DeepWorkerError):
    """Raised when trying to connect to an agent that doesn't exist."""


class IPCError(DeepWorkerError):
    """Base exception for IPC-related errors."""


class IPCConnectionError(IPCError):
    """Raised when IPC connection fails."""


class IPCTimeoutError(IPCError):
    """Raised when IPC operation times out."""


class RegistryError(DeepWorkerError):
    """Base exception for registry-related errors."""


class RegistryLockError(RegistryError):
    """Raised when unable to acquire registry lock."""


class ProcessNotAliveError(DeepWorkerError):
    """Raised when attempting to interact with a dead process."""


# ---------------------------------------------------------------------------
# No-Agent-Error-State Policy — exception taxonomy.
#
# See docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md and docs/POLICIES.md.
#
# The agent runtime classifies unhandled exceptions during ``_handle_invoke``
# as either *fatal* (transition the agent to ``AgentStatus.ERROR``) or
# *recoverable* (append a SystemMessage explaining the failure, leave the
# agent in ``IDLE`` so the user can just type "continue").
#
# Classification is by class name so it stays provider-agnostic — codex agents
# never import the claude-code module and vice versa.
# ---------------------------------------------------------------------------


class CheckpointCorruptedError(DeepWorkerError):
    """Raised by storage when the on-disk thread state cannot be loaded.

    Fatal — without checkpoint we cannot resume the conversation safely.
    """


class ProviderAuthFatalError(DeepWorkerError):
    """Raised by a provider when credentials are revoked / unrecoverable.

    Fatal — usage of the provider cannot continue without operator action.
    """


class AgentNotInitialisedError(DeepWorkerError):
    """Raised when the agent's runtime graph is missing — cannot proceed.

    Fatal — the agent has nothing to run.
    """


# Match by class name so the table stays provider-agnostic.  Adding to this
# set is an explicit decision that the new failure mode is genuinely
# non-recoverable.  Anything not on this list is recoverable by default.
FATAL_EXCEPTION_TYPES: frozenset[str] = frozenset({
    "ClaudeUsageLimitError",
    "CodexUsageLimitError",
    "GrokUsageLimitError",
    "CheckpointCorruptedError",
    "ProviderAuthFatalError",
    "AgentNotInitialisedError",
    # Raised by deepagents.middleware.summarization.SummarizationMiddleware
    # when the pre-summarization history offload to backend fails.
    # Continuing summarization would silently discard the evicted history.
    # See DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md.
    "HistoryOffloadError",
})


def is_fatal_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` should transition the agent to ERROR.

    All other exceptions are recoverable per the No-Agent-Error-State Policy
    — the runtime must convert them into a tool-result or SystemMessage so the
    user can just type "continue".
    """
    return type(exc).__name__ in FATAL_EXCEPTION_TYPES


# ---------------------------------------------------------------------------
# No-Agent-Error-State Policy — exception taxonomy.
#
# See docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md and docs/POLICIES.md.
#
# The agent runtime classifies unhandled exceptions during ``_handle_invoke``
# as either *fatal* (transition the agent to ``AgentStatus.ERROR``) or
# *recoverable* (append a SystemMessage explaining the failure, leave the
# agent in ``IDLE`` so the user can just type "continue").
#
# Classification is by class name so it stays provider-agnostic — codex agents
# never import the claude-code module and vice versa.
# ---------------------------------------------------------------------------


class CheckpointCorruptedError(DeepWorkerError):
    """Raised by storage when the on-disk thread state cannot be loaded.

    Fatal — without checkpoint we cannot resume the conversation safely.
    """


class ProviderAuthFatalError(DeepWorkerError):
    """Raised by a provider when credentials are revoked / unrecoverable.

    Fatal — usage of the provider cannot continue without operator action.
    """


class AgentNotInitialisedError(DeepWorkerError):
    """Raised when the agent's runtime graph is missing — cannot proceed.

    Fatal — the agent has nothing to run.
    """


# Match by class name so the table stays provider-agnostic.  Adding to this
# set is an explicit decision that the new failure mode is genuinely
# non-recoverable.  Anything not on this list is recoverable by default.
FATAL_EXCEPTION_TYPES: frozenset[str] = frozenset({
    "ClaudeUsageLimitError",
    "CodexUsageLimitError",
    "GrokUsageLimitError",
    "CheckpointCorruptedError",
    "ProviderAuthFatalError",
    "AgentNotInitialisedError",
    "HistoryOffloadError",
})


def is_fatal_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` should transition the agent to ERROR.

    All other exceptions are recoverable per the No-Agent-Error-State Policy
    — the runtime must convert them into a tool-result or SystemMessage so the
    user can just type "continue".
    """
    return type(exc).__name__ in FATAL_EXCEPTION_TYPES


# ---------------------------------------------------------------------------
# No-Agent-Error-State Policy — exception taxonomy.
#
# See docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md and docs/POLICIES.md.
#
# The agent runtime classifies unhandled exceptions during ``_handle_invoke``
# as either *fatal* (transition the agent to ``AgentStatus.ERROR``) or
# *recoverable* (append a SystemMessage explaining the failure, leave the
# agent in ``IDLE`` so the user can just type "continue").
#
# Classification is by class name so it stays provider-agnostic — codex agents
# never import the claude-code module and vice versa.
# ---------------------------------------------------------------------------


class CheckpointCorruptedError(DeepWorkerError):
    """Raised by storage when the on-disk thread state cannot be loaded.

    Fatal — without checkpoint we cannot resume the conversation safely.
    """


class ProviderAuthFatalError(DeepWorkerError):
    """Raised by a provider when credentials are revoked / unrecoverable.

    Fatal — usage of the provider cannot continue without operator action.
    """


class AgentNotInitialisedError(DeepWorkerError):
    """Raised when the agent's runtime graph is missing — cannot proceed.

    Fatal — the agent has nothing to run.
    """


# Match by class name so the table stays provider-agnostic.  Adding to this
# set is an explicit decision that the new failure mode is genuinely
# non-recoverable.  Anything not on this list is recoverable by default.
FATAL_EXCEPTION_TYPES: frozenset[str] = frozenset({
    "ClaudeUsageLimitError",
    "CodexUsageLimitError",
    "GrokUsageLimitError",
    "CheckpointCorruptedError",
    "ProviderAuthFatalError",
    "AgentNotInitialisedError",
    "HistoryOffloadError",
})


def is_fatal_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` should transition the agent to ERROR.

    All other exceptions are recoverable per the No-Agent-Error-State Policy
    — the runtime must convert them into a tool-result or SystemMessage so the
    user can just type "continue".
    """
    return type(exc).__name__ in FATAL_EXCEPTION_TYPES


# ---------------------------------------------------------------------------
# No-Silent-API-Errors Policy (docs/POLICIES.md §9) — recoverable provider/API
# error taxonomy.
#
# A provider's API can fail in a way the runtime CAN recover from (keep the
# conversation alive, stay IDLE) — e.g. a non-terminal grok ACP error or a
# codex protocol hiccup that is not a usage cap. Such an error is NOT fatal,
# but it MUST still be surfaced to the user as a visible, PERSISTENT error in
# the UI — never disguised as a benign "system: type continue" note and never
# a self-dismissing toast. The recoverable path keys on this set to choose the
# ``type="error"`` (persistent) notification instead of ``type="system"``.
#
# Matched by class name so the table stays provider-agnostic (codex agents
# never import the grok module and vice versa). The account-terminal provider
# errors (…UsageLimitError, ProviderAuthFatalError) are already fatal and
# surface as ERROR via FATAL_EXCEPTION_TYPES; this set is only the recoverable
# ones that would otherwise be swallowed.
# ---------------------------------------------------------------------------

PROVIDER_API_ERROR_TYPES: frozenset[str] = frozenset({
    # grok ACP JSON-RPC error reply that is NOT an account-terminal cap
    # (those are GrokUsageLimitError, already fatal). E.g. an upstream 5xx or
    # a malformed session error the runtime recovers from.
    "GrokAcpClientError",
    "GrokConcurrentPromptError",
    # codex exec protocol error that is not a usage cap.
    "CodexProtocolError",
    # claude-code CLI returned an upstream API error AS the turn (repacked
    # into a TextBlock or signalled via ResultMessage.is_error /
    # AssistantMessage.error). Recoverable at the account level; the request
    # itself failed. See DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md.
    "ClaudeApiResponseError",
})


def is_provider_api_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a recoverable provider/API error that must be
    surfaced to the user as a visible, persistent UI error (POLICIES.md §9).

    Fatal provider errors (usage/billing caps, auth) are handled separately by
    :func:`is_fatal_exception` and already surface as ``AgentStatus.ERROR``;
    this predicate covers the *recoverable* API failures that would otherwise
    be swallowed into a benign system note or a fleeting toast.
    """
    return type(exc).__name__ in PROVIDER_API_ERROR_TYPES


class CapabilityError(DeepWorkerError):
    """Raised when an explicitly-enabled optional capability is unavailable.

    Used by tri-state capability policies (web-search, semantic-search,
    provider extras): when the operator EXPLICITLY enables a capability
    (``--enable-web-search`` / ``pip install deep-worker[semantic-search]``)
    but the backing dependency/SDK is missing, the agent must fail loudly
    with an actionable remedy instead of silently skipping the capability.

    Attributes:
        capability: Machine-readable capability name (e.g. ``"web_search"``).
        remedy: Human-readable remediation hint (install command / env var).
    """

    def __init__(
        self,
        capability: str,
        remedy: str | None = None,
        message: str | None = None,
    ) -> None:
        self.capability = capability
        self.remedy = remedy
        if message is None:
            message = f"capability {capability!r} is enabled but unavailable"
            if remedy:
                message += f": {remedy}"
        super().__init__(message)
