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
})


def is_fatal_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` should transition the agent to ERROR.

    All other exceptions are recoverable per the No-Agent-Error-State Policy
    — the runtime must convert them into a tool-result or SystemMessage so the
    user can just type "continue".
    """
    return type(exc).__name__ in FATAL_EXCEPTION_TYPES
