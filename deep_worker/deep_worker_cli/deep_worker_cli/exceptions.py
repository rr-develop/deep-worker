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
