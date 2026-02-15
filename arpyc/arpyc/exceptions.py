"""Custom exceptions for arpyc."""

from __future__ import annotations


class ARPyCError(Exception):
    """Base exception for arpyc errors."""

    pass


class ARPyCConnectionError(ARPyCError):
    """Connection-related errors."""

    pass


class ARPyCTimeoutError(ARPyCError):
    """Timeout errors."""

    pass


class NotConnectedError(ARPyCConnectionError):
    """Raised when operation requires active connection."""

    pass


class ReconnectFailedError(ARPyCConnectionError):
    """Raised when auto-reconnect fails."""

    pass
