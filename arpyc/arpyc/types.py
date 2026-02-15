"""Type hints and Protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class EventHandler(Protocol):
    """Protocol for event handlers."""

    def on_event(self, name: str, payload: dict[str, Any]) -> None:
        """Handle incoming event."""
        ...


@runtime_checkable
class AsyncEventHandler(Protocol):
    """Protocol for async event handlers."""

    async def on_event(self, name: str, payload: dict[str, Any]) -> None:
        """Handle incoming event asynchronously."""
        ...


@runtime_checkable
class StreamCallback(Protocol):
    """Protocol for streaming callbacks."""

    def on_chunk(self, chunk: Any) -> None:
        """Handle stream chunk."""
        ...

    def on_done(self) -> None:
        """Handle stream completion."""
        ...

    def on_error(self, error: Exception) -> None:
        """Handle stream error."""
        ...
