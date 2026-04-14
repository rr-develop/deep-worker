"""Server uptime tracking utilities."""
import time
from typing import Any

# Module-level variable to track server start time
_server_start_time: float = time.time()


def set_server_start_time(start_time: float) -> None:
    """
    Set the server start time.

    Args:
        start_time: Unix timestamp of server start
    """
    global _server_start_time
    _server_start_time = start_time


def calculate_uptime() -> float:
    """
    Calculate server uptime in seconds.

    Returns:
        Uptime in seconds (float)
    """
    return time.time() - _server_start_time


def format_uptime(seconds: float) -> str:
    """
    Format uptime seconds into human-readable string.

    Examples:
        45.5 seconds -> "45s"
        150 seconds -> "2m 30s"
        3920 seconds -> "1h 5m 20s"
        183900 seconds -> "2d 3h 15m"

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted uptime string
    """
    seconds_int = int(seconds)  # Round down

    if seconds_int == 0:
        return "0s"

    days, remaining = divmod(seconds_int, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 and days == 0:  # Don't show seconds if days are present
        parts.append(f"{secs}s")

    return " ".join(parts)


def get_uptime_data() -> dict[str, Any]:
    """
    Get complete uptime data for WebSocket broadcasting.

    Returns:
        Dictionary with uptime_seconds and uptime_formatted
    """
    uptime_seconds = calculate_uptime()
    return {
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": format_uptime(uptime_seconds)
    }
