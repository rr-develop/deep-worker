"""IPC protocol for deep_worker communication."""

import json
from typing import Any


class IPCProtocol:
    """Protocol for encoding/decoding IPC messages.

    Messages are JSON-encoded with newline delimiters for framing.
    """

    @staticmethod
    def encode(message: dict[str, Any]) -> bytes:
        """Encode a message to bytes.

        Args:
            message: Dictionary to encode

        Returns:
            JSON-encoded bytes with newline terminator
        """
        json_str = json.dumps(message)
        return json_str.encode("utf-8") + b"\n"

    @staticmethod
    def decode(data: bytes) -> dict[str, Any]:
        """Decode bytes to a message.

        Args:
            data: JSON-encoded bytes

        Returns:
            Decoded dictionary

        Raises:
            json.JSONDecodeError: If data is not valid JSON
        """
        # Strip newline if present
        text = data.decode("utf-8").strip()
        return json.loads(text)
