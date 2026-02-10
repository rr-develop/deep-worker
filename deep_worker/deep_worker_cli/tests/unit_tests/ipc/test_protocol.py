"""Tests for IPC protocol."""

import json

import pytest

from deep_worker_cli.ipc.protocol import IPCProtocol


class TestIPCProtocol:
    """Tests for IPCProtocol encode/decode."""

    def test_encode_message(self):
        """Test encoding a message to JSON."""
        message = {"command": "invoke", "input": "Hello"}
        encoded = IPCProtocol.encode(message)

        # Should be JSON with newline
        assert encoded.endswith(b"\n")
        assert json.loads(encoded.decode("utf-8").strip()) == message

    def test_decode_message(self):
        """Test decoding a JSON message."""
        message = {"command": "status", "data": {}}
        encoded = json.dumps(message).encode("utf-8") + b"\n"

        decoded = IPCProtocol.decode(encoded)
        assert decoded == message

    def test_decode_message_without_newline(self):
        """Test decoding message without newline."""
        message = {"command": "attach"}
        encoded = json.dumps(message).encode("utf-8")

        decoded = IPCProtocol.decode(encoded)
        assert decoded == message

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        original = {
            "command": "invoke",
            "input": "Test message",
            "stream": True,
            "config": {"temperature": 0.7},
        }

        encoded = IPCProtocol.encode(original)
        decoded = IPCProtocol.decode(encoded)

        assert decoded == original

    def test_decode_invalid_json(self):
        """Test decoding invalid JSON raises error."""
        with pytest.raises(json.JSONDecodeError):
            IPCProtocol.decode(b"not valid json\n")

    def test_encode_complex_types(self):
        """Test encoding complex nested structures."""
        message = {
            "command": "invoke",
            "data": {
                "nested": {"key": "value", "number": 42, "list": [1, 2, 3]},
                "null_value": None,
                "bool_value": True,
            },
        }

        encoded = IPCProtocol.encode(message)
        decoded = IPCProtocol.decode(encoded)

        assert decoded == message
