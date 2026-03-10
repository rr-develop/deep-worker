"""Pytest configuration and fixtures for claude_code_provider tests."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_claude_sdk_client() -> Mock:
    """Create a mock ClaudeSDKClient for testing.

    Returns:
        Mock ClaudeSDKClient with connect, disconnect, query, and receive_response methods.
    """
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()
    client.receive_response = AsyncMock(return_value=iter([]))
    return client


@pytest.fixture
def sample_tool_metadata() -> list[dict[str, Any]]:
    """Sample tool metadata for testing.

    Returns:
        List of tool metadata dictionaries.
    """
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                },
                "required": ["arg"]
            }
        },
        {
            "name": "calculator",
            "description": "Calculate numbers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    ]
