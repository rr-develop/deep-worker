"""Real integration test for interactive mode via RPyC.

This test verifies that interactive mode works end-to-end with a real agent,
catching issues like RPyC proxy errors that mock-based tests miss.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# ============================================================================
# CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# DO NOT use 'pip install arpyc' or ANY virtual environment (.venv)!
# ALWAYS import arpyc from the arpyc source directory in this checkout.
# Using installed packages causes version conflicts and compatibility issues.
# This project MUST run globally without venv to ensure correct arpyc version.
# ============================================================================
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession

from .agent_fixture import agent_fixture


@pytest.mark.asyncio
async def test_interactive_send_message_real():
    """Test sending message via RPyC with real agent (no mocks).

    This test catches RPyC proxy errors like:
        AttributeError: cannot access '__reversed__'

    That mock-based tests don't catch.
    """
    # Get API key from environment
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No API key found (need DEEPSEEK_API_KEY or OPENAI_API_KEY)")

    # Use agent_fixture to create real agent
    with agent_fixture(
        agent_name="test_interactive_real",
        working_dir=Path("/tmp"),
        model="deepseek-chat",
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
    ) as agent:
        # Create RPyC session
        session = RpycAsyncSession("127.0.0.1", agent.rpyc_port)
        await session.connect()

        try:
            # Send message via RPyC (exactly how interactive.py does it)
            response = await session.async_call(
                session.root.invoke_agent,
                "What is 2+2?",  # Simple test message
                False  # stream=False
            )

            # CRITICAL: This is where '__reversed__' error would occur
            # if response is an RPyC proxy that doesn't properly expose magic methods

            # Verify response structure (this triggers proxy access)
            assert response is not None, "Response should not be None"

            # Try to access response dict keys (triggers __iter__ on proxy)
            success = response.get("success")
            assert success is not None, "Response should have 'success' key"

            # If we got this far without AttributeError - good!
            # Now verify content
            if success:
                assert "response" in response, "Successful response should have 'response' key"
                response_content = response.get("response")
                assert response_content is not None
                assert "content" in response_content
            else:
                # Error response - that's ok for this test
                # We're just checking for RPyC proxy errors
                pass

        finally:
            await session.disconnect()


@pytest.mark.asyncio
async def test_interactive_response_iteration():
    """Test that response dict can be iterated (tests __iter__ proxy method).

    RPyC proxies sometimes fail on iteration with AttributeError.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No API key found")

    with agent_fixture(
        agent_name="test_interactive_iter",
        working_dir=Path("/tmp"),
        model="deepseek-chat",
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
    ) as agent:
        session = RpycAsyncSession("127.0.0.1", agent.rpyc_port)
        await session.connect()

        try:
            response = await session.async_call(
                session.root.invoke_agent,
                "Hello",
                False
            )

            # CRITICAL: Iterate over response keys
            # This is where '__reversed__' or '__iter__' errors could occur
            keys = list(response.keys())
            assert len(keys) > 0, "Response should have keys"

            # Try iteration patterns that Textual/Python might use
            for key in response:
                # Access each key
                value = response[key]
                # This should not raise AttributeError

            # Try dict comprehension (another iteration pattern)
            items = {k: v for k, v in response.items()}
            assert items is not None

        finally:
            await session.disconnect()
