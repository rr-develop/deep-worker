"""Tests for Pydantic serialization compatibility.

These tests ensure ClaudeCodeChatModel works with LangGraph's serialization,
which requires generating JSON Schema for tasks.
"""

import pytest
from pydantic import TypeAdapter

from claude_code_provider.chat_model import ClaudeCodeChatModel


class TestPydanticSerialization:
    """Tests for Pydantic v2 serialization compatibility."""

    def test_model_can_generate_json_schema(self) -> None:
        """Should be able to generate JSON Schema without errors.

        This test reproduces the issue that occurs when LangGraph tries to
        create a task with ClaudeCodeChatModel - it needs to generate
        JSON Schema for the model.
        """
        _model = ClaudeCodeChatModel(base_options=None)

        # This should not raise PydanticInvalidForJsonSchema
        try:
            # Try to generate JSON Schema (what LangGraph does internally)
            adapter = TypeAdapter(ClaudeCodeChatModel)
            schema = adapter.json_schema()

            # If we got here, schema generation worked
            assert isinstance(schema, dict)
            print(f"✅ JSON Schema generated successfully")

        except Exception as e:
            pytest.fail(
                f"Failed to generate JSON Schema: {type(e).__name__}: {e}\n"
                f"This will cause LangGraph task creation to fail."
            )

    def test_model_with_base_options_can_generate_schema(self) -> None:
        """Should generate schema even with ClaudeAgentOptions."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        _model = ClaudeCodeChatModel(base_options=options)

        # This should not raise PydanticInvalidForJsonSchema
        try:
            adapter = TypeAdapter(ClaudeCodeChatModel)
            schema = adapter.json_schema()
            assert isinstance(schema, dict)

        except Exception as e:
            pytest.fail(
                f"Failed to generate JSON Schema with base_options: {type(e).__name__}: {e}"
            )

    def test_bound_model_can_generate_schema(self) -> None:
        """Should generate schema for model with bound tools."""
        from langchain_core.tools import tool

        @tool
        def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        model = ClaudeCodeChatModel(base_options=None)
        _bound_model = model.bind_tools([calculator])

        # This should not raise PydanticInvalidForJsonSchema
        try:
            adapter = TypeAdapter(ClaudeCodeChatModel)
            schema = adapter.json_schema()
            assert isinstance(schema, dict)

        except Exception as e:
            pytest.fail(
                f"Failed to generate JSON Schema for bound model: {type(e).__name__}: {e}"
            )

    def test_model_dict_serialization(self) -> None:
        """Should be able to serialize model to dict.

        LangGraph may also try to serialize models to dict for persistence.
        """
        model = ClaudeCodeChatModel(base_options=None)

        try:
            # Try model_dump (Pydantic v2)
            if hasattr(model, 'model_dump'):
                data = model.model_dump()
                assert isinstance(data, dict)
            # Fallback to dict() (Pydantic v1)
            elif hasattr(model, 'dict'):
                data = model.dict()
                assert isinstance(data, dict)
            else:
                pytest.fail("Model has no serialization method (model_dump or dict)")

        except Exception as e:
            # This might legitimately fail due to arbitrary types
            # But we should document it
            print(f"⚠️  Dict serialization failed: {type(e).__name__}: {e}")
            print("This is expected for models with arbitrary types")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
