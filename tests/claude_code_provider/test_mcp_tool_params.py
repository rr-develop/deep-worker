# pyright: reportPrivateUsage=false, reportUnusedImport=false
"""Tests for MCP tool schema generation.

Verifies that bind_tools generates correct JSON schemas from tool signatures,
especially for StructuredTool functions using ToolRuntime (which causes
CallableSchema error in Pydantic — fallback to inspect.signature).
"""

import pytest

from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata


class TestSchemaGeneration:
    """Schema must be generated from function signature when Pydantic fails."""

    def test_schema_from_structuredtool_signature(self):
        """bind_tools should generate schema from inspect.signature fallback."""
        from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
        from deepagents.backends.local_shell import LocalShellBackend
        from deepagents.middleware.filesystem import FilesystemMiddleware

        backend = LocalShellBackend(root_dir='/tmp')
        fs = FilesystemMiddleware(backend=backend)
        ls_tool = fs._create_ls_tool()
        glob_tool = fs._create_glob_tool()
        grep_tool = fs._create_grep_tool()
        execute_tool = fs._create_execute_tool()

        model = ClaudeCodeChatModel(model="sonnet")
        bound = model.bind_tools([ls_tool, glob_tool, grep_tool, execute_tool])

        # Check schemas are NOT empty
        for meta in bound._tools_metadata:
            schema = meta["input_schema"]
            assert schema.get("properties"), (
                f"Tool '{meta['name']}' has empty schema: {schema}"
            )

    def test_ls_schema_has_path(self):
        """ls schema must have 'path' parameter."""
        from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
        from deepagents.backends.local_shell import LocalShellBackend
        from deepagents.middleware.filesystem import FilesystemMiddleware

        backend = LocalShellBackend(root_dir='/tmp')
        fs = FilesystemMiddleware(backend=backend)
        ls_tool = fs._create_ls_tool()

        model = ClaudeCodeChatModel(model="sonnet")
        bound = model.bind_tools([ls_tool])

        schema = bound._tools_metadata[0]["input_schema"]
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_execute_schema_has_command(self):
        """execute schema must have 'command' parameter."""
        from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
        from deepagents.backends.local_shell import LocalShellBackend
        from deepagents.middleware.filesystem import FilesystemMiddleware

        backend = LocalShellBackend(root_dir='/tmp')
        fs = FilesystemMiddleware(backend=backend)
        execute_tool = fs._create_execute_tool()

        model = ClaudeCodeChatModel(model="sonnet")
        bound = model.bind_tools([execute_tool])

        schema = bound._tools_metadata[0]["input_schema"]
        assert "command" in schema["properties"]
        assert "command" in schema["required"]

    def test_glob_schema_has_pattern(self):
        """glob schema must have 'pattern' parameter."""
        from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
        from deepagents.backends.local_shell import LocalShellBackend
        from deepagents.middleware.filesystem import FilesystemMiddleware

        backend = LocalShellBackend(root_dir='/tmp')
        fs = FilesystemMiddleware(backend=backend)
        glob_tool = fs._create_glob_tool()

        model = ClaudeCodeChatModel(model="sonnet")
        bound = model.bind_tools([glob_tool])

        schema = bound._tools_metadata[0]["input_schema"]
        assert "pattern" in schema["properties"]
        assert "pattern" in schema["required"]
