"""Unit tests for DiffWidget display in InteractiveApp.

Tests cover:
- Detection of file operations in tool results
- Diff extraction and display
- EnhancedDiff widget rendering
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_worker_cli.cli.interactive import InteractiveApp


class TestDiffDisplay:
    """Test diff display functionality."""

    @pytest.mark.asyncio
    async def test_file_diff_message_displays_widget(self):
        """Test file_diff message displays EnhancedDiff widget."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            mounted_widgets = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_widgets.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Simulate file_diff message from agent
            diff_content = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    print("old")
+    print("new")
+    return True
"""

            await app._handle_realtime_message({
                "type": "file_diff",
                "tool_name": "edit_file",
                "tool_call_id": "call_123",
                "diff_content": diff_content,
                "file_path": "test.py"
            })

            # Should mount EnhancedDiff widget
            assert len(mounted_widgets) >= 1
            from deep_worker_cli.widgets.diff import EnhancedDiff
            diff_widgets = [w for w in mounted_widgets if isinstance(w, EnhancedDiff)]
            assert len(diff_widgets) == 1

    @pytest.mark.asyncio
    async def test_multiple_diff_messages(self):
        """Test multiple file_diff messages display multiple widgets."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            mounted_widgets = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_widgets.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Simulate multiple file diffs
            for i in range(3):
                await app._handle_realtime_message({
                    "type": "file_diff",
                    "diff_content": f"diff for file {i}",
                    "file_path": f"file{i}.py"
                })

            # Should mount 3 diff widgets
            from deep_worker_cli.widgets.diff import EnhancedDiff
            diff_widgets = [w for w in mounted_widgets if isinstance(w, EnhancedDiff)]
            assert len(diff_widgets) == 3


class TestStreamProcessorDiffExtraction:
    """Test diff extraction in stream processor."""

    @pytest.mark.asyncio
    async def test_detect_edit_file_tool_result(self):
        """Test detecting edit_file tool with diff in result."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import ToolMessage

        # ToolMessage with diff content (simulating edit_file result)
        diff_text = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old line
+new line
"""

        msg = ToolMessage(
            content=diff_text,
            tool_call_id="call_123"
        )
        msg.name = "edit_file"
        msg.status = "success"

        notifications = []

        async def notify(msg_dict):
            notifications.append(msg_dict)

        # Process with metadata containing tool name
        await process_message_chunk(
            message=msg,
            metadata={"tool_name": "edit_file"},
            notify_callback=notify,
            tool_call_buffer={}
        )

        # Should send file_diff notification if content has diff markers
        file_diffs = [n for n in notifications if n.get("type") == "file_diff"]
        assert len(file_diffs) == 1
        assert file_diffs[0]["diff_content"] == diff_text
        assert file_diffs[0]["tool_name"] == "edit_file"

    @pytest.mark.asyncio
    async def test_no_diff_for_other_tools(self):
        """Test no diff extraction for non-file-operation tools."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(
            content="Some tool output",
            tool_call_id="call_123"
        )
        msg.name = "bash"
        msg.status = "success"

        notifications = []

        async def notify(msg_dict):
            notifications.append(msg_dict)

        await process_message_chunk(
            message=msg,
            metadata={"tool_name": "bash"},
            notify_callback=notify,
            tool_call_buffer={}
        )

        # Should NOT send file_diff
        file_diffs = [n for n in notifications if n.get("type") == "file_diff"]
        assert len(file_diffs) == 0

    @pytest.mark.asyncio
    async def test_detect_write_file_tool(self):
        """Test detecting write_file tool."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import ToolMessage

        diff_text = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def new_function():
+    pass
"""

        msg = ToolMessage(
            content=diff_text,
            tool_call_id="call_456"
        )
        msg.name = "write_file"
        msg.status = "success"

        notifications = []

        async def notify(msg_dict):
            notifications.append(msg_dict)

        await process_message_chunk(
            message=msg,
            metadata={"tool_name": "write_file"},
            notify_callback=notify,
            tool_call_buffer={}
        )

        # Should send file_diff
        file_diffs = [n for n in notifications if n.get("type") == "file_diff"]
        assert len(file_diffs) == 1
        assert file_diffs[0]["tool_name"] == "write_file"
