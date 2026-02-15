"""Tests for TaskManager prompts module.

These tests verify that prompt formatting and parsing work correctly.
"""

import pytest
from deep_worker_cli.task_manager.prompts import (
    format_evaluation_prompt,
    format_followup_prompt,
    parse_manager_response,
    TASK_MANAGER_SYSTEM_PROMPT,
)


class TestPromptFormatting:
    """Tests for prompt formatting functions."""

    def test_format_evaluation_prompt_basic(self):
        """Test basic evaluation prompt formatting."""
        prompt = format_evaluation_prompt(
            initial_request="Create a hello world function",
            last_response="I created the function",
            session_file_path=None,
            project_path=None
        )

        assert "Task Manager AI" in prompt
        assert "Create a hello world function" in prompt
        assert "I created the function" in prompt
        assert "STATUS:" in prompt
        assert "REASONING:" in prompt
        assert "MESSAGE:" in prompt

    def test_format_evaluation_prompt_with_paths(self):
        """Test evaluation prompt with session and project paths."""
        prompt = format_evaluation_prompt(
            initial_request="Test request",
            last_response="Test response",
            session_file_path="/tmp/session.jsonl",
            project_path="/tmp/project"
        )

        assert "/tmp/session.jsonl" in prompt
        assert "/tmp/project" in prompt

    def test_format_followup_prompt(self):
        """Test follow-up prompt formatting."""
        prompt = format_followup_prompt(
            last_response="Updated response after correction"
        )

        assert "SUBORDINATE'S RESPONSE AFTER CORRECTION" in prompt
        assert "Updated response after correction" in prompt
        assert "Re-evaluate" in prompt


class TestManagerResponseParsing:
    """Tests for parsing manager AI responses."""

    def test_parse_done_status(self):
        """Test parsing 'done' status."""
        response = """STATUS: done
REASONING: Task is fully completed
MESSAGE: NONE"""

        status, message = parse_manager_response(response)

        assert status == 'done'
        assert message is None  # MESSAGE should be None for 'done'

    def test_parse_in_progress_status(self):
        """Test parsing 'in_progress' status."""
        response = """STATUS: in_progress
REASONING: Tests are missing
MESSAGE: Please add tests for the function"""

        status, message = parse_manager_response(response)

        assert status == 'in_progress'
        assert message == "Please add tests for the function"

    def test_parse_unable_status(self):
        """Test parsing 'unable' status."""
        response = """STATUS: unable
REASONING: Contradictory requirements
MESSAGE: Cannot be both sync and async simultaneously"""

        status, message = parse_manager_response(response)

        assert status == 'unable'
        assert message == "Cannot be both sync and async simultaneously"

    def test_parse_multiline_message(self):
        """Test parsing multiline MESSAGE field."""
        response = """STATUS: in_progress
REASONING: Multiple issues found
MESSAGE: Please fix the following:
1. Add error handling
2. Add documentation
3. Add tests"""

        status, message = parse_manager_response(response)

        assert status == 'in_progress'
        assert "Add error handling" in message
        assert "Add documentation" in message
        assert "Add tests" in message

    def test_parse_empty_result_raises_error(self):
        """Test that empty result raises ValueError."""
        with pytest.raises(ValueError, match="Empty evaluation result"):
            parse_manager_response("")

    def test_parse_missing_status_raises_error(self):
        """Test that missing STATUS raises ValueError."""
        response = """REASONING: Some reasoning
MESSAGE: Some message"""

        with pytest.raises(ValueError, match="Missing STATUS field"):
            parse_manager_response(response)

    def test_parse_invalid_status_raises_error(self):
        """Test that invalid STATUS value raises ValueError."""
        response = """STATUS: invalid_status
REASONING: Some reasoning
MESSAGE: Some message"""

        with pytest.raises(ValueError, match="Invalid STATUS value"):
            parse_manager_response(response)

    def test_parse_in_progress_without_message_gets_default(self):
        """Test that in_progress without MESSAGE gets default message."""
        response = """STATUS: in_progress
REASONING: Work is incomplete"""

        status, message = parse_manager_response(response)

        assert status == 'in_progress'
        assert message == "Additional work is needed to complete the initial request."

    def test_parse_unable_without_message_gets_default(self):
        """Test that unable without MESSAGE gets default message."""
        response = """STATUS: unable
REASONING: Cannot be completed"""

        status, message = parse_manager_response(response)

        assert status == 'unable'
        assert message == "Task cannot be completed due to identified constraints."


class TestSystemPrompt:
    """Tests for TaskManager system prompt."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert TASK_MANAGER_SYSTEM_PROMPT
        assert "expert at evaluating" in TASK_MANAGER_SYSTEM_PROMPT
        assert "IF THE USER ASKS NOT TO DO SOMETHING" in TASK_MANAGER_SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
