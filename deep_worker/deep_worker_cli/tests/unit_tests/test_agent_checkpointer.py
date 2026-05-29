"""Unit tests for AgentProcess checkpointer selection logic."""

import asyncio
import logging
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """Isolate the global ``deep_worker_cli`` logger mutated by
    ``AgentProcess.start()``.

    ``AgentProcess.start()`` configures the real package logger
    (``logging.getLogger("deep_worker_cli")``): it attaches a stderr
    handler and raises the level to INFO so ``log_token_usage()`` output
    is visible in a real agent process (see agent_process.py ~line 575).
    That is correct PRODUCTION behaviour, but in-process tests that drive
    ``start()`` leak it into the shared interpreter. Child loggers
    (``deep_worker_cli.file_ops`` / ``.clipboard`` / ``.image_utils``)
    then inherit the INFO effective level, and later tests that assert on
    DEBUG-level records via ``caplog`` (e.g. test_exception_handling.py)
    silently capture nothing and fail. Snapshot the level + handlers and
    restore them after each test so the side effect cannot escape.
    """
    pkg_logger = logging.getLogger("deep_worker_cli")
    saved_level = pkg_logger.level
    saved_handlers = list(pkg_logger.handlers)
    try:
        yield
    finally:
        pkg_logger.setLevel(saved_level)
        for handler in list(pkg_logger.handlers):
            if handler not in saved_handlers:
                pkg_logger.removeHandler(handler)


async def _run_start_until_create_cli_agent(process: AgentProcess, mock_create) -> None:
    """Run `process.start()` in a task, wait for `create_cli_agent` to
    be invoked (that's the point each checkpointer test actually
    verifies), then signal shutdown so `start()` can return cleanly.

    Avoids deadlocking on `await self.shutdown_event.wait()` at the
    end of `start()` (see agent_process.py ~line 476).
    """
    task = asyncio.create_task(process.start())
    # Poll the mock in a non-blocking way. We use `asyncio.sleep(0)`
    # to yield control to `start()` without introducing a real timer.
    # After a bounded number of yields, we consider the test failed
    # rather than hang forever.
    for _ in range(2000):  # ~2000 * <=1ms yields ≈ worst case 2s
        if mock_create.called:
            break
        await asyncio.sleep(0.001)
    else:
        process.shutdown_event.set()
        await task
        raise AssertionError(
            "create_cli_agent was not called within the yield budget; "
            "something in `start()` is blocking before agent init."
        )

    # Let `start()` exit its `shutdown_event.wait()`.
    process.shutdown_event.set()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.fixture
def agent_config(tmp_path):
    """Create test agent config."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    return AgentConfig(
        model="claude-sonnet-4-5-20250929",
        working_dir=working_dir,
    )


class TestAgentProcessInit:
    """Tests for AgentProcess initialization with checkpointer flags."""

    def test_init_defaults(self, agent_config):
        """Test default initialization without flags."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        assert process.memory_only is False
        assert process.use_sqlite is False
        assert process._checkpointer_ctx is None

    def test_init_with_memory_only(self, agent_config):
        """Test initialization with memory_only flag."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            memory_only=True,
        )

        assert process.memory_only is True
        assert process.use_sqlite is False

    def test_init_with_use_sqlite(self, agent_config):
        """Test initialization with use_sqlite flag."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            use_sqlite=True,
        )

        assert process.memory_only is False
        assert process.use_sqlite is True

    def test_init_with_both_flags_raises_error(self, agent_config):
        """Test that both flags raise ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            AgentProcess(
                name="test-agent",
                config=agent_config,
                memory_only=True,
                use_sqlite=True,
            )


class TestCheckpointerSelectionLogic:
    """Tests for checkpointer selection priority logic."""

    @pytest.mark.asyncio
    async def test_use_sqlite_flag_overrides_test_detection(self, agent_config, monkeypatch):
        """Test --use-sqlite flag forces SQLite even in test environment."""
        # Set test environment indicator
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            use_sqlite=True,
        )

        # Mock get_checkpointer and create_cli_agent
        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer_ctx.__aenter__ = AsyncMock(return_value=mock_checkpointer)

        with patch("deep_worker_cli.sessions.get_checkpointer", return_value=mock_checkpointer_ctx):
            with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
                mock_create.return_value = (MagicMock(), MagicMock())

                await _run_start_until_create_cli_agent(process, mock_create)

                # Verify create_cli_agent was called with checkpointer (not None)
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is mock_checkpointer
                assert call_kwargs["checkpointer"] is not None

    @pytest.mark.asyncio
    async def test_memory_only_flag_overrides_production(self, agent_config, monkeypatch):
        """Test --memory-only flag forces memory even in production."""
        # Clear test indicators (simulate production)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("DEEP_WORKER_TEST_MODE", raising=False)

        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            memory_only=True,
        )

        with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            with patch("deep_worker_cli.utils.is_running_tests", return_value=False):
                await _run_start_until_create_cli_agent(process, mock_create)

                # Verify create_cli_agent was called with checkpointer=None
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is None

    @pytest.mark.asyncio
    async def test_test_environment_defaults_to_memory(self, agent_config):
        """Test that test environment automatically uses InMemorySaver."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            with patch("deep_worker_cli.utils.is_running_tests", return_value=True):
                await _run_start_until_create_cli_agent(process, mock_create)

                # Verify checkpointer is None (InMemorySaver fallback)
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is None

    @pytest.mark.asyncio
    async def test_production_defaults_to_sqlite(self, agent_config):
        """Test that production environment uses SQLite by default."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer_ctx.__aenter__ = AsyncMock(return_value=mock_checkpointer)

        with patch("deep_worker_cli.sessions.get_checkpointer", return_value=mock_checkpointer_ctx):
            with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
                mock_create.return_value = (MagicMock(), MagicMock())
                with patch("deep_worker_cli.utils.is_running_tests", return_value=False):
                    await _run_start_until_create_cli_agent(process, mock_create)

                    # Verify create_cli_agent was called with checkpointer
                    mock_create.assert_called_once()
                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["checkpointer"] is mock_checkpointer
                    assert call_kwargs["checkpointer"] is not None


class TestCheckpointerCleanup:
    """Tests for checkpointer cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_closes_checkpointer(self, agent_config):
        """Test cleanup() closes checkpointer context manager."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        # Mock checkpointer context manager
        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer_ctx.__aexit__ = AsyncMock()
        process._checkpointer_ctx = mock_checkpointer_ctx

        # Call cleanup
        await process.cleanup()

        # Verify __aexit__ was called
        mock_checkpointer_ctx.__aexit__.assert_called_once_with(None, None, None)
        assert process._checkpointer_ctx is None

    @pytest.mark.asyncio
    async def test_cleanup_without_checkpointer(self, agent_config):
        """Test cleanup() when no checkpointer exists."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        # No checkpointer
        process._checkpointer_ctx = None

        # Should not raise
        await process.cleanup()

        assert process._checkpointer_ctx is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_exception(self, agent_config, capsys):
        """Test cleanup() handles exception gracefully."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        # Mock checkpointer that raises exception
        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer_ctx.__aexit__ = AsyncMock(side_effect=Exception("DB error"))
        process._checkpointer_ctx = mock_checkpointer_ctx

        # Should not raise, but print warning
        await process.cleanup()

        # Verify error message was printed
        captured = capsys.readouterr()
        assert "Error closing checkpointer" in captured.err

        # Context should be cleared even after error
        assert process._checkpointer_ctx is None


class TestPriorityLogic:
    """Tests to verify priority order of checkpointer selection."""

    @pytest.mark.asyncio
    async def test_priority_1_use_sqlite_beats_all(self, agent_config, monkeypatch):
        """Test priority 1: --use-sqlite overrides everything."""
        # Set test indicators
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            use_sqlite=True,
        )

        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer_ctx.__aenter__ = AsyncMock(return_value=mock_checkpointer)

        with patch("deep_worker_cli.sessions.get_checkpointer", return_value=mock_checkpointer_ctx):
            with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
                mock_create.return_value = (MagicMock(), MagicMock())

                await _run_start_until_create_cli_agent(process, mock_create)

                # Should use SQLite despite test environment
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is not None

    @pytest.mark.asyncio
    async def test_priority_2_memory_only_beats_test_and_prod(self, agent_config):
        """Test priority 2: --memory-only overrides test detection and production."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            memory_only=True,
        )

        with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            # Test both in production and test mode
            with patch("deep_worker_cli.utils.is_running_tests", return_value=False):
                await _run_start_until_create_cli_agent(process, mock_create)

                # Should use memory despite production
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is None

    @pytest.mark.asyncio
    async def test_priority_3_test_detection_beats_production(self, agent_config):
        """Test priority 3: test detection beats default production."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            with patch("deep_worker_cli.utils.is_running_tests", return_value=True):
                await _run_start_until_create_cli_agent(process, mock_create)

                # Should use memory in test environment
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["checkpointer"] is None

    @pytest.mark.asyncio
    async def test_priority_4_default_production_uses_sqlite(self, agent_config):
        """Test priority 4: default production uses SQLite."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
        )

        mock_checkpointer_ctx = AsyncMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer_ctx.__aenter__ = AsyncMock(return_value=mock_checkpointer)

        with patch("deep_worker_cli.sessions.get_checkpointer", return_value=mock_checkpointer_ctx):
            with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
                mock_create.return_value = (MagicMock(), MagicMock())
                with patch("deep_worker_cli.utils.is_running_tests", return_value=False):
                    await _run_start_until_create_cli_agent(process, mock_create)

                    # Should use SQLite in normal production
                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["checkpointer"] is not None
