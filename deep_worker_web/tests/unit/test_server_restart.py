"""Tests for server restart endpoint."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.main import app


class TestServerRestartEndpoint:
    """Tests for /api/server/restart endpoint."""

    def test_server_restart_endpoint_exists(self):
        """Test that the restart endpoint is registered."""
        client = TestClient(app)

        # Mock threading to prevent actual restart
        with patch('threading.Thread'):
            response = client.post("/api/server/restart")
            # Should not be 404 (endpoint exists)
            assert response.status_code != 404, "Restart endpoint should exist"

    def test_server_restart_endpoint_returns_success(self):
        """
        Test that server restart endpoint returns success response.

        The restart endpoint should:
        1. Return a success response immediately
        2. Start the restart process in the background
        3. Not crash or hang
        """
        client = TestClient(app)

        # Mock the restart thread to prevent actual restart during test
        with patch('app.routers.server.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            # Call the restart endpoint
            response = client.post("/api/server/restart")

            # Should return success immediately
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data["success"] is True, f"Expected success=True, got {data}"
            assert "message" in data, "Response should include message"
            assert "restart" in data["message"].lower(), "Message should mention restart"

            # Should have started restart thread
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_server_restart_calls_do_restart_function(self):
        """Test that restart endpoint calls do_restart in background thread."""
        client = TestClient(app)

        # Mock threading.Thread to capture the target function
        with patch('app.routers.server.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            response = client.post("/api/server/restart")

            # Verify Thread was created with do_restart as target
            mock_thread.assert_called_once()
            call_kwargs = mock_thread.call_args[1]
            assert 'target' in call_kwargs, "Thread should have target parameter"
            assert call_kwargs['target'].__name__ == 'do_restart', (
                f"Expected do_restart, got {call_kwargs['target'].__name__}"
            )
            assert call_kwargs.get('daemon') is True, "Thread should be daemon"


class TestDoRestartFunction:
    """Tests for do_restart() function logic."""

    def test_do_restart_kills_parent_process(self):
        """
        Test: do_restart() should kill parent process.

        This ensures the port is freed for the new process.
        """
        from app.routers.server import do_restart

        # Mock psutil.Process to simulate parent
        mock_current_process = Mock()
        mock_parent_process = Mock()

        # Parent process
        mock_parent_process.name.return_value = "python3"
        mock_parent_process.pid = 12345

        mock_current_process.parent.return_value = mock_parent_process
        mock_current_process.pid = 12346
        mock_current_process.name.return_value = "python3"

        with patch('psutil.Process') as mock_process_class, \
             patch('subprocess.Popen') as mock_popen, \
             patch('os.kill') as mock_kill, \
             patch('os._exit') as mock_exit, \
             patch('os.getpid', return_value=12346), \
             patch('os.getppid', return_value=12345), \
             patch('time.sleep'):

            mock_process_class.return_value = mock_current_process
            mock_popen.return_value = Mock(pid=99999)

            # Mock os.kill to simulate successful kill
            # First call: SIGTERM succeeds
            # Second call (pid, 0): raise OSError to indicate process is dead
            mock_kill.side_effect = [None, OSError("Process not found")]

            # Call do_restart
            do_restart()

            # Parent process should be killed with SIGTERM
            assert mock_kill.call_count >= 1, "Should call os.kill at least once"
            first_call = mock_kill.call_args_list[0]
            assert first_call[0][0] == 12345, f"Should kill parent PID 12345, got {first_call[0][0]}"

            # Should call os._exit after killing parent
            mock_exit.assert_called_once_with(0)

    def test_do_restart_starts_new_process(self):
        """Test that do_restart() starts a new server process."""
        from app.routers.server import do_restart

        mock_current_process = Mock()
        mock_parent_process = Mock()
        mock_parent_process.name.return_value = "python3"
        mock_parent_process.pid = 12345
        mock_current_process.parent.return_value = mock_parent_process

        with patch('psutil.Process') as mock_process_class, \
             patch('subprocess.Popen') as mock_popen, \
             patch('os.kill') as mock_kill, \
             patch('os._exit') as mock_exit, \
             patch('os.getpid', return_value=12346), \
             patch('os.getppid', return_value=12345), \
             patch('time.sleep'):

            mock_process_class.return_value = mock_current_process
            mock_popen_instance = Mock(pid=99999)
            mock_popen.return_value = mock_popen_instance

            # Mock kill to simulate process death
            mock_kill.side_effect = [None, OSError()]

            do_restart()

            # Should have started new process
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args

            # Verify subprocess.Popen arguments
            assert call_args[1].get('start_new_session') is True, "Should start new session"
            assert call_args[1].get('stdin') is not None, "Should redirect stdin"
            assert call_args[1].get('stdout') is not None, "Should redirect stdout"
            assert call_args[1].get('stderr') is not None, "Should redirect stderr"
