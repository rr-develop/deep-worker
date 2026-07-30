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

    def test_server_restart_schedules_do_restart_in_background_not_inline(self):
        """Endpoint must protect response delivery by scheduling restart.

        ``do_restart`` sleeps briefly, then calls ``os._exit(0)``. The endpoint
        must therefore pass it to a background thread and return the JSON
        response; calling it inline would kill the process before the response
        can flush.
        """
        client = TestClient(app)

        with patch('app.routers.server.do_restart') as mock_do_restart, \
             patch('app.routers.server.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            response = client.post("/api/server/restart")

            assert response.status_code == 200
            mock_do_restart.assert_not_called()
            mock_thread.assert_called_once()
            call_kwargs = mock_thread.call_args[1]
            assert call_kwargs.get('target') is mock_do_restart
            assert call_kwargs.get('daemon') is True, "Thread should be daemon"
            mock_thread_instance.start.assert_called_once()


class TestDoRestartFunction:
    """Tests for do_restart() function logic."""

    def test_do_restart_does_not_kill_parent_process(self):
        """
        Test: do_restart() must not kill its parent shell/process.

        The web UI restart endpoint used to kill ``os.getppid()``. When the
        server was launched by a shell/supervisor wrapper this killed the
        wrapper, started the replacement before the old server released the
        port, and left the server's listening address down. Restart must
        instead launch a detached helper and exit the current server
        process only.
        """
        from app.routers.server import do_restart

        with patch('subprocess.Popen') as mock_popen, \
             patch('os.kill') as mock_kill, \
             patch('os._exit') as mock_exit, \
             patch('os.getpid', return_value=12346), \
             patch('os.getppid', return_value=12345), \
             patch('time.sleep'), \
             patch('sys.argv', ['/tmp/checkout/web_app/run.py']):
            mock_popen.return_value = Mock(pid=99999)

            do_restart()

            mock_kill.assert_not_called()
            mock_exit.assert_called_once_with(0)

    def test_do_restart_starts_detached_helper_for_current_checkout(self):
        """Test that do_restart() starts a detached helper for this checkout."""
        from app.routers.server import do_restart

        run_py = "/tmp/checkout/web_app/run.py"

        with patch('subprocess.Popen') as mock_popen, \
             patch('os._exit') as mock_exit, \
             patch('os.getpid', return_value=12346), \
             patch('time.sleep'), \
             patch('sys.argv', [run_py]):
            mock_popen_instance = Mock(pid=99999)
            mock_popen.return_value = mock_popen_instance

            do_restart()

            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            argv = call_args[0][0]

            assert argv[:3] == [sys.executable, "-c", argv[2]]
            assert argv[3] == "12346"
            assert argv[4] == run_py
            assert call_args[1].get('cwd') == str(Path(run_py).parent)
            assert call_args[1].get('start_new_session') is True, "Should start new session"
            assert call_args[1].get('stdin') is not None, "Should redirect stdin"
            assert call_args[1].get('stdout') is not None, "Should redirect stdout"
            assert call_args[1].get('stderr') is not None, "Should redirect stderr"
            mock_exit.assert_called_once_with(0)
