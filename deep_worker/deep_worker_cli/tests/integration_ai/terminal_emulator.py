"""Terminal emulator for testing dwcli UI in controlled environment.

This module provides classes for running and testing the dwcli terminal UI
by emulating a terminal, capturing output, and sending input programmatically.

Uses pyte library to handle ANSI escape sequences and VT100 terminal emulation.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path
from typing import Optional

import pyte

logger = logging.getLogger(__name__)


class TerminalEmulator:
    """Terminal emulator using pyte for handling ANSI escape sequences.

    This class wraps pyte.Screen to provide a virtual terminal that can:
    - Process ANSI/VT100 escape sequences
    - Track cursor position
    - Maintain screen buffer
    - Provide text snapshots of the terminal state
    """

    def __init__(self, columns: int = 80, lines: int = 24):
        """Initialize terminal emulator.

        Args:
            columns: Terminal width in characters (default: 80)
            lines: Terminal height in lines (default: 24)
        """
        self.columns = columns
        self.lines = lines

        # Create pyte screen and stream
        self.screen = pyte.Screen(columns, lines)
        self.stream = pyte.Stream(self.screen)

        logger.debug(f"Terminal emulator initialized: {columns}x{lines}")

    def feed(self, data: bytes) -> None:
        """Feed terminal data (including escape sequences) to the emulator.

        Args:
            data: Raw bytes from terminal (including ANSI sequences)
        """
        try:
            text = data.decode("utf-8", errors="replace")
            self.stream.feed(text)
        except Exception as e:
            logger.warning(f"Failed to feed data to terminal: {e}")

    def get_screen_text(self, strip: bool = True) -> str:
        """Get current terminal screen as text.

        Args:
            strip: Whether to strip trailing whitespace from each line

        Returns:
            String representation of terminal screen
        """
        lines = []
        for y in range(self.screen.lines):
            line = "".join(self.screen.buffer[y][x].data for x in range(self.screen.columns))
            if strip:
                line = line.rstrip()
            lines.append(line)

        return "\n".join(lines)

    def get_cursor_position(self) -> tuple[int, int]:
        """Get current cursor position.

        Returns:
            Tuple of (line, column) starting from 0
        """
        return (self.screen.cursor.y, self.screen.cursor.x)

    def get_line(self, line_num: int, strip: bool = True) -> str:
        """Get specific line from terminal screen.

        Args:
            line_num: Line number (0-indexed)
            strip: Whether to strip trailing whitespace

        Returns:
            Line text
        """
        if line_num < 0 or line_num >= self.screen.lines:
            return ""

        line = "".join(self.screen.buffer[line_num][x].data for x in range(self.screen.columns))
        return line.rstrip() if strip else line

    def find_text(self, text: str, case_sensitive: bool = True) -> Optional[tuple[int, int]]:
        """Find text in terminal screen.

        Args:
            text: Text to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            Tuple of (line, column) where text starts, or None if not found
        """
        screen_text = self.get_screen_text()
        if not case_sensitive:
            screen_text = screen_text.lower()
            text = text.lower()

        for line_num in range(self.screen.lines):
            line = self.get_line(line_num)
            if not case_sensitive:
                line = line.lower()

            col = line.find(text)
            if col != -1:
                return (line_num, col)

        return None

    def clear(self) -> None:
        """Clear the terminal screen."""
        self.screen.reset()

    def resize(self, columns: int, lines: int) -> None:
        """Resize terminal.

        Args:
            columns: New width
            lines: New height
        """
        self.columns = columns
        self.lines = lines
        self.screen.resize(lines, columns)


class TerminalSession:
    """Base class for managing terminal sessions in emulated terminal.

    This class:
    - Launches command as subprocess with pseudo-terminal
    - Captures and processes terminal output
    - Sends input to terminal
    - Provides screen snapshots
    - Handles cleanup
    """

    def __init__(
        self,
        command: list[str],
        columns: int = 120,
        lines: int = 40,
        working_dir: Optional[Path] = None,
        session_name: str = "terminal_session",
    ):
        """Initialize terminal session.

        Args:
            command: Command to run (e.g., ["dwcli", "agent_name"])
            columns: Terminal width (default: 120)
            lines: Terminal height (default: 40)
            working_dir: Working directory for process
            session_name: Name for logging
        """
        self.command = command
        self.columns = columns
        self.lines = lines
        self.working_dir = working_dir or Path.cwd()
        self.session_name = session_name

        # Terminal emulator
        self.emulator = TerminalEmulator(columns, lines)

        # Process state
        self.process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.pid: Optional[int] = None

        # Read buffer
        self._read_buffer = bytearray()
        self._running = False

        logger.debug(f"TerminalSession created: {session_name}")

    def start(self, timeout: float = 10.0) -> None:
        """Start command in pseudo-terminal.

        Args:
            timeout: Maximum time to wait for command to start

        Raises:
            RuntimeError: If command fails to start
            TimeoutError: If startup exceeds timeout
        """
        logger.info(f"Starting terminal session: {self.session_name}")
        logger.info(f"Command: {' '.join(self.command)}")

        # Fork process with pseudo-terminal
        self.pid, self.master_fd = pty.fork()

        if self.pid == 0:
            # Child process - execute command
            os.chdir(str(self.working_dir))
            os.execvp(self.command[0], self.command)
        else:
            # Parent process - setup terminal
            try:
                # Set terminal size
                winsize = struct.pack("HHHH", self.lines, self.columns, 0, 0)
                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)

                # Make master_fd non-blocking
                flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
                fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                self._running = True

                # Wait for initial output (UI to render)
                start_time = time.time()
                while time.time() - start_time < timeout:
                    self._read_available_output()
                    screen = self.emulator.get_screen_text()

                    # Check if UI is ready (look for typical dwcli indicators)
                    if screen and len(screen.strip()) > 0:
                        logger.info("dwcli UI appears to be ready")
                        time.sleep(0.5)  # Give it moment to fully render
                        self._read_available_output()
                        return

                    time.sleep(0.1)

                raise TimeoutError(f"Command did not start within {timeout}s")

            except Exception as e:
                logger.error(f"Failed to start command: {e}")
                self.close()
                raise RuntimeError(f"Failed to start command: {e}") from e

    def _read_available_output(self) -> bytes:
        """Read all available output from terminal.

        Returns:
            Raw bytes read from terminal
        """
        if not self.master_fd or not self._running:
            return b""

        output = bytearray()

        try:
            while True:
                # Use select to check if data is available
                ready, _, _ = select.select([self.master_fd], [], [], 0.01)

                if not ready:
                    break

                try:
                    chunk = os.read(self.master_fd, 4096)
                    if not chunk:
                        # EOF - process terminated
                        self._running = False
                        break

                    output.extend(chunk)
                    self.emulator.feed(chunk)

                except OSError:
                    break

        except Exception as e:
            logger.warning(f"Error reading output: {e}")

        return bytes(output)

    def send_input(self, text: str, press_enter: bool = False) -> None:
        """Send input to dwcli terminal.

        Args:
            text: Text to send
            press_enter: Whether to send Enter after text
        """
        import os
        debug_input = os.environ.get("DWCLI_DEBUG_INPUT") == "1"

        if not self.master_fd or not self._running:
            raise RuntimeError("Terminal session not running")

        if press_enter:
            text = text + "\n"

        try:
            encoded = text.encode("utf-8")
            bytes_written = os.write(self.master_fd, encoded)

            if debug_input:
                logger.info(f"[PTY_INPUT] Sent {len(encoded)} bytes: {repr(text)}")
                logger.info(f"[PTY_INPUT] Raw bytes: {repr(encoded)}")
                logger.info(f"[PTY_INPUT] Bytes written: {bytes_written}")

            # Give terminal time to process
            time.sleep(0.1)
            output = self._read_available_output()

            if debug_input and output:
                logger.info(f"[PTY_OUTPUT] Got {len(output)} bytes after input")

        except OSError as e:
            logger.error(f"Failed to send input: {e}")
            raise RuntimeError(f"Failed to send input: {e}") from e

    def send_key(self, key: str) -> None:
        """Send special key to terminal.

        Args:
            key: Key name (e.g., "up", "down", "enter", "escape", "tab")
        """
        key_map = {
            "enter": "\r",
            "return": "\r",
            "tab": "\t",
            "escape": "\x1b",
            "esc": "\x1b",
            "backspace": "\x7f",
            "delete": "\x1b[3~",
            "up": "\x1b[A",
            "down": "\x1b[B",
            "right": "\x1b[C",
            "left": "\x1b[D",
            "home": "\x1b[H",
            "end": "\x1b[F",
            "page_up": "\x1b[5~",
            "page_down": "\x1b[6~",
        }

        key_lower = key.lower()
        if key_lower not in key_map:
            raise ValueError(f"Unknown key: {key}. Valid keys: {list(key_map.keys())}")

        import os
        debug_input = os.environ.get("DWCLI_DEBUG_INPUT") == "1"

        if debug_input:
            logger.info(f"[PTY_KEY] Sending key '{key}' -> {repr(key_map[key_lower])}")

        self.send_input(key_map[key_lower])

    def get_screen_text(self, strip: bool = True) -> str:
        """Get current terminal screen as text.

        Args:
            strip: Whether to strip trailing whitespace

        Returns:
            Current terminal screen text
        """
        # Ensure we've read all available output
        self._read_available_output()
        return self.emulator.get_screen_text(strip=strip)

    def wait_for_text(
        self,
        text: str,
        timeout: float = 5.0,
        case_sensitive: bool = True,
        interval: float = 0.1,
    ) -> bool:
        """Wait for specific text to appear on screen.

        Args:
            text: Text to wait for
            timeout: Maximum time to wait
            case_sensitive: Whether to match case
            interval: How often to check (seconds)

        Returns:
            True if text appeared, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            self._read_available_output()

            if self.emulator.find_text(text, case_sensitive) is not None:
                logger.debug(f"Found text: {repr(text)}")
                return True

            time.sleep(interval)

        logger.warning(f"Timeout waiting for text: {repr(text)}")
        return False

    def wait_for_prompt(self, timeout: float = 5.0) -> bool:
        """Wait for input prompt to appear.

        This looks for common prompt indicators in dwcli.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if prompt appeared, False if timeout
        """
        # Look for cursor or common prompt patterns
        start_time = time.time()

        while time.time() - start_time < timeout:
            self._read_available_output()
            screen = self.get_screen_text()

            # Check for signs of input prompt
            # (This may need adjustment based on actual dwcli UI)
            if screen.strip():  # Screen has content
                time.sleep(0.2)  # Brief pause to ensure stable
                return True

            time.sleep(0.1)

        return False

    def get_cursor_position(self) -> tuple[int, int]:
        """Get current cursor position.

        Returns:
            Tuple of (line, column)
        """
        self._read_available_output()
        return self.emulator.get_cursor_position()

    def is_running(self) -> bool:
        """Check if terminal session is still running.

        Returns:
            True if running, False otherwise
        """
        if not self._running or not self.pid:
            return False

        try:
            # Check if process is still alive
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            self._running = False
            return False

    def close(self) -> None:
        """Close terminal session and cleanup resources."""
        logger.info(f"Closing terminal session: {self.session_name}")

        self._running = False

        # Close master FD
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

        # Terminate process if still running
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGTERM)
                # Wait briefly for graceful shutdown
                for _ in range(10):
                    try:
                        os.kill(self.pid, 0)
                        time.sleep(0.1)
                    except (OSError, ProcessLookupError):
                        break
                else:
                    # Force kill if still alive
                    try:
                        os.kill(self.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
            except (OSError, ProcessLookupError):
                pass

            self.pid = None

    def __enter__(self) -> TerminalSession:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._running:
            self.close()


class DwcliTerminalSession(TerminalSession):
    """Convenience class for dwcli terminal sessions."""

    def __init__(
        self,
        agent_name: str,
        columns: int = 120,
        lines: int = 40,
        working_dir: Optional[Path] = None,
    ):
        """Initialize dwcli terminal session.

        Args:
            agent_name: Name of agent to connect to
            columns: Terminal width (default: 120)
            lines: Terminal height (default: 40)
            working_dir: Working directory for dwcli process
        """
        super().__init__(
            command=["dwcli", agent_name],
            columns=columns,
            lines=lines,
            working_dir=working_dir,
            session_name=f"dwcli_{agent_name}",
        )
        self.agent_name = agent_name


class DeepAgentsTerminalSession(TerminalSession):
    """Convenience class for deepagents terminal sessions."""

    def __init__(
        self,
        agent_name: Optional[str] = None,
        columns: int = 120,
        lines: int = 40,
        working_dir: Optional[Path] = None,
        skip_permissions: bool = True,  # ← DEFAULT: True for integration tests
    ):
        """Initialize deepagents terminal session.

        Args:
            agent_name: Optional agent name (if None, uses default)
            columns: Terminal width (default: 120)
            lines: Terminal height (default: 40)
            working_dir: Working directory for deepagents process
            skip_permissions: Skip permission prompts (default: True for tests)
                             ⚠️ CRITICAL: Must be True to prevent approval menu blocking tests!
        """
        command = ["deepagents"]
        if agent_name:
            command.append(agent_name)

        # CRITICAL: Add --auto-approve by default for integration tests
        # Without this, approval menus will block test execution
        if skip_permissions:
            command.append("--auto-approve")

        super().__init__(
            command=command,
            columns=columns,
            lines=lines,
            working_dir=working_dir,
            session_name=f"deepagents_{agent_name or 'default'}",
        )
        self.agent_name = agent_name


def create_terminal_snapshot(
    screen_text: str,
    title: Optional[str] = None,
    show_cursor: bool = False,
    cursor_pos: Optional[tuple[int, int]] = None,
) -> str:
    """Create a formatted terminal snapshot for debugging.

    Args:
        screen_text: Raw screen text
        title: Optional title for snapshot
        show_cursor: Whether to show cursor position
        cursor_pos: Cursor position (line, col)

    Returns:
        Formatted snapshot string
    """
    lines = ["=" * 80]

    if title:
        lines.append(f"  {title}")
        lines.append("=" * 80)

    if show_cursor and cursor_pos:
        lines.append(f"Cursor: Line {cursor_pos[0]}, Column {cursor_pos[1]}")
        lines.append("-" * 80)

    lines.append(screen_text)
    lines.append("=" * 80)

    return "\n".join(lines)
