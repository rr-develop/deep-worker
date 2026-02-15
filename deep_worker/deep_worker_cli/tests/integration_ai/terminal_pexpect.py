"""PTY-based terminal session using pexpect for true terminal emulation.

This module provides terminal session management using pexpect, which properly
handles interactive programs like Textual applications that require real PTY
terminal protocols (raw mode, terminal control sequences, etc.).

Unlike pyte-based emulation which only renders ANSI codes, pexpect provides
a full PTY connection to the subprocess, allowing proper terminal interaction.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import pexpect
except ImportError:
    pexpect = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PexpectTerminalSession:
    """Terminal session using pexpect for real PTY interaction.

    This class provides true PTY-based terminal interaction, which is necessary
    for interactive applications like Textual that require:
    - Raw mode terminal settings
    - Proper signal handling
    - Terminal control sequences
    - Real keyboard input processing

    Unlike pyte-based emulation, this uses an actual PTY connection.
    """

    def __init__(
        self,
        command: list[str],
        columns: int = 120,
        lines: int = 40,
        working_dir: Optional[Path] = None,
        session_name: str = "pexpect_session",
        timeout: float = 10.0,
    ):
        """Initialize pexpect-based terminal session.

        Args:
            command: Command to run (e.g., ["dwcli", "agent_name"])
            columns: Terminal width (default: 120)
            lines: Terminal height (default: 40)
            working_dir: Working directory for process
            session_name: Name for logging
            timeout: Default timeout for expect operations

        Raises:
            ImportError: If pexpect is not available
        """
        if pexpect is None:
            raise ImportError("pexpect is required for PexpectTerminalSession")

        self.command = command
        self.columns = columns
        self.lines = lines
        self.working_dir = working_dir or Path.cwd()
        self.session_name = session_name
        self.timeout = timeout

        self.child: Optional[pexpect.spawn] = None
        self._last_output = ""

        logger.debug(f"PexpectTerminalSession created: {session_name}")

    def start(self, timeout: float | None = None) -> None:
        """Start command in real PTY.

        Args:
            timeout: Maximum time to wait for command to start (overrides default)

        Raises:
            RuntimeError: If command fails to start
        """
        if timeout is None:
            timeout = self.timeout

        logger.info(f"Starting PTY session: {self.session_name}")
        logger.info(f"Command: {' '.join(self.command)}")
        logger.info(f"Terminal size: {self.lines}x{self.columns}")

        try:
            # Spawn process in PTY
            self.child = pexpect.spawn(
                self.command[0],
                args=self.command[1:],
                timeout=timeout,
                cwd=str(self.working_dir),
            )

            # Set terminal size
            self.child.setwinsize(self.lines, self.columns)

            # Set encoding
            self.child.encoding = "utf-8"

            logger.info(f"PTY session started (PID: {self.child.pid})")

            # Wait for command to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Try to read available output
                    self.child.expect(pexpect.TIMEOUT, timeout=0.1)
                except pexpect.TIMEOUT:
                    pass
                except pexpect.EOF:
                    raise RuntimeError(f"Command exited immediately")

                # Check if we have any output
                if self.child.buffer and len(self.child.buffer.strip()) > 0:
                    logger.info("PTY session ready")
                    return

                time.sleep(0.1)

            raise TimeoutError(f"Command did not respond within {timeout}s")

        except Exception as e:
            self.close()
            logger.error(f"Failed to start PTY session: {e}")
            raise RuntimeError(f"Failed to start PTY session: {e}") from e

    def send_input(self, text: str, wait_for_prompt: bool = False) -> None:
        """Send text input to terminal.

        Args:
            text: Text to send
            wait_for_prompt: If True, wait for prompt before returning

        Raises:
            RuntimeError: If session is not running
        """
        if not self.child:
            raise RuntimeError("Terminal session not running")

        try:
            # Use sendline() to properly handle string input with pexpect
            # sendline() encodes the string and adds a newline automatically
            # But we want to send without newline first, so use send_bytes + encode
            encoded = text.encode('utf-8')
            self.child.write(encoded)
            logger.debug(f"Sent input: {repr(text)}")

            # Wait a moment for processing
            time.sleep(0.1)

        except pexpect.exceptions.TIMEOUT:
            logger.warning("Timeout sending input")
        except Exception as e:
            logger.error(f"Failed to send input: {e}")
            raise RuntimeError(f"Failed to send input: {e}") from e

    def send_key(self, key: str) -> None:
        """Send special key to terminal.

        Args:
            key: Key name (e.g., "enter", "up", "down")
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
            raise ValueError(f"Unknown key: {key}")

        self.send_input(key_map[key_lower])

    def get_screen_text(self) -> str:
        """Get current terminal output as text.

        Returns:
            Text from terminal buffer
        """
        if not self.child:
            return ""

        try:
            # pexpect's buffer contains bytes from the subprocess output
            # Combine buffer and any accumulated output
            output = b''

            # Get buffer contents (bytes)
            if hasattr(self.child, 'buffer') and self.child.buffer:
                if isinstance(self.child.buffer, bytes):
                    output += self.child.buffer
                else:
                    output += self.child.buffer.encode('utf-8') if isinstance(self.child.buffer, str) else b''

            # Get before contents if available (bytes)
            if hasattr(self.child, 'before') and self.child.before:
                if isinstance(self.child.before, bytes):
                    output += self.child.before
                else:
                    output += self.child.before.encode('utf-8') if isinstance(self.child.before, str) else b''

            # Decode bytes to string
            try:
                text = output.decode('utf-8', errors='replace')
            except (AttributeError, UnicodeDecodeError):
                text = str(output)

            return text
        except Exception as e:
            logger.warning(f"Failed to get screen text: {e}")
            return ""

    def expect_pattern(self, pattern: str, timeout: float | None = None) -> bool:
        """Wait for pattern to appear in output.

        Args:
            pattern: Regex pattern to expect
            timeout: Maximum time to wait (uses default if None)

        Returns:
            True if pattern found, False if timeout
        """
        if not self.child:
            return False

        if timeout is None:
            timeout = self.timeout

        try:
            self.child.expect(pattern, timeout=timeout)
            return True
        except pexpect.TIMEOUT:
            return False
        except pexpect.EOF:
            logger.warning("Got EOF while waiting for pattern")
            return False

    def expect_text(self, text: str, timeout: float | None = None) -> bool:
        """Wait for specific text to appear in output.

        Args:
            text: Exact text to find
            timeout: Maximum time to wait

        Returns:
            True if text found, False if timeout
        """
        # Escape special regex characters
        pattern = "".join(
            f"\\{c}" if c in r".^$*+?{}[]|()\\" else c for c in text
        )
        return self.expect_pattern(pattern, timeout)

    def interact(self) -> None:
        """Allow user to interact with the terminal.

        This is useful for debugging - allows manual input.
        """
        if self.child:
            self.child.interact()

    def close(self) -> None:
        """Close the terminal session."""
        if self.child:
            try:
                self.child.terminate(force=True)
            except Exception:
                pass
            self.child = None

            logger.info(f"PTY session closed: {self.session_name}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
