"""CLI-specific backend and middleware overrides for per-command timeout support.

Subclasses `LocalShellBackend` and `FilesystemMiddleware` to add:
1. Per-command `timeout` keyword argument to `execute()`, so the LLM can override
   the default timeout for long-running commands
2. Async subprocess execution to prevent event loop blocking during command execution

CRITICAL BUG FIX (commit 2026-04-01):
=====================================
The original implementation used synchronous `subprocess.run()` which BLOCKED
the entire asyncio event loop during command execution. This caused:
- Agent freezing during long-running commands (sleep, npm install, pytest)
- RPC calls like get_status() timing out while commands execute
- Multiple client connections hanging simultaneously
- RPyC garbage collection failures with "result expired" errors

The fix replaces `subprocess.run()` with `asyncio.create_subprocess_shell()`,
making execute() truly async. This allows the event loop to remain responsive
and process concurrent RPC calls while commands execute in the background.

RELATED FIX (commit 2026-04-02):
================================
Enhanced `dwcli --stop` command to FORCE KILL frozen agents:
1. Try graceful shutdown with SIGTERM (wait up to 5 seconds)
2. If agent doesn't respond, FORCE KILL with SIGKILL (no mercy!)
3. Verify process is dead (not just zombie) before returning success
4. Clean up registry entries for dead processes
5. Report what happened (graceful vs forced kill)

This ensures `dwcli --stop` ALWAYS stops agents, even if they're frozen
in infinite loops or unresponsive to SIGTERM.

Uses monkey-patching via `patch_filesystem_middleware()` to replace the SDK's
`FilesystemMiddleware` class reference with `CLIFilesystemMiddleware`, so the
SDK constructs the CLI subclass transparently.

When the SDK adds this natively, these subclasses can be removed.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess  # noqa: S404
from typing import Annotated, cast

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    SandboxBackendProtocol,
)
from deepagents.middleware.filesystem import (
    EXECUTE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
    FilesystemState,
)
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.tools import BaseTool, StructuredTool

from deep_worker_cli.constants import DEFAULT_EXECUTE_TIMEOUT

logger = logging.getLogger(__name__)

_TIMEOUT_DESC = (
    "Optional timeout in seconds. Overrides the default. Use for long-running commands."
)


class CLIShellBackend(LocalShellBackend):
    """Local shell backend with per-command timeout override support.

    Extends `LocalShellBackend` to accept an optional `timeout` keyword on
    each `execute()` call.
    """

    async def execute(  # type: ignore[override]
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Execute a shell command asynchronously with optional per-command timeout.

        CRITICAL: This method is async to prevent blocking the event loop during
        long-running commands (e.g., sleep 15, npm install, pytest). When commands
        execute, the event loop remains responsive for concurrent RPC calls like
        get_status().

        Overrides `LocalShellBackend.execute()` to:
        1. Use async subprocess execution (asyncio.create_subprocess_shell)
        2. Accept a per-command `timeout` that overrides the default

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                Overrides the default timeout set at init.

                If `None`, falls back to the instance-level timeout
                configured at init time (defaults to 120s if not overridden).

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

        Raises:
            ValueError: If the effective timeout (per-command or instance
                default) is not positive.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            # Create async subprocess - DOES NOT block event loop!
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                cwd=str(self.cwd),
            )

            # Wait for process with timeout - event loop remains responsive!
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                # Use terminate() first (SIGTERM), then kill() if needed (SIGKILL)
                try:
                    proc.terminate()  # Send SIGTERM
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        # Process didn't terminate gracefully, force kill
                        proc.kill()  # Send SIGKILL
                        await proc.wait()
                except ProcessLookupError:
                    pass  # Process already died

                return ExecuteResponse(
                    output=(
                        f"Error: Command timed out after {effective_timeout} seconds."
                        " For long-running commands, re-run using the timeout parameter."
                    ),
                    exit_code=124,  # Standard timeout exit code
                    truncated=False,
                )

            # Decode output
            stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

            # Combine stdout and stderr
            # Prefix each stderr line with [stderr] for clear attribution.
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                stderr_lines = stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            return ExecuteResponse(
                output=output,
                exit_code=proc.returncode if proc.returncode is not None else 0,
                truncated=truncated,
            )

        except Exception as e:  # noqa: BLE001
            # Broad exception catch is intentional: we want to catch all
            # execution errors and return a consistent ExecuteResponse rather
            # than propagating exceptions that would crash the agent loop.
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


def _format_execute_result(result: ExecuteResponse) -> str:
    """Format an `ExecuteResponse` for LLM consumption.

    Args:
        result: The execution response to format.

    Returns:
        Formatted string with output, status, and truncation info.
    """
    parts = [result.output]
    if result.exit_code is not None:
        status = "succeeded" if result.exit_code == 0 else "failed"
        parts.append(f"\n[Command {status} with exit code {result.exit_code}]")
    if result.truncated:
        parts.append("\n[Output was truncated due to size limits]")
    return "".join(parts)


def _get_sandbox_backend(
    backend: BackendProtocol,
) -> SandboxBackendProtocol | None:
    """Unwrap a `CompositeBackend` to find the sandbox backend.

    `CompositeBackend.execute()` delegates to `self.default.execute()` but
    doesn't forward extra kwargs like `timeout`. This helper reaches through
    to the actual sandbox backend so we can call `execute(timeout=...)` on it.

    The returned backend must support a `timeout` kwarg on `execute()`
    (e.g., `CLIShellBackend`). The base `SandboxBackendProtocol` does not
    define this parameter -- the `timeout` kwarg is specific to the CLI
    subclass.

    Args:
        backend: The resolved backend, possibly a `CompositeBackend`.

    Returns:
        The sandbox backend if found, `None` otherwise.
    """
    if isinstance(backend, CompositeBackend):
        default = backend.default
        if isinstance(default, SandboxBackendProtocol):
            return default
        logger.warning(
            "CompositeBackend.default is %s, not a SandboxBackendProtocol. "
            "The execute tool timeout parameter will be unavailable.",
            type(default).__name__,
        )
        return None
    if isinstance(backend, SandboxBackendProtocol):
        return backend
    return None


class CLIFilesystemMiddleware(FilesystemMiddleware):
    """Filesystem middleware with per-command timeout on the execute tool.

    Overrides `_create_execute_tool` to expose a `timeout` parameter in the
    tool schema so the LLM can pass it for long-running commands.
    """

    def _create_execute_tool(self) -> BaseTool:
        """Create the execute tool with an additional `timeout` parameter.

        Returns:
            A `StructuredTool` for shell command execution with timeout support.
        """
        tool_description = (
            self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION
        )

        def sync_execute(
            command: Annotated[
                str, "Shell command to execute in the sandbox environment."
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[int | None, _TIMEOUT_DESC] = None,
        ) -> str:
            """Synchronous wrapper for execute tool.

            CRITICAL: execute() is now async, so we need to run it in event loop.
            This is needed for backwards compatibility with sync tool calls.

            Args:
                command: Shell command to execute.
                runtime: Tool runtime context for backend resolution.
                timeout: Optional per-command timeout in seconds.

            Returns:
                Formatted command output for LLM consumption.
            """
            if timeout is not None and timeout <= 0:
                return f"Error: timeout must be a positive integer, got {timeout}."

            resolved_backend = self._get_backend(runtime)
            proto = _get_sandbox_backend(resolved_backend)

            if proto is None:
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution."
                )

            sandbox = cast("CLIShellBackend", proto)
            try:
                # CRITICAL: execute() is now async!
                # Try to get running event loop, otherwise create new one
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop - use asyncio.create_task()
                    # But this is tricky - sync function can't await
                    # Use run_coroutine_threadsafe instead
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        sandbox.execute(command, timeout=timeout),
                        loop
                    )
                    result = future.result()
                except RuntimeError:
                    # No event loop running - create one
                    result = asyncio.run(sandbox.execute(command, timeout=timeout))
            except NotImplementedError as e:
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            return _format_execute_result(result)

        async def async_execute(
            command: Annotated[
                str, "Shell command to execute in the sandbox environment."
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[  # noqa: ASYNC109
                int | None, _TIMEOUT_DESC
            ] = None,
        ) -> str:
            """Asynchronous wrapper for execute tool.

            CRITICAL: Now calls execute() directly since it's async!
            No need for asyncio.to_thread() wrapper anymore.

            Args:
                command: Shell command to execute.
                runtime: Tool runtime context for backend resolution.
                timeout: Optional per-command timeout in seconds.

            Returns:
                Formatted command output for LLM consumption.
            """
            if timeout is not None and timeout <= 0:
                return f"Error: timeout must be a positive integer, got {timeout}."

            resolved_backend = self._get_backend(runtime)
            proto = _get_sandbox_backend(resolved_backend)

            if proto is None:
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution."
                )

            sandbox = cast("CLIShellBackend", proto)
            try:
                # CRITICAL: execute() is now async, call it directly with await!
                # This prevents event loop blocking during long-running commands.
                result = await sandbox.execute(command, timeout=timeout)
            except NotImplementedError as e:
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            return _format_execute_result(result)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
        )


def patch_filesystem_middleware() -> None:
    """Monkey-patch the SDK to use `CLIFilesystemMiddleware`.

    Must be called before `create_deep_agent` is invoked so the SDK's internal
    `FilesystemMiddleware(backend=...)` calls construct our subclass instead.

    Patches two module-level references:

    - `deepagents.middleware.filesystem.FilesystemMiddleware`
    - `deepagents.graph.FilesystemMiddleware`

    Both must be patched because `graph.py` imports the class at the top level
    and uses it directly when constructing middleware stacks. If the SDK adds
    additional import sites, this patch must be updated accordingly. Validate
    when upgrading the SDK version.
    """
    import deepagents.graph as graph_module
    import deepagents.middleware.filesystem as fs_module

    fs_module.FilesystemMiddleware = CLIFilesystemMiddleware  # type: ignore[misc]
    graph_module.FilesystemMiddleware = CLIFilesystemMiddleware  # type: ignore[misc]
