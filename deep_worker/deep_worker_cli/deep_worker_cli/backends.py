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
import itertools
import logging
import subprocess  # noqa: S404
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, cast

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

if TYPE_CHECKING:
    from asyncio.subprocess import Process

    from deepagents.middleware.parked_job_registry import (
        ParkedJob,
        ParkedJobRegistry,
    )

_TIMEOUT_DESC = (
    "Optional timeout in seconds (default 120). On timeout the command is "
    "NOT killed — it keeps running in the background and you get a job_id. "
    "Use wait_for_job(job_id) to wait longer or kill_job(job_id) to stop it."
)


@dataclass
class _BackgroundJob:
    """A shell command that outlived its execute() timeout and is still
    running in the background.

    The reader task drains the process pipes to completion and stores the
    final ``(stdout, stderr)`` in ``result``. Because the task is shielded
    from the timeout cancellation, the command keeps running after execute()
    returns — the agent decides whether to ``wait_for_job`` or ``kill_job``.

    ``owner`` is the ``CLIShellBackend`` that parked the job. Required so
    ``wait_for_job`` / ``kill_job`` on a *different* backend instance can
    still drop the job from the owner's ``_jobs`` (fifth locus, 2026-07-31 —
    see docs/DESIGN_FIX_SHELL_JOB_CROSS_BACKEND.md).
    """

    job_id: str
    command: str
    proc: "Process"
    reader: "asyncio.Task[tuple[bytes, bytes]]"
    started_at: float
    result: tuple[bytes, bytes] | None = field(default=None)
    owner: object | None = field(default=None, repr=False)

    @property
    def finished(self) -> bool:
        return self.reader.done()


class CLIShellBackend(LocalShellBackend):
    """Local shell backend with per-command timeout override support.

    Extends `LocalShellBackend` to accept an optional `timeout` keyword on
    each `execute()` call. On timeout the command is NOT killed: it is moved
    to a per-backend background-job registry and the agent can later
    ``wait_for_job`` or ``kill_job`` it. The registry lives on the backend
    instance, which is created once per agent and persists across tool calls.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._jobs: dict[str, _BackgroundJob] = {}
        # Alias map: the originating tool call's ``tool_use_id`` → canonical
        # ``job_N``. Background jobs are keyed in ``_jobs`` by ``job_N`` only;
        # this lets the model reach a job by EITHER id. See the mj2_vac
        # incident (2026-06-17): the model passed the execute call's
        # ``toolu_…`` tool_use_id instead of the advertised ``job_1`` and got
        # "no background job" forever. Both id spaces must resolve.
        self._job_aliases: dict[str, str] = {}
        self._job_counter = itertools.count(1)

    @staticmethod
    def _current_tool_use_id() -> str | None:
        """The tool_use_id of the in-flight tool call, if a provider set it.

        The claude-code provider stashes it in a ContextVar (vendored SDK
        ``_mcp_context``) for the duration of each tool handler. Best-effort:
        returns ``None`` when the provider/context is absent (other providers,
        direct backend use, tests) — the job is then reachable only by its
        ``job_N`` id, which is always valid."""
        try:
            # Optional, provider-only vendored SDK; not a deep_worker_cli
            # dependency, so ``ty`` can't resolve it. Guarded at runtime.
            from claude_agent_sdk._mcp_context import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
                get_current_claude_tool_use_id,
            )
        except Exception:  # noqa: BLE001 — provider not installed / no context
            return None
        try:
            tid = get_current_claude_tool_use_id()
        except Exception:  # noqa: BLE001
            return None
        return tid or None

    def _resolve_job(self, job_id: str) -> _BackgroundJob | None:
        """Look up a parked SHELL job by EITHER its canonical ``job_N`` id or
        the ``tool_use_id`` alias of the ``execute`` call that spawned it.

        Resolution order (fifth locus — cross-backend visibility):

        1. This instance's ``_jobs`` / ``_job_aliases`` (fast path).
        2. The process-wide ``ParkedJobRegistry`` for ``kind="shell"`` —
           handle is the ``_BackgroundJob`` itself, so a *different*
           ``CLIShellBackend`` instance can still wait/kill the process.

        Tool jobs parked by middleware / proxy-MCP live in the same registry
        under ``kind="tool"`` and are handled by ``_resolve_parked``.
        """
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        canonical = self._job_aliases.get(job_id)
        if canonical is not None:
            found = self._jobs.get(canonical)
            if found is not None:
                return found
        return self._resolve_shell_from_registry(job_id)

    def _resolve_shell_from_registry(self, job_id: str) -> _BackgroundJob | None:
        """Shell job from the shared registry (any owning backend instance)."""
        registry = self._shared_registry()
        if registry is None:
            return None
        parked = registry.resolve(job_id)
        if parked is None or parked.kind != "shell":
            return None
        handle = parked.handle
        if isinstance(handle, _BackgroundJob):
            return handle
        # Backward-compat: older register() put the backend as handle.
        if isinstance(handle, CLIShellBackend):
            return handle._jobs.get(parked.job_id)
        return None

    @staticmethod
    def _resolve_parked(job_id: str) -> "ParkedJob | None":
        """Look up a job parked by the tool-call middleware (``kind="tool"``).

        Both parking layers share ONE id namespace via ``ParkedJobRegistry``,
        so ``wait_for_job``/``kill_job`` must be able to see either kind —
        otherwise the model is handed an id the tools cannot honour, which is
        exactly the observed defect (2026-07-31). Best-effort: returns ``None``
        when deepagents is unavailable (e.g. isolated backend tests).

        Only ``ImportError`` is tolerated: any OTHER exception raised while
        importing means the registry module itself is broken, and silently
        disabling the bridge would resurrect the very "no background job"
        symptom this fix removes.
        """
        try:
            from deepagents.middleware.parked_job_registry import (  # noqa: PLC0415
                get_parked_job_registry,
            )
        except ImportError:  # deepagents genuinely absent (isolated tests)
            return None
        job = get_parked_job_registry().resolve(job_id)
        if job is None or job.kind != "tool":
            return None
        return job

    @staticmethod
    def _shared_registry() -> "ParkedJobRegistry | None":
        """The process-wide ``ParkedJobRegistry``, or ``None`` when deepagents
        is not importable (isolated backend tests).

        Narrow on purpose — see ``_resolve_parked``: a broken registry module
        must surface, not silently degrade.
        """
        try:
            from deepagents.middleware.parked_job_registry import (  # noqa: PLC0415
                get_parked_job_registry,
            )
        except ImportError:  # deepagents genuinely absent (isolated tests)
            return None
        return get_parked_job_registry()

    def _allocate_job_id(self, tool_use_id: str | None) -> str:
        """Mint the canonical ``job_N`` for a shell job.

        Ids come from the SHARED counter so that a job parked here and a job
        parked by ``BoundedToolCallMiddleware`` can never collide. If the
        middleware already parked work for this very tool call, reuse its id
        instead of minting a second one for the same command.
        """
        registry = self._shared_registry()
        if registry is None:
            return f"job_{next(self._job_counter)}"
        if tool_use_id:
            existing = registry.resolve_by_alias(tool_use_id)
            if existing is not None:
                return str(existing.job_id)
        return registry.allocate()

    def _register_shared(
        self,
        job_id: str,
        tool_use_id: str | None,
        *,
        job: _BackgroundJob,
    ) -> None:
        """Publish a shell job into the shared registry so every id the model
        may be given resolves in ONE place — including on a *different*
        ``CLIShellBackend`` instance (handle is the ``_BackgroundJob``, not
        ``self``)."""
        registry = self._shared_registry()
        if registry is None:
            return
        registry.register(
            job_id,
            kind="shell",
            handle=job,
            alias=tool_use_id or None,
            tool_name="execute",
        )

    def _drop_job(self, job: _BackgroundJob) -> None:
        """Remove a finished/killed job from the owning backend's ``_jobs``,
        purge aliases, and drop the shared-registry entry.

        Must work when called from a backend that is NOT the owner (cross-
        instance wait/kill).
        """
        owner = job.owner if isinstance(job.owner, CLIShellBackend) else self
        owner._jobs.pop(job.job_id, None)
        stale = [
            a for a, canon in owner._job_aliases.items() if canon == job.job_id
        ]
        for alias in stale:
            owner._job_aliases.pop(alias, None)
        # Also clear this instance if it held a stale alias copy.
        if owner is not self:
            self._jobs.pop(job.job_id, None)
            stale_self = [
                a for a, canon in self._job_aliases.items() if canon == job.job_id
            ]
            for alias in stale_self:
                self._job_aliases.pop(alias, None)
        registry = self._shared_registry()
        if registry is not None:
            registry.drop(job.job_id)

    def _format_partial(self, stdout: bytes, stderr: bytes) -> str:
        """Format already-captured output for an in-flight/finished job."""
        stdout_s = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_s = stderr.decode("utf-8", errors="replace") if stderr else ""
        parts: list[str] = []
        if stdout_s:
            parts.append(stdout_s)
        if stderr_s:
            parts.extend(f"[stderr] {ln}" for ln in stderr_s.strip().split("\n"))
        return "\n".join(parts) if parts else "<no output yet>"

    async def execute(  # ty: ignore[invalid-method-override]
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        # LSP-breaking override: base `LocalShellBackend.execute` is a
        # synchronous method without a `timeout` kwarg. This class
        # intentionally replaces it with an async version so that shell
        # commands don't block the rpyc event loop while the agent is
        # serving RPC calls like `get_status`. Callers must always await
        # this method (see middleware/shell_tool). See
        # docs/DESIGN_ASYNC_NOTIFICATIONS.md for the refactor rationale.

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

        # Retry loop for transient OSError (e.g. ConnectionRefusedError)
        # during subprocess creation. Retry once after 0.5s sleep.
        # Other exception types (ValueError, TimeoutError, etc.) are NOT
        # transient — they fall through to the outer handler directly.
        # See DESIGN_FIX_EXECUTE_TOOL_CONNECTION_REFUSED_RETRY.md (#1192).
        try:
            for _attempt in (1, 2):
                try:
                    proc = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=self._env,
                        cwd=str(self.cwd),
                        start_new_session=True,
                    )
                except OSError as exc:
                    if _attempt == 1:
                        logger.warning(
                            "execute() subprocess creation failed (attempt 1/2), "
                            "retrying after 0.5s: %s: %s",
                            type(exc).__name__,
                            exc,
                        )
                        await asyncio.sleep(0.5)
                        continue  # retry once
                    raise  # second failure → outer except Exception
                break  # success → proceed to pipe reading

            # Drain the pipes in a dedicated task so that, if the initial
            # wait times out, the command can keep running in the background
            # WITHOUT us cancelling the read (which would lose its output and
            # require killing the process). ``shield`` lets the timeout fire
            # while the reader task survives.
            reader: asyncio.Task[tuple[bytes, bytes]] = asyncio.ensure_future(
                proc.communicate()
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    asyncio.shield(reader),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # NEW BEHAVIOR (2026-05-21): do NOT kill on timeout. Register
                # the still-running command as a background job and tell the
                # agent its job_id + the actions it can take. The agent then
                # decides: wait_for_job(job_id) to wait longer, or
                # kill_job(job_id) to stop it.
                import time as _time

                _tool_use_id = self._current_tool_use_id()
                job_id = self._allocate_job_id(_tool_use_id)
                bg_job = _BackgroundJob(
                    job_id=job_id,
                    command=command,
                    proc=proc,
                    reader=reader,
                    started_at=_time.monotonic(),
                    owner=self,
                )
                self._jobs[job_id] = bg_job
                # Also reach this job by the originating tool_use_id, so the
                # model can't get stuck if it passes ``toolu_…`` instead of
                # the advertised ``job_N`` (mj2_vac, 2026-06-17).
                if _tool_use_id and _tool_use_id != job_id:
                    self._job_aliases[_tool_use_id] = job_id
                self._register_shared(job_id, _tool_use_id, job=bg_job)
                return ExecuteResponse(
                    output=(
                        f"Command still running after {effective_timeout}s timeout — "
                        f"it was NOT killed. Background job id: {job_id}\n"
                        f"  • wait_for_job(job_id=\"{job_id}\", timeout=<seconds>) "
                        f"to wait for it to finish\n"
                        f"  • kill_job(job_id=\"{job_id}\") to stop it\n"
                        f"Pass job_id exactly as shown above (\"{job_id}\") — it is "
                        f"NOT a tool_use_id.\n"
                        f"Command: {command}"
                    ),
                    exit_code=None,  # Still running — no exit code yet.
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

    def _finalize_job(self, job: _BackgroundJob) -> ExecuteResponse:
        """Build the final ExecuteResponse for a finished background job and
        drop it from the registry."""
        stdout_bytes, stderr_bytes = job.reader.result()
        self._drop_job(job)
        output = self._format_partial(stdout_bytes, stderr_bytes)
        if output == "<no output yet>":
            output = "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True
        return ExecuteResponse(
            output=output,
            exit_code=job.proc.returncode if job.proc.returncode is not None else 0,
            truncated=truncated,
        )

    @staticmethod
    def _parked_tool_task(parked: "ParkedJob") -> object | None:
        """The awaitable/cancellable handle inside a parked tool job.

        The async path parks a ``_ParkedToolJob`` (whose ``.task`` is an
        ``asyncio.Task``); the sync path parks a ``concurrent.futures.Future``
        directly. Both expose ``cancel``/``done``.
        """
        handle = parked.handle
        return getattr(handle, "task", handle)

    async def _wait_parked_tool(
        self, parked: "ParkedJob", timeout: int | None
    ) -> ExecuteResponse:
        """Wait for a job parked by the tool-call middleware.

        Returns the tool's own result once it finishes (status + result, per
        design review) so the agent is not left blind after a long tool
        outlived the outer bound. While it is still running the answer restates
        the canonical id — never an error.
        """
        job_id = parked.job_id
        name = parked.tool_name or "tool"
        task = self._parked_tool_task(parked)
        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        if isinstance(task, asyncio.Task):
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                return ExecuteResponse(
                    output=(
                        f"Job '{job_id}' (tool '{name}') is STILL running "
                        f"after waiting {effective_timeout}s — not killed.\n"
                        f"  • wait_for_job(job_id=\"{job_id}\", "
                        f"timeout=<seconds>) to keep waiting\n"
                        f"  • kill_job(job_id=\"{job_id}\") to stop it"
                    ),
                    exit_code=None,
                    truncated=False,
                )
            except asyncio.CancelledError:
                return ExecuteResponse(
                    output=f"Job '{job_id}' (tool '{name}') was cancelled.",
                    exit_code=1,
                    truncated=False,
                )
            except Exception as exc:  # noqa: BLE001 — surface the tool's error
                return ExecuteResponse(
                    output=(
                        f"Job '{job_id}' (tool '{name}') failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    exit_code=1,
                    truncated=False,
                )
            return ExecuteResponse(
                output=self._render_tool_result(job_id, name, result),
                exit_code=self._tool_exit_code(result),
                truncated=False,
            )

        # Sync path: a thread-backed future cannot be awaited without blocking
        # the loop, so only report its state. It is never cancellable.
        done = bool(getattr(task, "done", lambda: False)())
        if done:
            finished = getattr(task, "result", lambda: None)()
            return ExecuteResponse(
                output=self._render_tool_result(job_id, name, finished),
                exit_code=self._tool_exit_code(finished),
                truncated=False,
            )
        return ExecuteResponse(
            output=(
                f"Job '{job_id}' (tool '{name}') is STILL running on a worker "
                f"thread — it cannot be cancelled. Wait again with "
                f"wait_for_job(job_id=\"{job_id}\")."
            ),
            exit_code=None,
            truncated=False,
        )

    @staticmethod
    def _tool_exit_code(result: object) -> int:
        """Map a finished tool's own status onto an execute-style exit code.

        A tool that completed but reported failure must not look like a clean
        success just because the JOB finished. The content is returned either
        way — only the code distinguishes them.
        """
        return 1 if getattr(result, "status", None) == "error" else 0

    @staticmethod
    def _render_tool_result(job_id: str, name: str, result: object) -> str:
        """Map a finished tool's result into execute-style output."""
        content = getattr(result, "content", result)
        text = content if isinstance(content, str) else str(content)
        return f"Job '{job_id}' (tool '{name}') finished.\n{text}"

    async def _kill_parked_tool(self, parked: "ParkedJob") -> ExecuteResponse:
        """Cancel a job parked by the tool-call middleware.

        ``task.cancel()`` is the correct API. Per design review we then wait
        BRIEFLY and bounded for the task to settle — never unbounded, because
        work that swallows ``CancelledError`` would hang us forever.
        """
        job_id = parked.job_id
        name = parked.tool_name or "tool"
        task = self._parked_tool_task(parked)

        if not isinstance(task, asyncio.Task):
            return ExecuteResponse(
                output=(
                    f"Job '{job_id}' (tool '{name}') runs on a worker thread "
                    f"and cannot be killed; it will finish on its own."
                ),
                exit_code=1,
                truncated=False,
            )

        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:  # noqa: BLE001 — the tool's own failure is fine here
            pass

        registry = self._shared_registry()
        if registry is not None:
            registry.drop(job_id)

        if task.done():
            return ExecuteResponse(
                output=f"Job '{job_id}' (tool '{name}') killed.",
                exit_code=0,
                truncated=False,
            )
        return ExecuteResponse(
            output=(
                f"Job '{job_id}' (tool '{name}'): cancellation requested, but "
                f"the task has not stopped yet."
            ),
            exit_code=1,
            truncated=False,
        )

    async def wait_for_job(
        self, job_id: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        """Wait up to ``timeout`` seconds for a background job to finish.

        Returns the job's full output + exit code when it finishes. If it is
        still running after ``timeout``, returns a message restating the same
        wait/kill options — the job is left running.

        ``job_id`` may be either the canonical ``job_N`` id or the
        ``tool_use_id`` of the ``execute`` call that spawned the job.
        """
        job = self._resolve_job(job_id)
        if job is None:
            parked = self._resolve_parked(job_id)
            if parked is not None:
                return await self._wait_parked_tool(parked, timeout)
            return ExecuteResponse(
                output=f"Error: no background job with id '{job_id}'.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            await asyncio.wait_for(asyncio.shield(job.reader), timeout=effective_timeout)
        except asyncio.TimeoutError:
            # Restate the CANONICAL job_N id (job.job_id), not whatever id the
            # caller passed — this nudges the model back onto the stable id.
            return ExecuteResponse(
                output=(
                    f"Job '{job.job_id}' is STILL running after waiting "
                    f"{effective_timeout}s — not killed.\n"
                    f"  • wait_for_job(job_id=\"{job.job_id}\", timeout=<seconds>) "
                    f"to keep waiting\n"
                    f"  • kill_job(job_id=\"{job.job_id}\") to stop it\n"
                    f"Command: {job.command}"
                ),
                exit_code=None,
                truncated=False,
            )

        return self._finalize_job(job)

    async def kill_job(self, job_id: str) -> ExecuteResponse:
        """Terminate a background job (SIGTERM, then SIGKILL) and return any
        output captured before it died.

        ``job_id`` may be either the canonical ``job_N`` id or the
        ``tool_use_id`` of the ``execute`` call that spawned the job."""
        job = self._resolve_job(job_id)
        if job is None:
            parked = self._resolve_parked(job_id)
            if parked is not None:
                return await self._kill_parked_tool(parked)
            return ExecuteResponse(
                output=f"Error: no background job with id '{job_id}'.",
                exit_code=1,
                truncated=False,
            )

        # Signal the whole process group (the command runs under
        # start_new_session=True, so its pgid == its pid). This reaches the
        # shell AND its children (e.g. the `sleep` in `sleep 30 && echo`),
        # which a bare proc.terminate() would leak. Fall back to the plain
        # proc signals if the group lookup fails.
        import os
        import signal

        def _signal_group(sig: int) -> None:
            try:
                os.killpg(os.getpgid(job.proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                try:
                    job.proc.send_signal(sig)
                except ProcessLookupError:
                    pass

        try:
            _signal_group(signal.SIGTERM)
            try:
                await asyncio.wait_for(job.proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                _signal_group(signal.SIGKILL)
                await job.proc.wait()
        except ProcessLookupError:
            pass  # Already dead.

        # The reader task resolves once the pipes close after the kill.
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(job.reader, timeout=5.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            stdout_bytes, stderr_bytes = b"", b""
        self._drop_job(job)

        captured = self._format_partial(stdout_bytes, stderr_bytes)
        return ExecuteResponse(
            output=(
                f"Job '{job.job_id}' killed.\n"
                f"Output captured before kill:\n{captured}"
            ),
            exit_code=job.proc.returncode,
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


# ═══════════════════════════════════════════════════════════════════════════
# 🛑 PROTECTED INVARIANT — DO NOT BREAK (a related internal incident)
# ═══════════════════════════════════════════════════════════════════════════
# The agent's final backend may be wrapped in transparent proxies
# (PrivateSummariesBackend → CompositeBackend → CLIShellBackend). Resolving the
# sandbox here is what decides whether the `execute` shell tool works AND
# whether it is even exposed to the model (see _supports_execution in
# deepagents/middleware/filesystem.py — the SAME invariant lives there).
#
# RULES for anyone (human or agent) editing backend wrapping or this resolver:
#   1. If you add a NEW backend wrapper, it MUST be unwrappable here — either
#      expose its inner backend via a `_backend` attribute (the convention
#      _unwrap_proxy_backend follows) OR update _unwrap_proxy_backend.
#   2. NEVER replace these `_unwrap_proxy_backend(...)` calls with a bare
#      `isinstance(backend, CompositeBackend)` — that is exactly the regression
#      (an earlier change) that silently stripped `execute` from an agent.
#   3. Mirror ANY change here in deepagents `_supports_execution`.
# Breakage symptom: agent reports "I have no execute tool" / "backend does not
# support command execution". Covered by tests/unit_tests/test_sandbox_backend_unwrap.py.
# ═══════════════════════════════════════════════════════════════════════════
def _unwrap_proxy_backend(backend: BackendProtocol) -> BackendProtocol:
    """Peel off transparent proxy wrappers to reach the wrapped backend.

    Some backends are proxies that delegate to an inner backend held in
    ``_backend`` (e.g. ``PrivateSummariesBackend``, summaries.py). Such a
    proxy subclasses ``BackendProtocol`` — NOT ``CompositeBackend`` — so the
    composite/sandbox ``isinstance`` checks below do not see through it. We
    follow the ``_backend`` chain so a proxy(composite(sandbox)) still resolves
    to the sandbox. See a related internal incident analysis (not included here).

    Iterative + visited-set so a self-referential or cyclic wrapper can never
    spin. We stop at the first non-proxy (composite/sandbox/plain) backend.
    """
    seen: set[int] = set()
    current = backend
    # A "proxy" here = has a private ``_backend`` AND is not itself a structural
    # backend type we know how to introspect (composite/sandbox). This avoids
    # accidentally unwrapping a real backend that merely happens to expose a
    # ``_backend`` attribute.
    while (
        not isinstance(current, (CompositeBackend, SandboxBackendProtocol))
        and hasattr(current, "_backend")
        and id(current) not in seen
    ):
        seen.add(id(current))
        inner = getattr(current, "_backend", None)
        if inner is None or inner is current:
            break
        current = inner
    return current


def _get_sandbox_backend(
    backend: BackendProtocol,
) -> SandboxBackendProtocol | None:
    """Unwrap proxy + `CompositeBackend` layers to find the sandbox backend.

    `CompositeBackend.execute()` delegates to `self.default.execute()` but
    doesn't forward extra kwargs like `timeout`. This helper reaches through
    to the actual sandbox backend so we can call `execute(timeout=...)` on it.

    It first peels transparent proxy wrappers (``_backend`` chain, e.g.
    ``PrivateSummariesBackend``) — without this, an agent whose final backend
    is ``PrivateSummariesBackend(CompositeBackend(CLIShellBackend))`` resolves
    to ``None`` and ``execute`` wrongly reports "backend does not support
    command execution" (a related internal incident).

    The returned backend must support a `timeout` kwarg on `execute()`
    (e.g., `CLIShellBackend`). The base `SandboxBackendProtocol` does not
    define this parameter -- the `timeout` kwarg is specific to the CLI
    subclass.

    Args:
        backend: The resolved backend, possibly proxy- and/or composite-wrapped.

    Returns:
        The sandbox backend if found, `None` otherwise.
    """
    backend = _unwrap_proxy_backend(backend)
    if isinstance(backend, CompositeBackend):
        # The composite's default may itself be proxy-wrapped.
        default = _unwrap_proxy_backend(backend.default)
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
    """Filesystem middleware with per-command timeout on the execute tool
    plus background-job control.

    - Overrides `_create_execute_tool` to expose a `timeout` parameter so the
      LLM can pass it for long-running commands.
    - Adds `wait_for_job` and `kill_job` tools so that when `execute` times
      out (and the command keeps running in the background — see
      `CLIShellBackend`), the agent can wait for it or stop it by job_id.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        # Expose the background-job tools alongside the filesystem tools.
        # ``self.tools`` is built by the base ``__init__`` (it already
        # includes our overridden ``execute``); appending here means the
        # new tools flow through the same registration path that surfaces
        # ``execute`` to the model (incl. the claude-code proxy MCP).
        self.tools = [*self.tools, self._create_wait_for_job_tool(), self._create_kill_job_tool()]

    def _resolve_sandbox(
        self, runtime: ToolRuntime[None, FilesystemState]
    ) -> "CLIShellBackend | None":
        """Resolve the CLIShellBackend for job tools, or None if unavailable."""
        proto = _get_sandbox_backend(self._get_backend(runtime))
        if proto is None or not isinstance(proto, CLIShellBackend):
            return None
        return proto

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

    def _create_wait_for_job_tool(self) -> BaseTool:
        """Tool to wait for a background job (created when execute timed out)."""

        _NOT_AVAILABLE = (
            "Error: background jobs are not available — this agent's backend "
            "does not support command execution."
        )

        def _run_sync(job_id: str, timeout: int | None, runtime: ToolRuntime[None, FilesystemState]) -> str:
            sandbox = self._resolve_sandbox(runtime)
            if sandbox is None:
                return _NOT_AVAILABLE
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    sandbox.wait_for_job(job_id, timeout=timeout), loop
                )
                result = future.result()
            except RuntimeError:
                result = asyncio.run(sandbox.wait_for_job(job_id, timeout=timeout))
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"
            return _format_execute_result(result)

        def sync_wait(
            job_id: Annotated[
                str,
                "The background job id from execute's timeout message, e.g. "
                "'job_1' or 'job_2'. Copy it VERBATIM from that message. This "
                "is NOT a tool_use_id (a 'toolu_…' value) — do not pass one. "
                "(The originating execute call's tool_use_id is also accepted "
                "as a fallback, but the 'job_N' id is the correct value.)",
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[
                int | None,
                "Seconds to wait (default 120). If still running after this, "
                "you get the job_id back and can wait again or kill it.",
            ] = None,
        ) -> str:
            return _run_sync(job_id, timeout, runtime)

        async def async_wait(
            job_id: Annotated[
                str,
                "The background job id from execute's timeout message, e.g. "
                "'job_1' or 'job_2'. Copy it VERBATIM from that message. This "
                "is NOT a tool_use_id (a 'toolu_…' value) — do not pass one. "
                "(The originating execute call's tool_use_id is also accepted "
                "as a fallback, but the 'job_N' id is the correct value.)",
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[  # noqa: ASYNC109
                int | None,
                "Seconds to wait (default 120). If still running after this, "
                "you get the job_id back and can wait again or kill it.",
            ] = None,
        ) -> str:
            sandbox = self._resolve_sandbox(runtime)
            if sandbox is None:
                return _NOT_AVAILABLE
            try:
                result = await sandbox.wait_for_job(job_id, timeout=timeout)
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"
            return _format_execute_result(result)

        return StructuredTool.from_function(
            name="wait_for_job",
            description=(
                "Wait for a background shell job (created when `execute` timed "
                "out) to finish, up to `timeout` seconds. Returns its output "
                "and exit code if it finishes, or the job_id again (still "
                "running) so you can wait more or kill it."
            ),
            func=sync_wait,
            coroutine=async_wait,
        )

    def _create_kill_job_tool(self) -> BaseTool:
        """Tool to kill a background job (created when execute timed out)."""

        _NOT_AVAILABLE = (
            "Error: background jobs are not available — this agent's backend "
            "does not support command execution."
        )

        def sync_kill(
            job_id: Annotated[
                str,
                "The background job id from execute's timeout message, e.g. "
                "'job_1' or 'job_2'. Copy it VERBATIM from that message. This "
                "is NOT a tool_use_id (a 'toolu_…' value) — do not pass one. "
                "(The originating execute call's tool_use_id is also accepted "
                "as a fallback, but the 'job_N' id is the correct value.)",
            ],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            sandbox = self._resolve_sandbox(runtime)
            if sandbox is None:
                return _NOT_AVAILABLE
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(sandbox.kill_job(job_id), loop)
                result = future.result()
            except RuntimeError:
                result = asyncio.run(sandbox.kill_job(job_id))
            return _format_execute_result(result)

        async def async_kill(
            job_id: Annotated[
                str,
                "The background job id from execute's timeout message, e.g. "
                "'job_1' or 'job_2'. Copy it VERBATIM from that message. This "
                "is NOT a tool_use_id (a 'toolu_…' value) — do not pass one. "
                "(The originating execute call's tool_use_id is also accepted "
                "as a fallback, but the 'job_N' id is the correct value.)",
            ],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            sandbox = self._resolve_sandbox(runtime)
            if sandbox is None:
                return _NOT_AVAILABLE
            result = await sandbox.kill_job(job_id)
            return _format_execute_result(result)

        return StructuredTool.from_function(
            name="kill_job",
            description=(
                "Terminate a background shell job (created when `execute` timed "
                "out) by its job_id. Returns any output captured before it was "
                "stopped."
            ),
            func=sync_kill,
            coroutine=async_kill,
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

    # Monkey-patch: ty sees this as narrowing the module-level class
    # reference from `FilesystemMiddleware` to `CLIFilesystemMiddleware`
    # (a subclass). That's exactly the intent — every call to
    # `FilesystemMiddleware(...)` in the SDK will now construct our
    # subclass. ty flags the assignment because class slots are invariant;
    # we suppress only this one (the whole function is documented as an
    # intentional runtime-patch).
    fs_module.FilesystemMiddleware = CLIFilesystemMiddleware  # ty: ignore[invalid-assignment]
    graph_module.FilesystemMiddleware = CLIFilesystemMiddleware  # ty: ignore[invalid-assignment]
