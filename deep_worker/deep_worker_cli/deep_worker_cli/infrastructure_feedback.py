"""Universal local reporting tool for agent-infrastructure feedback."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from deep_worker_cli import storage_paths

_DIR_NAME = "infrastructure_feedback"
_FILE_NAME = "reports.jsonl"
_FILE_MODE = 0o600
_UNSAFE_DIRECTORY = "unsafe directory component"
_DIRECTORY_CHANGED = "directory component changed during open"
_UNSAFE_DIRECTORY_OWNER = "unsafe directory ownership or type"
_UNSAFE_REPORT_TARGET = "unsafe report target"
_REPORT_TARGET_CHANGED = "report target changed during open"
_UNSAFE_REPORT_FILE = "unsafe report file"
_SHORT_REPORT_WRITE = "short report write"
_REDACTIONS = (
    re.compile(r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)\S+"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|\S+)"
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)


class InfrastructureFeedbackArgs(BaseModel):
    """Validated arguments accepted by the feedback tool."""

    kind: Literal["problem", "improvement"] = Field(
        ...,
        description=(
            "Use 'problem' for faulty or suboptimal infrastructure and "
            "'improvement' for a requested enhancement."
        ),
    )
    component: str = Field(..., min_length=1, max_length=120)
    summary: str = Field(..., min_length=1, max_length=300)
    details: str = Field(..., min_length=1, max_length=4000)
    expected_behavior: str | None = Field(default=None, max_length=2000)


def _redact(value: str | None, max_length: int) -> str | None:
    if value is None:
        return None
    out = value
    for pattern in _REDACTIONS:
        out = pattern.sub(
            lambda match: (
                f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]"
            ),
            out,
        )
    return out[:max_length]


def _open_dir_component(parent_fd: int, name: str, *, create: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    before = None
    if create:
        with suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
    if not nofollow:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise OSError(_UNSAFE_DIRECTORY)
    fd = os.open(name, flags | nofollow, dir_fd=parent_fd)
    info = os.fstat(fd)
    if before is not None and (before.st_dev, before.st_ino) != (
        info.st_dev,
        info.st_ino,
    ):
        os.close(fd)
        raise OSError(_DIRECTORY_CHANGED)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        os.close(fd)
        raise OSError(_UNSAFE_DIRECTORY_OWNER)
    return fd


def _append_report(record: dict[str, object]) -> None:
    home = Path.home().resolve()
    root = storage_paths.deep_worker_home()
    relative = root.relative_to(home)
    home_fd = os.open(
        home, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    held = [home_fd]
    try:
        parent_fd = home_fd
        for component in relative.parts:
            parent_fd = _open_dir_component(parent_fd, component, create=True)
            held.append(parent_fd)
        report_dir_fd = _open_dir_component(parent_fd, _DIR_NAME, create=True)
        held.append(report_dir_fd)
        flags = os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(
                _FILE_NAME,
                flags | os.O_CREAT | os.O_EXCL,
                _FILE_MODE,
                dir_fd=report_dir_fd,
            )
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            before = os.stat(_FILE_NAME, dir_fd=report_dir_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise OSError(_UNSAFE_REPORT_TARGET) from exc
            file_fd = os.open(_FILE_NAME, flags, dir_fd=report_dir_fd)
            after = os.fstat(file_fd)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(file_fd)
                raise OSError(_REPORT_TARGET_CHANGED) from exc
        try:
            info = os.fstat(file_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
            ):
                raise OSError(_UNSAFE_REPORT_FILE)
            os.fchmod(file_fd, _FILE_MODE)
            secured = os.fstat(file_fd)
            if stat.S_IMODE(secured.st_mode) != _FILE_MODE:
                raise OSError(_UNSAFE_REPORT_FILE)
            data = (
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode()
            if os.write(file_fd, data) != len(data):
                raise OSError(_SHORT_REPORT_WRITE)
        finally:
            os.close(file_fd)
    finally:
        for fd in reversed(held):
            os.close(fd)


def make_infrastructure_feedback_tool(
    agent_name: str,
    provider: str | None = None,
    *,
    commit: str | None = None,
    dirty: bool | None = None,
) -> BaseTool:
    """Build a feedback tool bound to ``agent_name`` and ``provider``.

    Args:
        agent_name: Reporting agent identity stamped on every record.
        provider: Active model provider name. Recorded so that reports can be
            attributed to the provider the agent was running on; ``None`` when
            the caller cannot determine it.
        commit: Full git SHA of the tree this agent is running, captured at
            process start. Agents run concurrently from different worktrees on
            different commits, so the agent name alone does not identify the
            code a report describes. ``None`` when unresolvable.
        dirty: Whether that checkout had uncommitted changes. ``None`` means
            undetermined and is NOT interchangeable with ``False`` — the
            latter asserts a clean tree, which must never be fabricated.

    Returns:
        The bound infrastructure-feedback tool.
    """

    @tool(
        "report_infrastructure_feedback",
        args_schema=InfrastructureFeedbackArgs,
        description=(
            "You MUST use this tool whenever infrastructure used for your work is "
            "faulty, hangs, returns no result, is unnecessarily slow, or otherwise "
            "behaves suboptimally. You MAY also use it to propose improvements to "
            "agent infrastructure. Never include credentials, tokens, passwords, "
            "raw environment dumps, or full logs. This only records feedback; it "
            "does not authorize code changes, create issues, or open pull requests."
        ),
    )
    async def report_infrastructure_feedback(  # noqa: RUF029 - direct event-loop I/O
        kind: Literal["problem", "improvement"],
        component: str,
        summary: str,
        details: str,
        expected_behavior: str | None = None,
    ) -> dict[str, object]:
        report_id = str(uuid4())
        record: dict[str, object] = {
            "schema_version": 1,
            "report_id": report_id,
            "created_at": datetime.now(UTC).isoformat(),
            "agent_name": agent_name,
            "provider": provider,
            # Always present, even when unresolved, so consumers never need a
            # missing-key special case. Not redacted: a SHA is a hex digest
            # and a bool, neither can carry a credential.
            "commit": commit,
            "dirty": dirty,
            "kind": kind,
            "component": _redact(component, 120),
            "summary": _redact(summary, 300),
            "details": _redact(details, 4000),
            "expected_behavior": _redact(expected_behavior, 2000),
        }
        try:
            _append_report(record)
        except Exception as exc:  # noqa: BLE001 - tool boundary returns failure data
            return {"success": False, "error": type(exc).__name__, "message": str(exc)}
        return {"success": True, "report_id": report_id}

    return report_infrastructure_feedback


def assemble_common_tools(
    agent_name: str,
    provider: str | None = None,
    *,
    commit: str | None = None,
    dirty: bool | None = None,
) -> list[BaseTool]:
    """Return provider-independent tools available to every agent role.

    The tools themselves are the same for every provider; ``provider``,
    ``commit`` and ``dirty`` are only recorded on emitted reports so feedback
    can be attributed to the runtime and the exact revision the agent ran.
    """
    return [
        make_infrastructure_feedback_tool(
            agent_name, provider, commit=commit, dirty=dirty
        )
    ]
