from __future__ import annotations

# ruff: noqa: ASYNC240
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

import deepagents
from deep_worker_cli import (
    agent as agent_module,
    infrastructure_feedback as feedback_module,
    storage_paths,
)
from deep_worker_cli.agent import _merge_common_tools_into_subagents
from deep_worker_cli.infrastructure_feedback import (
    InfrastructureFeedbackArgs,
    _open_dir_component,
    assemble_common_tools,
    make_infrastructure_feedback_tool,
)
from deep_worker_cli.runtime.provider_chain import all_provider_names


@pytest.fixture
def isolated_feedback_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    root = home / ".deep_worker"
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.setattr(storage_paths, "deep_worker_home", lambda _home=None: root)
    return root


def test_schema_rejects_invalid_and_oversized_values() -> None:
    with pytest.raises(ValidationError):
        InfrastructureFeedbackArgs(
            kind="other", component="lsp", summary="bad", details="x"
        )
    with pytest.raises(ValidationError):
        InfrastructureFeedbackArgs(
            kind="problem", component="lsp", summary="x" * 301, details="x"
        )


@pytest.mark.asyncio
async def test_real_tool_appends_private_redacted_records(
    isolated_feedback_home: Path,
) -> None:
    first = make_infrastructure_feedback_tool("agent-a")
    second = make_infrastructure_feedback_tool("agent-b:tm")
    result1 = await first.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "find_symbol hung",
            "details": "Authorization: Bearer secret-token after 90 seconds",
        }
    )
    result2 = await second.ainvoke(
        {
            "kind": "improvement",
            "component": "project_memory",
            "summary": "show index freshness",
            "details": "Expose the last completed indexing timestamp",
        }
    )
    assert result1["success"] is True
    assert result2["success"] is True
    directory = isolated_feedback_home / "infrastructure_feedback"
    path = directory / "reports.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["agent_name"] for record in records] == ["agent-a", "agent-b:tm"]
    assert [record["provider"] for record in records] == [None, None]
    assert "secret-token" not in records[0]["details"]
    assert "[REDACTED]" in records[0]["details"]
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_redaction_reapplies_all_field_bounds_and_handles_quoted_values(
    isolated_feedback_home: Path,
) -> None:
    suffixes = {
        "component": (120, ' password="multi word"'),
        "summary": (300, " secret=x"),
        "details": (4000, " api_key=x"),
        "expected_behavior": (2000, " access_token=x"),
    }
    arguments = {"kind": "problem"}
    for field, (limit, suffix) in suffixes.items():
        arguments[field] = "x" * (limit - len(suffix)) + suffix

    result = await make_infrastructure_feedback_tool("agent").ainvoke(arguments)

    assert result["success"] is True
    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    for field, (limit, _suffix) in suffixes.items():
        assert len(record[field]) <= limit
    serialized = json.dumps(record)
    assert "multi word" not in serialized
    assert "api_key=x" not in serialized
    assert "access_token=x" not in serialized


@pytest.mark.asyncio
async def test_async_invocation_stays_on_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_thread = threading.get_ident()
    writer_threads: list[int] = []
    monkeypatch.setattr(
        feedback_module,
        "_append_report",
        lambda _record: writer_threads.append(threading.get_ident()),
    )

    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "runtime",
            "summary": "thread probe",
            "details": "probe",
        }
    )

    assert result["success"] is True
    assert writer_threads == [caller_thread]


@pytest.mark.asyncio
async def test_symlink_target_fails_closed(
    isolated_feedback_home: Path,
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    victim = isolated_feedback_home / "victim"
    victim.write_text("unchanged")
    (directory / "reports.jsonl").symlink_to(victim)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "unsafe path",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert victim.read_text() == "unchanged"


@pytest.mark.asyncio
async def test_unsafe_directory_mode_fails_closed(
    isolated_feedback_home: Path,
) -> None:
    isolated_feedback_home.mkdir(mode=0o755)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "unsafe directory",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert not (isolated_feedback_home / "infrastructure_feedback").exists()


@pytest.mark.asyncio
async def test_unsafe_directory_owner_fails_closed(
    isolated_feedback_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    root_inode = isolated_feedback_home.stat().st_ino
    real_fstat = os.fstat

    def wrong_owner_for_root(fd: int) -> os.stat_result | SimpleNamespace:
        info = real_fstat(fd)
        if info.st_ino == root_inode:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_uid=os.geteuid() + 1,
                st_nlink=info.st_nlink,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
            )
        return info

    monkeypatch.setattr(os, "fstat", wrong_owner_for_root)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "wrong directory owner",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert not (isolated_feedback_home / "infrastructure_feedback").exists()


@pytest.mark.asyncio
async def test_create_and_existing_file_open_branches_use_required_flags(
    isolated_feedback_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not isolated_feedback_home.exists()
    real_open = os.open
    report_flags: list[int] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "reports.jsonl":
            report_flags.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", recording_open)
    tool = make_infrastructure_feedback_tool("agent")
    arguments = {
        "kind": "problem",
        "component": "filesystem",
        "summary": "branch check",
        "details": "test",
    }
    assert (await tool.ainvoke(arguments))["success"] is True
    assert report_flags[-1] & os.O_CREAT
    assert report_flags[-1] & os.O_EXCL

    report_flags.clear()
    assert (await tool.ainvoke(arguments))["success"] is True
    assert report_flags[0] & os.O_EXCL
    assert not report_flags[-1] & os.O_EXCL


@pytest.mark.asyncio
async def test_non_regular_target_fails_before_write(
    isolated_feedback_home: Path,
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    target = directory / "reports.jsonl"
    os.mkfifo(target, mode=0o600)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "non-regular target",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert stat.S_ISFIFO(target.lstat().st_mode)


@pytest.mark.asyncio
async def test_link_count_failure_precedes_chmod_and_write(
    isolated_feedback_home: Path,
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    target = directory / "reports.jsonl"
    target.write_text("unchanged")
    target.chmod(0o640)
    os.link(target, directory / "second-link")
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "linked target",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert target.read_text() == "unchanged"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.asyncio
async def test_wrong_file_owner_precedes_chmod_and_write(
    isolated_feedback_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    target = directory / "reports.jsonl"
    target.write_text("unchanged")
    target.chmod(0o640)
    real_fstat = os.fstat

    def wrong_owner_for_file(fd: int) -> os.stat_result | SimpleNamespace:
        info = real_fstat(fd)
        if stat.S_ISREG(info.st_mode):
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_uid=os.geteuid() + 1,
                st_nlink=info.st_nlink,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
            )
        return info

    monkeypatch.setattr(os, "fstat", wrong_owner_for_file)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "wrong owner",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert target.read_text() == "unchanged"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.asyncio
async def test_noop_fchmod_fails_before_write(
    isolated_feedback_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    target = directory / "reports.jsonl"
    target.write_text("unchanged")
    target.chmod(0o640)
    monkeypatch.setattr(os, "fchmod", lambda _fd, _mode: None)

    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "chmod mutation",
            "details": "test",
        }
    )

    assert result["success"] is False
    assert target.read_text() == "unchanged"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.asyncio
async def test_report_inode_swap_precedes_chmod_and_write(
    isolated_feedback_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_feedback_home.mkdir(mode=0o700)
    directory = isolated_feedback_home / "infrastructure_feedback"
    directory.mkdir(mode=0o700)
    target = directory / "reports.jsonl"
    target.write_text("original")
    target.chmod(0o640)
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "reports.jsonl" and not flags & os.O_EXCL and not swapped:
            swapped = True
            target.rename(directory / "old-report")
            target.write_text("replacement")
            target.chmod(0o640)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)
    result = await make_infrastructure_feedback_tool("agent").ainvoke(
        {
            "kind": "problem",
            "component": "filesystem",
            "summary": "inode swap",
            "details": "test",
        }
    )
    assert result["success"] is False
    assert swapped, result
    assert target.read_text() == "replacement"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_directory_component_swap_fails_closed_without_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    component = parent / "component"
    component.mkdir(mode=0o700)
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "component" and dir_fd == parent_fd and not swapped:
            swapped = True
            component.rename(parent / "original")
            component.mkdir(mode=0o700)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(os, "open", swapping_open)
    try:
        with pytest.raises(OSError, match="changed during open"):
            _open_dir_component(parent_fd, "component")
    finally:
        os.close(parent_fd)


def test_description_states_mandatory_and_optional_policy() -> None:
    description = make_infrastructure_feedback_tool("agent").description
    assert "MUST use this tool" in description
    assert "MAY also use it" in description
    assert "Never include credentials" in description


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(all_provider_names()))
async def test_report_records_the_bound_provider(
    provider: str,
    isolated_feedback_home: Path,
) -> None:
    tool = make_infrastructure_feedback_tool("agent", provider=provider)

    result = await tool.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "find_symbol hung",
            "details": "no result after 90 seconds",
        }
    )

    assert result["success"] is True
    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    assert json.loads(path.read_text())["provider"] == provider


@pytest.mark.parametrize("provider", sorted(all_provider_names()))
def test_common_tool_exists_for_every_registered_provider(provider: str) -> None:
    tools = assemble_common_tools(f"agent-{provider}", provider=provider)
    assert [tool.name for tool in tools] == ["report_infrastructure_feedback"]


def test_common_tool_is_independent_of_web_policy() -> None:
    assert assemble_common_tools("agent")[0].name == "report_infrastructure_feedback"


@pytest.mark.parametrize("provider", sorted(all_provider_names()))
@pytest.mark.parametrize("role_suffix", ["", ":tm"])
def test_create_cli_agent_wires_common_tool_for_every_provider_and_role(
    provider: str,
    role_suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        def with_config(self, _config: object) -> FakeGraph:
            return self

    def capture_create_deep_agent(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(deepagents, "create_deep_agent", capture_create_deep_agent)
    monkeypatch.setattr(agent_module, "list_subagents", lambda **_kwargs: [])
    assistant_id = f"matrix-{provider}{role_suffix}"
    agent_module.create_cli_agent(
        model=FakeListChatModel(responses=["ok"]),
        assistant_id=assistant_id,
        tools=[],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        project_memory_dir=tmp_path,
        provider=provider,
    )

    assert "report_infrastructure_feedback" in {tool.name for tool in captured["tools"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(all_provider_names()))
async def test_create_cli_agent_binds_real_provider_into_reports(
    provider: str,
    tmp_path: Path,
    isolated_feedback_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        def with_config(self, _config: object) -> FakeGraph:
            return self

    def capture_create_deep_agent(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(deepagents, "create_deep_agent", capture_create_deep_agent)
    monkeypatch.setattr(agent_module, "list_subagents", lambda **_kwargs: [])
    agent_module.create_cli_agent(
        model=FakeListChatModel(responses=["ok"]),
        assistant_id=f"bound-{provider}",
        tools=[],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        project_memory_dir=tmp_path,
        provider=provider,
    )

    tool = next(
        candidate
        for candidate in captured["tools"]
        if candidate.name == "report_infrastructure_feedback"
    )
    # No permission repair here on purpose: production must leave the storage
    # root private, so the bound tool writes on the first attempt.
    assert stat.S_IMODE(isolated_feedback_home.stat().st_mode) == 0o700
    result = await tool.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "hung",
            "details": "no result",
        }
    )

    assert result["success"] is True
    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    assert json.loads(path.read_text())["provider"] == provider


def test_general_purpose_inherits_common_default_tool() -> None:
    common = assemble_common_tools("agent")
    assert common[0].name == "report_infrastructure_feedback"


def test_configured_subagent_explicit_tools_keep_common_tool() -> None:
    explicit = make_infrastructure_feedback_tool("duplicate")
    merged = _merge_common_tools_into_subagents(
        [{"name": "configured", "tools": [explicit]}],
        assemble_common_tools("agent"),
    )
    assert [tool.name for tool in merged[0]["tools"]] == [
        "report_infrastructure_feedback"
    ]


def test_compiled_subagent_is_rejected_when_common_tool_cannot_be_injected() -> None:
    with pytest.raises(ValueError, match="compiled subagents"):
        _merge_common_tools_into_subagents(
            [{"name": "compiled", "runnable": object()}],
            assemble_common_tools("agent"),
        )


# =========================================================================
# commit provenance: which revision produced this report
# =========================================================================

@pytest.mark.asyncio
async def test_report_records_commit_and_dirty(
    isolated_feedback_home: Path,
) -> None:
    """The bound commit and dirty flag land on the record verbatim.

    Agents run concurrently from different worktrees on different commits, so
    a report that names only the agent cannot be tied back to the code that
    produced it.
    """
    sha = "a" * 40
    tool = make_infrastructure_feedback_tool(
        "agent", "claude-code", commit=sha, dirty=False
    )

    result = await tool.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "hangs",
            "details": "no result",
        }
    )

    assert result["success"] is True
    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert record["commit"] == sha
    assert record["dirty"] is False


@pytest.mark.asyncio
async def test_dirty_true_is_recorded_and_sha_still_reported(
    isolated_feedback_home: Path,
) -> None:
    """dirty=True must survive to the record alongside the SHA.

    The SHA on its own would claim the agent ran exactly that commit. When the
    checkout was modified that claim is false, and the flag is the only thing
    that says so.
    """
    sha = "b" * 40
    tool = make_infrastructure_feedback_tool(
        "agent", "claude-code", commit=sha, dirty=True
    )

    await tool.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "hangs",
            "details": "no result",
        }
    )

    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert record["commit"] == sha
    assert record["dirty"] is True


@pytest.mark.asyncio
async def test_unknown_commit_records_null_not_a_fake_clean(
    isolated_feedback_home: Path,
) -> None:
    """Unresolvable provenance is null/None — never a fabricated clean tree.

    Defaulting dirty to False here would let a report assert "ran exactly this
    commit, unmodified" on evidence we never had.
    """
    tool = make_infrastructure_feedback_tool("agent")

    await tool.ainvoke(
        {
            "kind": "improvement",
            "component": "cli",
            "summary": "nicer output",
            "details": "please",
        }
    )

    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert "commit" in record and "dirty" in record, (
        "the fields must always be present so consumers need no special case"
    )
    assert record["commit"] is None
    assert record["dirty"] is None


@pytest.mark.asyncio
async def test_assemble_common_tools_forwards_commit_provenance(
    isolated_feedback_home: Path,
) -> None:
    """The assembly helper must not silently drop the provenance it is given.

    Asserting only that a tool comes back would stay green if the helper
    dropped commit/dirty on the floor, so drive the tool and read the record.
    """
    sha = "c" * 40
    tools = assemble_common_tools("agent", "claude-code", commit=sha, dirty=True)
    tool = next(t for t in tools if t.name == "report_infrastructure_feedback")

    await tool.ainvoke(
        {
            "kind": "problem",
            "component": "cli",
            "summary": "dropped",
            "details": "provenance",
        }
    )

    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert record["commit"] == sha
    assert record["dirty"] is True


@pytest.mark.asyncio
async def test_create_cli_agent_binds_commit_provenance_into_reports(
    tmp_path: Path,
    isolated_feedback_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: what create_cli_agent is given must reach the record.

    The tool is built deep inside agent construction, so a break anywhere in
    the chain would silently produce reports with null provenance rather than
    an error.
    """
    captured: dict[str, object] = {}

    class FakeGraph:
        def with_config(self, _config: object) -> FakeGraph:
            return self

    def capture_create_deep_agent(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(deepagents, "create_deep_agent", capture_create_deep_agent)
    monkeypatch.setattr(agent_module, "list_subagents", lambda **_kwargs: [])
    sha = "d" * 40
    agent_module.create_cli_agent(
        model=FakeListChatModel(responses=["ok"]),
        assistant_id="provenance-agent",
        tools=[],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        project_memory_dir=tmp_path,
        provider="claude-code",
        commit=sha,
        dirty=True,
    )

    tool = next(
        candidate
        for candidate in captured["tools"]
        if candidate.name == "report_infrastructure_feedback"
    )
    result = await tool.ainvoke(
        {
            "kind": "problem",
            "component": "lsp",
            "summary": "hung",
            "details": "no result",
        }
    )

    assert result["success"] is True
    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert record["commit"] == sha
    assert record["dirty"] is True


@pytest.mark.asyncio
async def test_task_manager_reports_carry_the_same_revision(
    tmp_path: Path,
    isolated_feedback_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TaskManager's own reports must carry the process revision.

    The TM builds a second agent through its own create_cli_agent call. The
    provider is recoverable there from the model, but the revision is not —
    if it is not threaded through the constructor, TM-filed feedback silently
    reports null provenance and cannot be traced to the code that produced it.
    """
    from deep_worker_cli.task_manager import TaskManager

    captured: dict[str, object] = {}

    class FakeGraph:
        def with_config(self, _config: object) -> FakeGraph:
            return self

    def capture_create_deep_agent(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(deepagents, "create_deep_agent", capture_create_deep_agent)
    monkeypatch.setattr(agent_module, "list_subagents", lambda **_kwargs: [])
    monkeypatch.setattr(
        "deep_worker_cli.config.create_model",
        lambda **_kwargs: FakeListChatModel(responses=["ok"]),
    )

    sha = "e" * 40
    tm = TaskManager(
        agent_name="tm_provenance",
        config=SimpleNamespace(
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
            working_dir=str(tmp_path),
        ),
        session_manager=SimpleNamespace(),
        commit=sha,
        dirty=True,
    )
    await tm.start()

    tool = next(
        candidate
        for candidate in captured["tools"]
        if candidate.name == "report_infrastructure_feedback"
    )
    await tool.ainvoke(
        {
            "kind": "problem",
            "component": "cli",
            "summary": "tm hit this",
            "details": "filed by the manager agent",
        }
    )

    path = isolated_feedback_home / "infrastructure_feedback" / "reports.jsonl"
    record = json.loads(path.read_text())
    assert record["commit"] == sha, (
        "TaskManager reports must name the revision the process runs"
    )
    assert record["dirty"] is True
