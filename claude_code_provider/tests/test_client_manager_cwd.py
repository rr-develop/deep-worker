"""TDD: ClientManager pins the Claude CLI subprocess ``cwd`` to the agent's
``working_dir`` so session transcripts are written under the agent's own
project slug (``~/.claude/projects/<sanitize(working_dir)>/``) — the SAME slug
the restore path reads from.

See ``docs/DESIGN_FIX_CWD_SLUG_MISMATCH_ON_RESTART.md``.

The bug: ``get_or_create_client`` built ``ClaudeAgentOptions`` WITHOUT ``cwd``,
so the CLI inherited the daemon cwd (``/``) and wrote transcripts under the
shared ``-`` slug, while ``resolve_restore_session``/``session_synthesis`` read
from ``slug(working_dir)``. The mismatch made restore find no candidate →
synthesis → resume-under-wrong-slug → fresh empty session → "forgets".

These tests capture the ``options`` actually handed to ``ClaudeSDKClient`` and
assert ``options.cwd`` tracks ``working_dir`` in BOTH construction branches
(default + ``replace(base_options)``), is left ``None`` when no ``working_dir``
is given (inherit, unchanged), and survives the ``resume`` ``replace`` call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from claude_code_provider.client_manager import ClientManager
from claude_code_provider.custom_types import ToolMetadata


def _fake_client_capturing_options(captured: list[Any]) -> Any:
    """A fake ``ClaudeSDKClient`` factory that records the ``options`` it is
    constructed with and returns a connect-able mock."""

    def _factory(options: Any = None) -> Mock:
        captured.append(options)
        client = Mock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.query = AsyncMock()
        return client

    return _factory


@pytest.fixture
def tools_metadata() -> list[ToolMetadata]:
    return []


class TestCwdPinnedToWorkingDir:
    """``options.cwd`` must equal the agent's ``working_dir``."""

    @pytest.mark.asyncio
    async def test_default_branch_sets_cwd(
        self, tools_metadata: list[ToolMetadata]
    ) -> None:
        """base_options=None (default ClaudeAgentOptions branch) → cwd set."""
        manager = ClientManager(base_options=None)
        captured: list[Any] = []
        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient",
            side_effect=_fake_client_capturing_options(captured),
        ):
            await manager.get_or_create_client(
                thread_id="t-default",
                tools_metadata=tools_metadata,
                working_dir="/path/to/working_dir",
            )
        assert captured, "ClaudeSDKClient was never constructed"
        assert captured[-1].cwd == "/path/to/working_dir"

    @pytest.mark.asyncio
    async def test_replace_branch_sets_cwd_and_keeps_base_fields(
        self, tools_metadata: list[ToolMetadata]
    ) -> None:
        """base_options present (replace branch) → cwd set, base fields kept."""
        from claude_agent_sdk import ClaudeAgentOptions

        base = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=base)
        captured: list[Any] = []
        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient",
            side_effect=_fake_client_capturing_options(captured),
        ):
            await manager.get_or_create_client(
                thread_id="t-replace",
                tools_metadata=tools_metadata,
                working_dir="/path/to/working_dir",
            )
        opts = captured[-1]
        assert opts.cwd == "/path/to/working_dir"
        # base field preserved through replace()
        assert opts.model == "claude-sonnet-4-5-20250929"
        # provider invariants still applied
        assert opts.permission_mode == "bypassPermissions"

    @pytest.mark.asyncio
    async def test_working_dir_none_leaves_cwd_none(
        self, tools_metadata: list[ToolMetadata]
    ) -> None:
        """No working_dir → cwd stays None (inherit, unchanged behavior)."""
        manager = ClientManager(base_options=None)
        captured: list[Any] = []
        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient",
            side_effect=_fake_client_capturing_options(captured),
        ):
            await manager.get_or_create_client(
                thread_id="t-none",
                tools_metadata=tools_metadata,
                working_dir=None,
            )
        assert captured[-1].cwd is None

    @pytest.mark.asyncio
    async def test_resume_path_preserves_cwd(
        self, tools_metadata: list[ToolMetadata]
    ) -> None:
        """resume replace() must carry BOTH resume and cwd (Q4, observed in live testing)."""
        manager = ClientManager(base_options=None)
        captured: list[Any] = []
        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient",
            side_effect=_fake_client_capturing_options(captured),
        ):
            await manager.get_or_create_client(
                thread_id="t-resume",
                tools_metadata=tools_metadata,
                working_dir="/path/to/working_dir",
                resume_session_id="34927a7f-dd53-4db9-b4d2-286e59bd7f29",
            )
        opts = captured[-1]
        assert opts.resume == "34927a7f-dd53-4db9-b4d2-286e59bd7f29"
        assert opts.fork_session is True
        # The whole point: resume replace() did NOT drop cwd.
        assert opts.cwd == "/path/to/working_dir"


class TestWriteReadSlugAgreement:
    """Guard: the slug used to WRITE a synthesized transcript and the slug the
    resumed CLI READS from (its cwd) are the same function of working_dir."""

    def test_synthesis_dir_matches_cwd_slug(self) -> None:
        from claude_code_provider.session_synthesis import session_project_dir

        wd = "/path/to/working_dir"
        # The directory synthesis writes into is derived purely from cwd; the
        # resumed client we now launch uses cwd=wd, so the CLI looks in the
        # very same place. Encode that invariant.
        write_dir = session_project_dir(wd)
        assert write_dir.name == "-path-to-working-dir"
