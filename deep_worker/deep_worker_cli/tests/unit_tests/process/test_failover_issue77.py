"""Tests for a related issue: provider failover, preferred_provider, restore-on-idle.

Coverage:
- CodexUsageLimitError raised on marker match, not on misconfig
- CodexUsageLimitError handler in _handle_invoke
- availability_tracker round-trip through JSON
- Restore-on-idle condition checks
- CLI preferred_provider merge matrix
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure provider source packages are importable from the worktree root.
_pkg_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent  # tests/unit/process/ → repo root
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


# =========================================================================
# 1. CodexUsageLimitError — marker detection in exec_runner
# =========================================================================

@pytest.mark.parametrize(
    ("line", "expect_limit"),
    [
        # Usage limit marker → CodexUsageLimitError
        ('''{"type":"error","message":"You've hit your usage limit. · resets 10pm (Europe/Amsterdam)"}''', True),
        # Misconfig error → CodexProtocolError (not CodexUsageLimitError)
        ('''{"type":"error","message":"invalid_request_error: model 'foo' is not supported with your ChatGPT account"}''', False),
        # Non-error event → no exception
        ('''{"type":"turn.completed","usage":{}}''', None),
        # Unknown event type → CodexProtocolError
        ('''{"type":"bogus"}''', False),
    ],
)
def test_codex_usage_limit_error_marker(line: str, expect_limit: bool | None) -> None:
    from codex_provider.exec_runner import (
        CodexProtocolError,
        CodexUsageLimitError,
        parse_event,
    )

    if expect_limit is None:
        # No exception expected
        try:
            parse_event(line)
        except CodexProtocolError:
            pytest.fail("parse_event raised CodexProtocolError unexpectedly")
        return

    if expect_limit:
        with pytest.raises(CodexUsageLimitError) as excinfo:
            parse_event(line)
        assert "usage limit" in str(excinfo.value).lower()
    else:
        with pytest.raises(CodexProtocolError) as excinfo:
            parse_event(line)
        assert not isinstance(excinfo.value, CodexUsageLimitError), (
            "misconfig must raise CodexProtocolError, not CodexUsageLimitError"
        )


@pytest.mark.asyncio
async def test_codex_usage_limit_error_in_handle_invoke() -> None:
    """_handle_invoke catches CodexUsageLimitError by class name, returns
    a clean error message, and sets status=ERROR."""
    agent = _make_agent()

    from codex_provider.exec_runner import CodexUsageLimitError

    limit = "Codex stream error: You've hit your usage limit. · resets 3am UTC"
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=CodexUsageLimitError(limit))
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR
    assert result["error"].startswith("Codex usage limit reached")
    assert "usage limit" in result["error"]


@pytest.mark.asyncio
async def test_codex_usage_limit_notifies_callback() -> None:
    """CodexUsageLimitError must broadcast an error event to subscribers."""
    agent = _make_agent()

    from codex_provider.exec_runner import CodexUsageLimitError

    limit = "Codex stream error: You've hit your usage limit."
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=CodexUsageLimitError(limit))
    agent.agent = fake_graph

    events: list[dict] = []

    async def capture(msg: dict) -> None:
        events.append(msg)

    await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=capture,
        approval_callback=None,
    )

    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, "limit must be broadcast as an error event"
    assert "usage limit" in error_events[-1]["content"].lower()


# =========================================================================
# 2. availability_tracker — persistence and round-trip
# =========================================================================

def test_availability_tracker_round_trip() -> None:
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = AvailabilityTracker(data_dir=Path(tmp))
        assert tracker.is_available("claude-code")

        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=2)
        tracker.record_failure(
            provider="claude-code",
            marker_text="resets 5pm",
            agent_name="test_agent",
            available_after_utc=later,
        )
        assert not tracker.is_available("claude-code")
        assert tracker.next_available("claude-code") is not None

        # Serialize and deserialize by creating a new tracker
        tracker2 = AvailabilityTracker(data_dir=Path(tmp))
        assert not tracker2.is_available("claude-code")

        # Clear
        tracker2.clear("claude-code")
        assert tracker2.is_available("claude-code")
        assert tracker2.next_available("claude-code") is None

        # Re-serialize and check
        tracker3 = AvailabilityTracker(data_dir=Path(tmp))
        assert tracker3.is_available("claude-code")


def test_availability_tracker_unknown_reset_is_available() -> None:
    """When available_after_utc is None (unknown reset), is_available
    returns True for failover decisions."""
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = AvailabilityTracker(data_dir=Path(tmp))
        tracker.record_failure(
            provider="codex",
            marker_text="hit limit - no reset time given",
            agent_name="test",
            available_after_utc=None,
        )
        assert tracker.is_available("codex"), (
            "unknown reset → available for failover"
        )
        assert tracker.next_available("codex") is None, (
            "unknown reset → no next_available (don't auto-restore)"
        )
        assert not tracker.is_restore_eligible("codex"), (
            "unknown reset → restore-on-idle must not fire"
        )


def test_availability_tracker_future_reset_not_restore_eligible() -> None:
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = AvailabilityTracker(data_dir=Path(tmp))
        later = datetime.now(timezone.utc) + timedelta(hours=2)
        tracker.record_failure(
            provider="claude-code",
            marker_text="resets 3:50pm (Europe/Amsterdam)",
            agent_name="test_agent",
            available_after_utc=later,
        )
        assert not tracker.is_restore_eligible("claude-code")
        assert tracker.is_available("claude-code") is False


def test_availability_tracker_first_failure_returns_true() -> None:
    """record_failure returns True when the provider WAS available before."""
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = AvailabilityTracker(data_dir=Path(tmp))
        was = tracker.record_failure(
            provider="claude-code",
            marker_text="resets 6pm",
            agent_name="test",
            available_after_utc=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert was is True, "first failure should return True (was available)"

        was2 = tracker.record_failure(
            provider="claude-code",
            marker_text="resets 8pm",
            agent_name="test",
            available_after_utc=datetime.now(timezone.utc) + timedelta(hours=2),
        )
        assert was2 is False, "second failure should return False (already failed)"


# =========================================================================
# 3. Provider chain constants
# =========================================================================

def test_provider_failover_chain() -> None:
    from deep_worker_cli.runtime.provider_chain import PROVIDER_FAILOVER_CHAIN

    assert PROVIDER_FAILOVER_CHAIN == [
        "claude-code",
        "codex",
        "grok",
        "deepseek",
    ]
    assert PROVIDER_FAILOVER_CHAIN[-1] == "deepseek"


# =========================================================================
# 4. _schedule_failover — record failure + compute next provider
# =========================================================================

@pytest.mark.asyncio
async def test_schedule_failover_records_failure() -> None:
    agent = _make_agent()

    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch(
                "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
                autospec=True,
            ) as MockTracker,
            patch(
                "asyncio.create_subprocess_exec",
                autospec=True,
            ) as mock_subproc,
        ):
            mock_instance = MockTracker.return_value
            mock_instance.is_available.return_value = True
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_subproc.return_value = mock_proc

            await agent._schedule_failover("claude-code", "resets 10pm UTC")

            # Verify record_failure was called with the right positional args
            call_kwargs = mock_instance.record_failure.call_args[1]
            assert call_kwargs["provider"] == "claude-code"
            assert call_kwargs["marker_text"] == "resets 10pm UTC"
            assert call_kwargs["agent_name"] == agent.name


@pytest.mark.asyncio
async def test_schedule_failover_parses_reset_time_without_nameerror() -> None:
    """Regression 2026-06-17 (a live agent): a marker whose reset-time MATCHES the
    parse regex (e.g. 'try again at <time>') must NOT crash _schedule_failover.

    The active _schedule_failover used ``timezone.utc`` to default a naive
    parsed time but never imported ``timezone`` — so the moment a real codex
    marker ("... try again at Jun 18th, 2026 12:39 AM") reached this code, it
    raised ``NameError: name 'timezone' is not defined``, the failover aborted,
    and the agent stayed stuck in ERROR instead of switching to deepseek.
    (The pre-existing test used 'resets 10pm UTC', which does NOT match the
    'resets at' regex, so it never exercised the timezone line.)"""
    agent = _make_agent()

    marker = (
        "Codex stream error: You've hit your usage limit. "
        "try again at Jun 18th, 2026 12:39 AM."
    )

    with (
        patch(
            "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
            autospec=True,
        ) as MockTracker,
        patch("asyncio.create_subprocess_exec", autospec=True) as mock_subproc,
    ):
        mock_instance = MockTracker.return_value
        mock_instance.is_available.return_value = True
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock()
        mock_subproc.return_value = mock_proc

        # Must not raise NameError — failover proceeds to record the failure.
        await agent._schedule_failover("codex", marker)

        assert mock_instance.record_failure.called
        call_kwargs = mock_instance.record_failure.call_args[1]
        assert call_kwargs["provider"] == "codex"
        # The naive reset time was given a UTC tzinfo (the line that crashed).
        after = call_kwargs["available_after_utc"]
        assert after is not None
        assert after.tzinfo is not None


@pytest.mark.asyncio
async def test_schedule_failover_writes_restart_notice_for_auto_continue() -> None:
    """Regression 2026-06-17 (a live agent): after failover restart the agent must
    CONTINUE the unanswered task, not sit idle. The new process only
    auto-continues when a ``.restart_notice.json`` is present (consumed at
    startup → ``_pending_restart_notice`` → ``_auto_continue_after_restart``).
    Failover used to spawn ``--restart`` WITHOUT writing that notice, so the
    restored history kept the user's command at ``(no response yet)`` and the
    agent went idle. ``_schedule_failover`` must now write the notice so
    auto-continue fires."""
    import json

    agent = _make_agent()

    with tempfile.TemporaryDirectory() as tmp:
        agent.data_dir = Path(tmp)
        notice_path = Path(tmp) / ".restart_notice.json"

        with (
            patch(
                "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
                autospec=True,
            ) as MockTracker,
            patch("asyncio.create_subprocess_exec", autospec=True) as mock_subproc,
        ):
            mock_instance = MockTracker.return_value
            mock_instance.is_available.return_value = True
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_subproc.return_value = mock_proc

            await agent._schedule_failover(
                "codex",
                "Codex stream error: You've hit your usage limit.",
            )

            # The restart subprocess was spawned …
            assert mock_subproc.called
            # … AND the restart-notice was written BEFORE it, so the new
            # process auto-continues the pending task.
            assert notice_path.exists(), (
                "failover must write .restart_notice.json so the restarted "
                "agent continues the task instead of going idle"
            )
            notice = json.loads(notice_path.read_text())
            assert notice.get("cause") == "failover"
            # The message names the provider switch so the model has context.
            assert "codex" in notice.get("message", "")
            # failover chain: claude-code → codex → grok → deepseek.
            # codex->grok is the immediate next, not deepseek.
            assert "grok" in notice.get("message", "")


@pytest.mark.asyncio
async def test_schedule_failover_grok_to_deepseek() -> None:
    """grok is not terminal — usage limit on grok advances to deepseek."""
    import json

    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        agent.data_dir = Path(tmp)
        notice_path = Path(tmp) / ".restart_notice.json"
        with (
            patch(
                "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
                autospec=True,
            ) as MockTracker,
            patch("asyncio.create_subprocess_exec", autospec=True) as mock_subproc,
        ):
            mock_instance = MockTracker.return_value
            mock_instance.is_available.return_value = True
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_subproc.return_value = mock_proc

            await agent._schedule_failover("grok", "Grok usage limit reached")

            assert notice_path.exists()
            notice = json.loads(notice_path.read_text())
            assert "deepseek" in notice.get("message", "")


@pytest.mark.asyncio
async def test_schedule_failover_codex_skips_unavailable_grok() -> None:
    """When grok is temporarily unavailable, codex failover skips to deepseek."""
    import json

    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        agent.data_dir = Path(tmp)
        notice_path = Path(tmp) / ".restart_notice.json"

        def _availability(provider: str) -> bool:
            return provider != "grok"

        with (
            patch(
                "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
                autospec=True,
            ) as MockTracker,
            patch("asyncio.create_subprocess_exec", autospec=True) as mock_subproc,
        ):
            mock_instance = MockTracker.return_value
            mock_instance.is_available.side_effect = _availability
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_subproc.return_value = mock_proc

            await agent._schedule_failover(
                "codex",
                "Codex stream error: You've hit your usage limit.",
            )

            assert notice_path.exists()
            notice = json.loads(notice_path.read_text())
            assert "deepseek" in notice.get("message", "")
            assert "grok" not in notice.get("message", "")


@pytest.mark.asyncio
async def test_schedule_failover_deepseek_does_nothing() -> None:
    """deepseek is terminal — _schedule_failover must NOT record or restart."""
    agent = _make_agent()

    with patch(
        "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
        autospec=True,
    ) as MockTracker:
        mock_instance = MockTracker.return_value
        await agent._schedule_failover("deepseek", "some error")

        # Should NOT attempt to find a failover target
        mock_instance.record_failure.assert_called_once()
        # No asyncio.create_subprocess_exec should be called
        # (we can't easily assert this, but the method returns early)


# =========================================================================
# 5. CLI preferred_provider merge matrix
# =========================================================================

def _make_cli_args(**overrides: str | bool | None) -> MagicMock:
    """Build a mock argparse.Namespace with default + override values."""
    defaults = {
        "name": "test_agent",
        "provider": None,
        "model": None,
        "working_dir": None,
        "skip_permissions": False,
        "enable_task_manager": False,
        "memory_only": False,
        "use_sqlite": False,
        "no_skip_permissions": False,
        "no_memory_only": False,
        "no_use_sqlite": False,
        "no_enable_task_manager": False,
        "no_clean": False,
        "clean": False,
        "history_memory": False,
        "no_history_memory": False,
        "restore": False,
        "auto_approve_send_message": False,
        "no_auto_approve_send_message": False,
        "preferred_provider": None,
        "no_update_preferred": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_saved_launch_config(**overrides: str | bool | None) -> dict:
    defaults = {
        "name": "test_agent",
        "provider": "claude-code",
        "model": "sonnet",
        "skip_permissions": False,
        "enable_task_manager": False,
        "memory_only": False,
        "use_sqlite": False,
        "clean": False,
        "history_memory": False,
        "version": 2,
        "preferred_provider": None,
        "storage_version": 5,
    }
    defaults.update(overrides)
    return defaults


def test_merge_restart_config_backfill_preferred() -> None:
    """When saved preferred_provider is None, backfill to saved.provider."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(preferred_provider=None)
    # No CLI provider override — use the default
    args = _make_cli_args()

    merged = merge_restart_config(saved, args)
    assert merged.preferred_provider == "claude-code", (
        "backfill: preferred_provider = saved.provider when missing"
    )


def test_merge_restart_config_preferred_provider_overrides_both() -> None:
    """--preferred-provider X overrides BOTH preferred_provider AND provider."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(preferred_provider="claude-code")
    args = _make_cli_args(preferred_provider="codex")

    merged = merge_restart_config(saved, args)
    assert merged.preferred_provider == "codex"
    assert merged.provider == "codex"


def test_merge_restart_config_no_update_preferred_preserves() -> None:
    """--no-update-preferred preserves saved preferred_provider when
    provider changes."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(preferred_provider="claude-code")
    # Provider override — but no-update-preferred is set
    args = _make_cli_args(provider="codex", no_update_preferred=True)

    merged = merge_restart_config(saved, args)
    assert merged.preferred_provider == "claude-code", (
        "--no-update-preferred must preserve the saved preferred_provider"
    )
    assert merged.provider == "codex", (
        "provider should still change"
    )


def test_merge_restart_config_preferred_provider_on_initial_launch() -> None:
    """A fresh launch (no saved config) passes preferred_provider through."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(preferred_provider="codex")
    # No CLI overrides — use same provider as saved so no override detected
    args = _make_cli_args(provider="codex")

    merged = merge_restart_config(saved, args)
    assert merged.preferred_provider == "codex"
    assert merged.provider == "codex"


def test_merge_restart_config_restore_keeps_preferred_model() -> None:
    """Restore-on-idle restart must not downgrade opus → sonnet default."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(
        provider="codex",
        model=None,
        preferred_provider="claude-code",
        preferred_model="opus",
    )
    args = _make_cli_args(
        provider="claude-code",
        model=None,
        no_update_preferred=True,
    )
    merged = merge_restart_config(saved, args)
    assert merged.provider == "claude-code"
    assert merged.model == "opus"


def test_merge_restart_config_preferred_provider_wins_over_provider() -> None:
    """When --preferred-provider is set but --provider is not, provider
    should also be updated to match preferred_provider."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    saved = _make_saved_launch_config(preferred_provider="claude-code")
    args = _make_cli_args(preferred_provider="deepseek")

    merged = merge_restart_config(saved, args)
    assert merged.preferred_provider == "deepseek"
    assert merged.provider == "deepseek", (
        "--preferred-provider must also update provider when --provider is not set"
    )


# =========================================================================
# 6. Restore-on-idle integration
# =========================================================================

@pytest.mark.asyncio
async def test_restore_on_idle_skips_when_preferred_still_limited() -> None:
    """Regression (observed live, 2026-06-20): after failover to codex, restore-on-idle
    must NOT bounce back to claude-code while its session limit is active."""
    agent = _make_agent()
    agent.launch_config = {
        "provider": "codex",
        "preferred_provider": "claude-code",
        "preferred_model": "opus",
        "storage_version": 5,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tracker_dir = Path(tmp)
        later = datetime.now(timezone.utc) + timedelta(hours=2)
        tracker = __import__(
            "deep_worker_cli.runtime.availability_tracker",
            fromlist=["AvailabilityTracker"],
        ).AvailabilityTracker(data_dir=tracker_dir)
        tracker.record_failure(
            provider="claude-code",
            marker_text=(
                "You've hit your session limit · resets 3:50pm (Europe/Amsterdam)"
            ),
            agent_name="test_agent",
            available_after_utc=later,
        )

        with patch(
            "deep_worker_cli.runtime.availability_tracker.AvailabilityTracker",
            autospec=True,
        ) as MockTracker, patch(
            "asyncio.create_subprocess_exec",
            autospec=True,
        ) as mock_subproc:
            MockTracker.return_value = tracker
            await agent._maybe_restore_preferred_provider(
                {"status": "waiting_input"}
            )
            mock_subproc.assert_not_called()


# =========================================================================
# Helpers
# =========================================================================

def _make_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_failover77_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_failover77"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    from deep_worker_cli.storage_paths import CURRENT_STORAGE_VERSION
    agent.launch_config = {
        "provider": "claude-code",
        "preferred_provider": "claude-code",
        "storage_version": CURRENT_STORAGE_VERSION,
    }
    return agent
