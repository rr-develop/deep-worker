"""Pytest configuration and fixtures for claude_code_provider tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_cli: test drives the real Claude CLI subprocess; it MUST see the "
        "real $HOME (the autouse HOME-isolation fixture is skipped for it) so "
        "synthesized session files and the CLI subprocess agree on the same "
        "~/.claude location.",
    )


@pytest.fixture(autouse=True)
def _isolate_claude_session_state(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Isolate the per-thread session-state PERSISTENCE introduced for
    history-restore (DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.4) so unit tests
    never read/write the real ``~/.deep_worker`` and can't contaminate each
    other across runs.

    ``ClientManager.get_sent_count`` now lazily seeds from
    ``~/.deep_worker/<agent>/claude_session_state.json``; without isolation a
    leaked file from one test (or a prior run) seeds a non-zero sent_count in
    another, so the model treats the first message as already-sent and never
    queries Claude (observed: test_chat_model failures). Point ``Path.home()``
    at a fresh tmp dir per test and reset the global ClientManager singleton's
    in-memory state.

    EXCEPTION — ``@pytest.mark.real_cli`` tests: these spawn the actual Claude
    CLI, which reads the REAL ``~/.claude`` (a subprocess can't see our
    monkeypatched ``Path.home()``). If we redirected ``Path.home()`` here, the
    synthesized session file would land in ``tmp_path/.claude`` while the CLI
    looked under the real home → "No conversation found". So we DON'T patch
    home for them. They are responsible for not polluting real state.
    """
    if request.node.get_closest_marker("real_cli") is None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    # Typed as Any so the deliberate protected-attr reset below mirrors the
    # rest of the suite (ClaudeCodeChatModel.client_manager is also Any) and
    # does not trip reportPrivateUsage under pyright strict.
    mgr: Any = None
    try:
        from deep_worker.providers.claude_code_provider.client_manager import get_global_client_manager
        mgr = get_global_client_manager()
        mgr._sent_count.clear()
        mgr._session_epoch.clear()
        mgr.clients.clear()
    except Exception:
        pass
    yield
    if mgr is not None:
        mgr._sent_count.clear()
        mgr._session_epoch.clear()
        mgr.clients.clear()


@pytest.fixture
def mock_claude_sdk_client() -> Mock:
    """Create a mock ClaudeSDKClient for testing.

    Returns:
        Mock ClaudeSDKClient with connect, disconnect, query, and receive_response methods.
    """
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()
    client.receive_response = AsyncMock(return_value=iter([]))
    return client


@pytest.fixture
def sample_tool_metadata() -> list[dict[str, Any]]:
    """Sample tool metadata for testing.

    Returns:
        List of tool metadata dictionaries.
    """
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                },
                "required": ["arg"]
            }
        },
        {
            "name": "calculator",
            "description": "Calculate numbers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    ]
