"""Tests for ``TaskManager._initialize_manager`` web-tool wiring.

TaskManager evaluates the subordinate's work. To verify factual claims
or to look up documentation/specs that the subordinate's solution should
match, the manager needs read-only access to the web. The same set
goes into every provider's manager:

- ``fetch_url`` — fetch a known URL and return its content as markdown.

The manager does NOT get:

- ``http_request`` — supports arbitrary HTTP verbs (POST/PUT/DELETE);
  unnecessary for evaluation and a needless write-side-effect surface.
- ``send_message_to_agent`` — gives the manager autonomous peer-to-peer
  agent messaging, which is outside its role.
- ``read_image`` — not a web tool; image inspection is out of scope.

These tests patch ``create_cli_agent`` and ``create_model`` so they
run fast and provider-agnostic; we only need to inspect the ``tools=``
kwarg actually delivered to ``create_cli_agent``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make ``codex_provider`` importable without installing it.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_worker_cli.task_manager.task_manager import TaskManager  # noqa: E402


def _stub_config(provider: str) -> MagicMock:
    cfg = MagicMock()
    cfg.provider = provider
    cfg.model = None
    cfg.working_dir = Path("/tmp")
    cfg.web_search_policy = "auto"
    return cfg


def _make_tm(provider: str) -> TaskManager:
    return TaskManager(
        agent_name="t1",
        config=_stub_config(provider),
        session_manager=MagicMock(),
    )


def _bound_tool_names(mock_create_cli_agent: MagicMock) -> set[str]:
    """Pull names of tools actually passed into ``create_cli_agent``.

    Tool name lives in different places depending on the tool's shape:
    - ``BaseTool`` instances expose ``.name``.
    - Functions decorated with ``@tool`` expose ``.name`` as well.
    - Plain functions (no decorator) — like ``fetch_url`` and
      ``http_request`` — only expose ``.__name__``.
    LangChain accepts all three shapes via ``create_agent(tools=...)``.
    """
    kwargs = mock_create_cli_agent.call_args.kwargs
    tools = kwargs.get("tools") or []
    out: set[str] = set()
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            name = getattr(t, "__name__", None)
        if name:
            out.add(name)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["codex", "deepseek", "openai", "anthropic", "google", "vertexai"],
)
async def test_initialize_manager_includes_fetch_url_tool(provider: str) -> None:
    """``fetch_url`` MUST be in the manager bound tool set for every
    provider — reading a known URL as markdown is a basic capability
    the manager needs to verify subordinate references against docs."""
    tm = _make_tm(provider)
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    names = _bound_tool_names(mock_create)
    assert "fetch_url" in names, (
        f"manager bound tools must include fetch_url for provider={provider!r}. "
        f"Got: {names!r}"
    )


@pytest.mark.asyncio
async def test_initialize_manager_does_not_include_http_request() -> None:
    """``http_request`` is intentionally NOT given to the manager. It
    supports POST/PUT/DELETE — unnecessary write-side-effect surface
    for an evaluator role. If the manager ever needs richer HTTP, add
    a tool of narrower scope.
    """
    tm = _make_tm("deepseek")
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    names = _bound_tool_names(mock_create)
    assert "http_request" not in names, (
        f"http_request must NOT be in manager bound tools. Got: {names!r}"
    )


@pytest.mark.asyncio
async def test_initialize_manager_does_not_include_send_message_to_agent() -> None:
    """``send_message_to_agent`` would give the manager autonomous
    peer-to-peer messaging — outside its evaluation role. Guard
    against accidental inclusion.
    """
    tm = _make_tm("deepseek")
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    names = _bound_tool_names(mock_create)
    assert "send_message_to_agent" not in names, (
        f"send_message_to_agent must NOT be in manager bound tools. "
        f"Got: {names!r}"
    )


@pytest.mark.asyncio
async def test_codex_manager_carries_web_tools_AND_native_passthrough() -> None:
    """Cross-check: a codex-provider manager carries BOTH the web tool
    (fetch_url) AND every Codex native passthrough tool
    in the same bound list. Regression guard against accidentally
    overwriting one set with the other.
    """
    from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS

    tm = _make_tm("codex")
    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    names = _bound_tool_names(mock_create)
    assert "fetch_url" in names, names
    for nt in NATIVE_TOOLS:
        assert nt.name in names, (
            f"codex native passthrough '{nt.name}' missing from manager "
            f"bound tools after web-tool addition. Got: {names!r}"
        )


def test_system_prompt_advertises_web_tools_to_model() -> None:
    """Awareness: the manager-agent MUST be told it has web tools so
    GPT-class models actually consider them. Just binding the tool
    schema is necessary but not sufficient for some providers; the
    system prompt should name the tools.
    """
    from deep_worker_cli.task_manager.prompts import TASK_MANAGER_SYSTEM_PROMPT

    assert "web_search" in TASK_MANAGER_SYSTEM_PROMPT, (
        "TASK_MANAGER_SYSTEM_PROMPT must mention the web_search tool "
        "by name so the model knows it can use it. "
        f"Current prompt: {TASK_MANAGER_SYSTEM_PROMPT!r}"
    )
    assert "fetch_url" in TASK_MANAGER_SYSTEM_PROMPT, (
        "TASK_MANAGER_SYSTEM_PROMPT must mention the fetch_url tool "
        "by name so the model knows it can use it."
    )
