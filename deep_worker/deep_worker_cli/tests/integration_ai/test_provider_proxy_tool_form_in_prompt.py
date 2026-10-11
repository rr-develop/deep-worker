"""Integration test (no mocks): the generated system prompt carries the
PROVIDER-CORRECT MCP-proxy invocation form for base deep-worker tools.

This exercises the REAL code path end-to-end with REAL provider model
objects (no MagicMock): ``_provider_of_model`` detects the provider from the
actual ``CodexChatModel`` / ``ClaudeCodeChatModel`` instance, and the real
``get_system_prompt`` renders the provider-specific note. It does NOT spin up
a CLI subprocess (no API keys / network), so it is deterministic and fast,
yet uses no test doubles.

See docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md (RC-D).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The provider packages live at the repo root (siblings of deep_worker/).
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _provider_of(model: object) -> str | None:
    from deep_worker_cli.agent import _provider_of_model

    return _provider_of_model(model)


def _prompt_for(model: object) -> str:
    from deep_worker_cli.agent import _provider_of_model, get_system_prompt

    return get_system_prompt(
        assistant_id="probe_agent",
        sandbox_type=None,
        project_memory_dir=None,
        provider=_provider_of_model(model),
    )


def test_codex_real_model_yields_pid_prefixed_mcp_form() -> None:
    try:
        from deep_worker.providers.codex_provider import CodexChatModel
    except Exception:  # noqa: BLE001
        pytest.skip("codex_provider not importable")
    model = CodexChatModel()
    assert _provider_of(model) == "codex"
    prompt = _prompt_for(model)
    # PID-carrying server name: mcp__deepworker_<pid>__<tool>.
    assert f"mcp__deepworker_{os.getpid()}__send_message_to_agent" in prompt


def test_grok_real_model_has_mcp_note_without_static_prefix() -> None:
    try:
        from deep_worker.providers.grok_provider import GrokChatModel
    except Exception:  # noqa: BLE001
        pytest.skip("grok_provider not importable")
    model = GrokChatModel()
    assert _provider_of(model) == "grok"
    prompt = _prompt_for(model)
    assert "mcp__deepworker__send_message_to_agent" not in prompt
    lowered = prompt.lower()
    assert "mcp" in lowered
    assert "disk" in lowered or "mcps/" in prompt
