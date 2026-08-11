"""Unit tests for the GrokChatModel resolved-model callback path.

Confirms that:
- _maybe_report_model() pulls modelId out of session/prompt result._meta and
  invokes the registered callback (sync or async).
- _record_resolved_model deduplicates: the callback fires once per distinct
  modelId, NOT on every turn.
- The model name surfaced by the callback equals what arrived in _meta.

This is the in-process counterpart of the on-disk extractor tested in
deep_worker_cli/tests/unit_tests/process/test_grok_persisted_model_display.
Together they pin both halves of the "no more 'unknown' in TUI" fix.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider.chat_model import GrokChatModel  # noqa: E402


pytestmark = pytest.mark.asyncio


async def test_report_model_invokes_sync_callback() -> None:
    m = GrokChatModel()
    seen: list[str] = []
    m.set_resolved_model_callback(lambda name: seen.append(name))
    await m._maybe_report_model(
        {"stopReason": "end_turn", "_meta": {"modelId": "grok-composer-2.5-fast"}}
    )
    assert seen == ["grok-composer-2.5-fast"]
    assert m.resolved_model_name == "grok-composer-2.5-fast"


async def test_report_model_invokes_async_callback() -> None:
    m = GrokChatModel()
    seen: list[str] = []

    async def cb(name: str) -> None:
        seen.append(name)

    m.set_resolved_model_callback(cb)
    await m._maybe_report_model({"_meta": {"modelId": "grok-4-fast"}})
    assert seen == ["grok-4-fast"]


async def test_report_model_deduplicates() -> None:
    """Same modelId on two consecutive turns must NOT re-fire the callback —
    that would spam the TUI / session_metadata.json writer."""
    m = GrokChatModel()
    seen: list[str] = []
    m.set_resolved_model_callback(lambda name: seen.append(name))
    payload = {"_meta": {"modelId": "grok-composer-2.5-fast"}}
    await m._maybe_report_model(payload)
    await m._maybe_report_model(payload)
    await m._maybe_report_model(payload)
    assert seen == ["grok-composer-2.5-fast"]


async def test_report_model_fires_again_on_change() -> None:
    """If grok upgrades the model mid-session (rare), the new id MUST fire."""
    m = GrokChatModel()
    seen: list[str] = []
    m.set_resolved_model_callback(lambda name: seen.append(name))
    await m._maybe_report_model({"_meta": {"modelId": "grok-composer-2.5-fast"}})
    await m._maybe_report_model({"_meta": {"modelId": "grok-composer-2.5-final"}})
    assert seen == ["grok-composer-2.5-fast", "grok-composer-2.5-final"]


async def test_report_model_noop_without_meta() -> None:
    m = GrokChatModel()
    seen: list[str] = []
    m.set_resolved_model_callback(lambda name: seen.append(name))
    await m._maybe_report_model({"stopReason": "end_turn"})
    await m._maybe_report_model({"_meta": {}})
    await m._maybe_report_model({"_meta": {"modelId": 42}})  # non-string
    assert seen == []
    assert m.resolved_model_name is None


async def test_resolved_model_name_property_before_first_report() -> None:
    """Fresh model — resolved_model_name is None until something reports."""
    m = GrokChatModel()
    assert m.resolved_model_name is None


async def test_resolved_model_name_seeded_from_constructor_when_explicit() -> None:
    """If the user passed an explicit --model, resolved_model_name starts
    out as that name so the TUI never shows 'unknown' even before turn 1."""
    m = GrokChatModel(model="grok-2.5-explicit")
    assert m.resolved_model_name == "grok-2.5-explicit"
