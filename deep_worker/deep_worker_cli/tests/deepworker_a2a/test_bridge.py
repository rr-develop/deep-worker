"""Unit tests for ``deepworker_a2a.bridge.TurnEventBridge`` (RC-2)."""

import pytest

from deepworker_a2a.bridge import TurnEventBridge


class _RecordingUpdater:
    def __init__(self):
        self.task_id = "task-1"
        self.calls = []

    async def add_artifact(self, parts, artifact_id=None, name=None,
                           append=None, last_chunk=None, **kw):
        text = parts[0].text if parts else ""
        self.calls.append({"text": text, "append": append})


@pytest.mark.asyncio
async def test_on_event_is_noop_when_not_streaming():
    up = _RecordingUpdater()
    b = TurnEventBridge(up, stream=False)
    await b.on_event({"type": "assistant", "content": "hi"})
    assert up.calls == []
    assert b.saw_event is True


@pytest.mark.asyncio
async def test_first_chunk_append_false_then_true():
    up = _RecordingUpdater()
    b = TurnEventBridge(up, stream=True)
    await b.on_event({"type": "assistant", "content": "Hello "})
    await b.on_event({"type": "assistant", "content": "world"})
    assert [c["append"] for c in up.calls] == [False, True]
    assert [c["text"] for c in up.calls] == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_tool_events_folded_into_prefixed_text():
    up = _RecordingUpdater()
    b = TurnEventBridge(up, stream=True)
    await b.on_event({"type": "tool_call", "name": "search"})
    await b.on_event({"type": "tool_result", "name": "search", "content": "ok"})
    assert up.calls[0]["text"] == "[tool:search]"
    assert up.calls[1]["text"] == "[tool_result:search] ok"


@pytest.mark.asyncio
async def test_non_output_events_emit_nothing():
    up = _RecordingUpdater()
    b = TurnEventBridge(up, stream=True)
    await b.on_event({"type": "user", "content": "ignored"})
    await b.on_event({"type": "agent_interrupted", "content": "x"})
    assert up.calls == []
