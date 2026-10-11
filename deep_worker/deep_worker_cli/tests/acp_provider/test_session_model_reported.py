"""The claude-code session reports the model it actually runs.

The ACP adapter never says which model it is running: the whole
``initialize`` → ``session/new`` → ``session/prompt`` exchange never
mentions one, and ``session/new``'s picker reports ``currentValue:
"default"`` — a selector, not a name. So the base ``_maybe_report_model``
(which reads ``_meta.modelId``) is a no-op for this provider:
``_resolved_model_name`` stayed empty, token stats recorded ``""`` and
the UI showed ``unknown``. Agents launched with ``-m opus`` reported the
ALIAS ``opus`` while the session actually ran ``claude-opus-5``.

The CLI writes its own transcript for the session, and every assistant
record carries the model the API answered with. These tests cover the
reader and — separately — the WIRING that calls it at the turn boundary.
Both axes are mutated: a test here fails if the reader breaks, AND a test
here fails if the call is deleted from ``_finalize_response``.

No test reads the real ``~/.claude`` or ``~/.deep_worker``: every one
points the reader at a tmp_path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from deep_worker.providers.claude_code_provider import chat_model as cc_chat
from deep_worker.providers.claude_code_provider import profile as prof

TEST_TIMEOUT = 10.0


def write_transcript(
    root: Path, session_id: str, records: list[dict[str, Any]],
    *, project: str = "-repos-somewhere",
) -> Path:
    """Lay out a transcript the way the CLI does: <project>/<id>.jsonl."""
    d = root / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def assistant(model: str | None, **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": []}
    if model is not None:
        message["model"] = model
    message.update(extra)
    return {"type": "assistant", "message": message}


# ---------------------------------------------------------------------------
# Axis 1: reading the transcript
# ---------------------------------------------------------------------------
class TestReadSessionModel:
    def test_reads_the_model_the_api_answered_with(self, tmp_path: Path) -> None:
        write_transcript(tmp_path, "sid-1", [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            assistant("claude-opus-5"),
        ])
        assert prof.read_session_model("sid-1", root=tmp_path) == "claude-opus-5"

    def test_value_is_verbatim_not_a_constant(self, tmp_path: Path) -> None:
        """Whatever the transcript says is what we report.

        An implementation that hardcodes a name passes the test above and
        fails this one.
        """
        write_transcript(tmp_path, "sid-2", [assistant("some-unreleased-xyz")])
        assert prof.read_session_model("sid-2", root=tmp_path) == "some-unreleased-xyz"

    def test_last_model_wins_when_a_session_changed_models(
        self, tmp_path: Path
    ) -> None:
        """A resumed session's early turns can name an older model; the
        current one is what the UI must show."""
        write_transcript(tmp_path, "sid-3", [
            assistant("claude-sonnet-5"),
            assistant("claude-opus-5"),
        ])
        assert prof.read_session_model("sid-3", root=tmp_path) == "claude-opus-5"

    def test_synthetic_records_name_no_model(self, tmp_path: Path) -> None:
        """'<synthetic>' marks a locally generated message — an error stub
        or a cancellation. The API never answered, so it is not a model."""
        write_transcript(tmp_path, "sid-4", [
            assistant("claude-opus-5"),
            assistant("<synthetic>"),
        ])
        assert prof.read_session_model("sid-4", root=tmp_path) == "claude-opus-5"

    def test_transcript_with_no_reply_yet_reports_nothing(
        self, tmp_path: Path
    ) -> None:
        """Before the first assistant reply the answer does not exist.

        'I don't know' must stay unknown rather than become a guess.
        """
        write_transcript(tmp_path, "sid-5", [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ])
        assert prof.read_session_model("sid-5", root=tmp_path) is None

    def test_missing_transcript_reports_nothing(self, tmp_path: Path) -> None:
        assert prof.read_session_model("no-such-session", root=tmp_path) is None

    def test_empty_session_id_reports_nothing(self, tmp_path: Path) -> None:
        assert prof.read_session_model("", root=tmp_path) is None

    def test_empty_session_id_does_not_scan_the_disk(
        self, tmp_path: Path
    ) -> None:
        """No id, no question to ask.

        ``glob("*/.jsonl")`` would match nothing anyway, so the guard is
        invisible in the return value — but it saves walking every project
        directory on a path that runs at every turn boundary.
        """
        with patch.object(
            Path, "glob", side_effect=AssertionError("scanned the disk")
        ):
            assert prof.read_session_model("", root=tmp_path) is None

    def test_finds_the_file_under_any_project_directory(
        self, tmp_path: Path
    ) -> None:
        """The directory is a mangling of cwd with corners of its own
        (``.deep_worker`` becomes ``--deep-worker``). We locate the file by
        the unique session id instead of rebuilding that convention."""
        write_transcript(
            tmp_path, "sid-6", [assistant("claude-opus-5")],
            project="-home-user--deep-worker-agents-x-workspace",
        )
        assert prof.read_session_model("sid-6", root=tmp_path) == "claude-opus-5"

    @pytest.mark.parametrize("model", [123, None, "", {"a": 1}])
    def test_non_string_model_is_ignored(
        self, tmp_path: Path, model: Any
    ) -> None:
        write_transcript(tmp_path, "sid-7", [assistant(model)])
        assert prof.read_session_model("sid-7", root=tmp_path) is None

    def test_corrupt_lines_are_skipped(self, tmp_path: Path) -> None:
        """A transcript being appended to can end mid-line."""
        d = tmp_path / "-p"
        d.mkdir(parents=True)
        (d / "sid-8.jsonl").write_text(
            "not json at all\n"
            + json.dumps(assistant("claude-opus-5")) + "\n"
            + '{"type": "assistant", "message": {"role": "assist'
        )
        assert prof.read_session_model("sid-8", root=tmp_path) == "claude-opus-5"

    def test_unreadable_transcript_reports_nothing_and_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_transcript(tmp_path, "sid-9", [assistant("claude-opus-5")])
        with caplog.at_level("WARNING"), \
                patch.object(Path, "open", side_effect=OSError("boom")):
            got = prof.read_session_model("sid-9", root=tmp_path)
        assert got is None
        # A silent swallow would leave a permanent 'unknown' in the UI
        # with nothing in the log to explain it.
        assert "OSError" in caplog.text


# ---------------------------------------------------------------------------
# Axis 2: the wiring — the reader is actually CALLED at the turn boundary
# ---------------------------------------------------------------------------
class _State:
    def __init__(self, session_id: str | None) -> None:
        self.session_id = session_id


class _Model(cc_chat.ClaudeCodeAcpChatModel):
    """Real class, cheap construction (the base __init__ is heavy)."""

    def __init__(self) -> None:
        self._resolved_model_name = None
        self._recorded: list[str] = []

    async def _record_resolved_model(self, model: str) -> None:
        self._recorded.append(model)
        self._resolved_model_name = model


@pytest.fixture
def _base_finalize():
    """Neutralise the base turn-end path; we only test our addition."""
    async def _noop(self, response, text_acc, state=None):  # noqa: ANN001
        return "MESSAGE"

    with patch.object(cc_chat.AcpChatModel, "_finalize_response", _noop):
        yield


class TestTurnEndReportsTheModel:
    @pytest.mark.asyncio
    async def test_finishing_a_turn_reports_the_transcript_model(
        self, _base_finalize: None
    ) -> None:
        """THE WIRING TEST.

        Drives the real ``_finalize_response``. Deleting the
        ``_report_transcript_model(state)`` call from it makes this fail —
        the reader would still work in isolation, which is exactly the
        blind spot this test exists to close.
        """
        seen: list[str] = []

        def _read(session_id: str) -> str:
            seen.append(session_id)
            return "claude-opus-5"

        m = _Model()
        with patch.object(cc_chat, "read_session_model", _read):
            out = await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid-live")), TEST_TIMEOUT
            )

        assert seen == ["sid-live"], "the turn must consult the transcript"
        assert m._recorded == ["claude-opus-5"]
        assert out == "MESSAGE", "the base result must be returned unchanged"

    @pytest.mark.asyncio
    async def test_no_session_id_reports_nothing(
        self, _base_finalize: None
    ) -> None:
        called: list[str] = []

        def _read(session_id: str) -> str:
            called.append(session_id)
            return "claude-opus-5"

        m = _Model()
        with patch.object(cc_chat, "read_session_model", _read):
            await asyncio.wait_for(
                m._finalize_response({}, [], _State(None)), TEST_TIMEOUT
            )

        assert called == []
        assert m._recorded == []

    @pytest.mark.asyncio
    async def test_unchanged_model_is_not_re_reported(
        self, _base_finalize: None
    ) -> None:
        """Every turn reads the file, but only a CHANGE is worth recording."""
        m = _Model()
        with patch.object(cc_chat, "read_session_model", lambda s: "claude-opus-5"):
            for _ in range(3):
                await asyncio.wait_for(
                    m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
                )

        assert m._recorded == ["claude-opus-5"]

    @pytest.mark.asyncio
    async def test_a_model_change_mid_session_is_reported(
        self, _base_finalize: None
    ) -> None:
        answers = iter(["claude-sonnet-5", "claude-opus-5"])
        m = _Model()
        with patch.object(cc_chat, "read_session_model", lambda s: next(answers)):
            await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )
            await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )

        assert m._recorded == ["claude-sonnet-5", "claude-opus-5"]

    @pytest.mark.asyncio
    async def test_unreported_model_leaves_the_name_unset(
        self, _base_finalize: None
    ) -> None:
        m = _Model()
        with patch.object(cc_chat, "read_session_model", lambda s: None):
            await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )

        assert m._recorded == []


class TestReportingNeverBreaksTheTurn:
    @pytest.mark.asyncio
    async def test_reader_exception_does_not_lose_the_turn(
        self, _base_finalize: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Losing a whole turn over a cosmetic field would be a far worse
        bug than the one being fixed."""
        def _boom(session_id: str) -> str:
            raise RuntimeError("reader blew up")

        m = _Model()
        with caplog.at_level("WARNING"), \
                patch.object(cc_chat, "read_session_model", _boom):
            out = await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )

        assert out == "MESSAGE"
        assert "reader blew up" in caplog.text

    @pytest.mark.asyncio
    async def test_record_channel_failure_does_not_lose_the_turn(
        self, _base_finalize: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _boom(model: str) -> None:
            raise RuntimeError("record blew up")

        m = _Model()
        m._record_resolved_model = _boom  # type: ignore[assignment]
        with caplog.at_level("WARNING"), \
                patch.object(cc_chat, "read_session_model", lambda s: "claude-opus-5"):
            out = await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )

        assert out == "MESSAGE"
        assert "record blew up" in caplog.text

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(
        self, _base_finalize: None
    ) -> None:
        def _cancel(session_id: str) -> str:
            raise asyncio.CancelledError()

        m = _Model()
        with patch.object(cc_chat, "read_session_model", _cancel):
            with pytest.raises(asyncio.CancelledError):
                await m._finalize_response({}, [], _State("sid"))


class TestNoProcessIsSpawned:
    """The owner's rule: never spawn a process just to learn the model.

    A subprocess answers for a DIFFERENT process than the live session, so
    it can disagree with it — and it costs seconds. The data is already on
    disk, written by the session we are running.
    """

    def test_reading_the_model_spawns_nothing(self, tmp_path: Path) -> None:
        write_transcript(tmp_path, "sid", [assistant("claude-opus-5")])
        import subprocess

        with patch.object(
            subprocess, "Popen", side_effect=AssertionError("spawned a process")
        ), patch.object(
            subprocess, "run", side_effect=AssertionError("spawned a process")
        ), patch.object(
            asyncio, "create_subprocess_exec",
            side_effect=AssertionError("spawned a process"),
        ):
            assert prof.read_session_model("sid", root=tmp_path) == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_the_turn_boundary_spawns_nothing(
        self, _base_finalize: None, tmp_path: Path
    ) -> None:
        write_transcript(tmp_path, "sid", [assistant("claude-opus-5")])
        import subprocess

        m = _Model()
        with patch.object(prof, "SESSION_TRANSCRIPT_ROOT", tmp_path), \
            patch.object(
                subprocess, "Popen",
                side_effect=AssertionError("spawned a process"),
            ), patch.object(
                asyncio, "create_subprocess_exec",
                side_effect=AssertionError("spawned a process"),
            ):
            await asyncio.wait_for(
                m._finalize_response({}, [], _State("sid")), TEST_TIMEOUT
            )

        assert m._recorded == ["claude-opus-5"]
