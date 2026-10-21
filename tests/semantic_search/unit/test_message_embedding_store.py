"""Unit tests for the incremental per-message embedding store.

Uses a lightweight fake embedder (no real model download / no torch-heavy
forward pass beyond tiny arrays) that mimics the parts
``resolve_sentence_transformer`` / ``embed_model_tokenizer`` rely on:
``_model.tokenizer`` (callable, ``model_max_length``,
``num_special_tokens_to_add``, ``decode``) and ``_model.encode`` /
``get_sentence_embedding_dimension``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deep_worker.providers.semantic_search.message_embedding_store import (
    STORE_DIR_NAME,
    MessageFlag,
    build_message_embeddings,
    build_message_metadata,
    classify_message,
    load_message_embeddings,
    load_message_metadata,
    message_embeddings_exist,
    message_metadata_exist,
    store_dir,
    store_file_entry_counts,
)

_DIM = 8


class _FakeTokenizer:
    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # one fake token id per whitespace word
        return {"input_ids": [(abs(hash(w)) % 1000) for w in text.split()]}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def decode(self, ids: list[int]) -> str:
        # deterministic placeholder text whose word-count == len(ids)
        return " ".join(f"t{i}" for i in ids)


class _FakeST:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return _DIM

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            out[i, h % _DIM] = 1.0  # unit vector along one axis
        return out


class _FakeEmbed:
    """Stands in for HuggingFaceEmbedding (exposes ._model)."""

    def __init__(self) -> None:
        self._model = _FakeST()


def _write_jsonl(path: Path, contents: list[str], role: str = "human") -> None:
    lines = []
    for c in contents:
        lines.append(json.dumps({
            "summarized_at": "2026-01-01T00:00:00",
            "message_type": role,
            "message": {"data": {"content": c}},
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@pytest.fixture
def fake_embed() -> _FakeEmbed:
    return _FakeEmbed()


class TestBuildAndLoad:
    def test_build_creates_store_files(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["hello world", "second message here"])

        stats = build_message_embeddings(summaries, embed_model=fake_embed)
        assert stats.total_messages == 2
        assert stats.new_messages == 2
        assert stats.rebuilt is True
        assert message_embeddings_exist(summaries)

        d = store_dir(summaries)
        assert (d / "embeddings.npy").exists()
        assert (d / "token_lengths.npy").exists()
        assert (d / "manifest.json").exists()
        # no stray temp files
        assert not list(d.glob("*.tmp*"))

    def test_load_shapes_and_norms(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["alpha beta", "gamma delta epsilon"])
        build_message_embeddings(summaries, embed_model=fake_embed)

        embs, lens = load_message_embeddings(summaries, as_torch=False)
        assert embs.shape == (2, _DIM)
        assert lens.shape == (2,)
        assert embs.dtype == np.float32
        assert lens.dtype == np.int64
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        # token lengths == word counts ("User: " prefix adds 1)
        assert int(lens[0]) == 3  # "User:" + "alpha" + "beta"
        assert int(lens[1]) == 4

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        with pytest.raises(FileNotFoundError):
            load_message_embeddings(summaries)


class TestIncremental:
    def test_append_only_extends(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write_jsonl(f, ["one two", "three four"])
        s1 = build_message_embeddings(summaries, embed_model=fake_embed)
        assert s1.total_messages == 2

        embs1, _ = load_message_embeddings(summaries, as_torch=False)

        # Append two more messages
        _write_jsonl(f, ["one two", "three four", "five six", "seven eight"])
        s2 = build_message_embeddings(summaries, embed_model=fake_embed)
        assert s2.new_messages == 2
        assert s2.total_messages == 4
        assert s2.rebuilt is False

        embs2, lens2 = load_message_embeddings(summaries, as_torch=False)
        assert embs2.shape == (4, _DIM)
        # first two rows are byte-identical (append-only stability)
        assert np.array_equal(embs2[:2], embs1)

    def test_noop_when_unchanged(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["x y z"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        s2 = build_message_embeddings(summaries, embed_model=fake_embed)
        assert s2.new_messages == 0
        assert s2.total_messages == 1
        assert s2.rebuilt is False

    def test_reset_rebuilds(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["aa bb"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        s = build_message_embeddings(summaries, embed_model=fake_embed, reset=True)
        assert s.rebuilt is True
        assert s.total_messages == 1

    def test_shrunk_file_triggers_rebuild(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write_jsonl(f, ["m1 a", "m2 b", "m3 c"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        # Rewrite (shrink) the file
        _write_jsonl(f, ["only one"])
        s = build_message_embeddings(summaries, embed_model=fake_embed)
        assert s.rebuilt is True
        assert s.total_messages == 1

    def test_model_change_triggers_rebuild(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["p q"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        # Tamper manifest's model name -> next build must rebuild
        d = store_dir(summaries)
        man = json.loads((d / "manifest.json").read_text())
        man["embedding_model"] = "some/other-model"
        (d / "manifest.json").write_text(json.dumps(man))
        s = build_message_embeddings(summaries, embed_model=fake_embed)
        assert s.rebuilt is True


class TestStoreDir:
    def test_store_dir_name(self, tmp_path: Path) -> None:
        assert store_dir(tmp_path).name == STORE_DIR_NAME


def _entry(role: str, content: Any, tool_calls: Any = None) -> dict[str, Any]:
    data: dict[str, Any] = {"content": content}
    if tool_calls is not None:
        data["additional_kwargs"] = {"tool_calls": tool_calls}
    return {"summarized_at": "t", "message_type": role, "message": {"data": data}}


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


class TestClassifyMessage:
    def test_plain_user(self) -> None:
        f = classify_message(_entry("human", "just a question"))
        assert f & MessageFlag.USER_ROLE
        assert f & MessageFlag.PLAIN_USER
        assert not (f & MessageFlag.TASK_MANAGER)

    def test_task_manager(self) -> None:
        f = classify_message(_entry("human", "TaskManager:> do the thing"))
        assert f & MessageFlag.USER_ROLE
        assert f & MessageFlag.TASK_MANAGER
        assert not (f & MessageFlag.PLAIN_USER)

    def test_agent_to_agent_human_transport(self) -> None:
        f = classify_message(_entry("human", "agent_1 -> agent_2 :> hi"))
        assert f & MessageFlag.USER_ROLE
        assert f & MessageFlag.AGENT_TO_AGENT
        assert not (f & MessageFlag.PLAIN_USER)

    def test_system_role(self) -> None:
        f = classify_message(_entry("system", "NOTICE: restarted"))
        assert f & MessageFlag.SYSTEM
        assert not (f & MessageFlag.USER_ROLE)

    def test_system_prefixed_human(self) -> None:
        f = classify_message(_entry("human", "System:> something"))
        assert f & MessageFlag.SYSTEM
        assert f & MessageFlag.USER_ROLE
        assert not (f & MessageFlag.PLAIN_USER)

    def test_tool_result(self) -> None:
        f = classify_message(_entry("tool", "result text"))
        assert f & MessageFlag.TOOL_RESULT
        assert not (f & MessageFlag.AGENT)

    def test_agent_plain(self) -> None:
        f = classify_message(_entry("ai", "let me think"))
        assert f & MessageFlag.AGENT
        assert not (f & MessageFlag.TOOL_CALL)

    def test_agent_with_tool_call_does_not_set_tool_call_bit(self) -> None:
        # An ai comment that carries a tool call is AGENT, but the embedding is of
        # the COMMENT, not of the call — TOOL_CALL (reserved) must stay 0.
        f = classify_message(
            _entry("ai", "calling a tool", tool_calls=[{"name": "x", "args": {}}])
        )
        assert f & MessageFlag.AGENT
        assert not (f & MessageFlag.TOOL_CALL)

    def test_agent_tool_use_block_does_not_set_tool_call_bit(self) -> None:
        content = [{"type": "text", "text": "do"}, {"type": "tool_use", "name": "g"}]
        f = classify_message(_entry("ai", content))
        assert f & MessageFlag.AGENT
        assert not (f & MessageFlag.TOOL_CALL)

    def test_summarization_heading(self) -> None:
        f = classify_message(_entry("ai", "## SESSION INTENT\nDo the thing."))
        assert f & MessageFlag.AGENT
        assert f & MessageFlag.SUMMARIZATION

    def test_non_summary_agent_heading_not_flagged(self) -> None:
        f = classify_message(_entry("ai", "## Status\nall good"))
        assert f & MessageFlag.AGENT
        assert not (f & MessageFlag.SUMMARIZATION)

    def test_agent_final_when_next_is_human(self) -> None:
        f = classify_message(_entry("ai", "done"), next_entry=_entry("human", "thanks"))
        assert f & MessageFlag.AGENT
        assert f & MessageFlag.AGENT_FINAL

    def test_agent_not_final_when_next_is_tool(self) -> None:
        f = classify_message(_entry("ai", "calling"), next_entry=_entry("tool", "out"))
        assert f & MessageFlag.AGENT
        assert not (f & MessageFlag.AGENT_FINAL)

    def test_agent_not_final_without_next(self) -> None:
        f = classify_message(_entry("ai", "last message"))
        assert not (f & MessageFlag.AGENT_FINAL)


class TestBuildMessageMetadata:
    def test_indices_and_flags_align_with_embeddings(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        import numpy as np

        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_entries(summaries / "a.jsonl", [
            _entry("human", "TaskManager:> alpha"),
            _entry("ai", "thinking", tool_calls=[{"name": "t"}]),
            _entry("tool", "tool output"),
            _entry("ai", ""),  # empty -> no embedding, must be skipped
            _entry("human", "plain question"),
        ])
        build_message_embeddings(summaries, embed_model=fake_embed)
        stats = build_message_metadata(summaries)

        # 4 non-empty messages (the empty ai is skipped)
        assert stats.total_messages == 4
        assert message_metadata_exist(summaries)

        idx, flags = load_message_metadata(summaries, as_torch=False)
        embs, _ = load_message_embeddings(summaries, as_torch=False)
        assert idx.shape[0] == embs.shape[0] == 4
        assert idx.dtype == np.int64 and flags.dtype == np.int64
        # contiguous 0..N-1
        assert list(idx) == [0, 1, 2, 3]
        # flag order: TaskManager, agent (tool-call comment, TOOL_CALL stays 0),
        # tool_result, plain_user
        assert flags[0] & MessageFlag.TASK_MANAGER
        assert flags[1] & MessageFlag.AGENT and not (flags[1] & MessageFlag.TOOL_CALL)
        assert flags[2] & MessageFlag.TOOL_RESULT
        assert flags[3] & MessageFlag.PLAIN_USER

    def test_summarization_and_agent_final_in_build(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_entries(summaries / "a.jsonl", [
            _entry("ai", "## SESSION INTENT\nrecap"),  # 0: summary, next=human -> final
            _entry("human", "TaskManager:> go"),       # 1
            _entry("ai", "calling", tool_calls=[{"name": "t"}]),  # 2: next=tool -> not final
            _entry("tool", "tool output"),             # 3
            _entry("ai", "here is the result"),        # 4: next=human -> final
            _entry("human", "thanks"),                 # 5
        ])
        build_message_embeddings(summaries, embed_model=fake_embed)
        stats = build_message_metadata(summaries)
        assert stats.flag_counts["SUMMARIZATION"] == 1
        assert stats.flag_counts["AGENT_FINAL"] == 2

        _, flags = load_message_metadata(summaries, as_torch=False)
        # row order == selection order: [ai-summary, human, ai-toolcall, tool, ai, human]
        assert flags[0] & MessageFlag.SUMMARIZATION
        assert flags[0] & MessageFlag.AGENT_FINAL      # next is human
        assert not (flags[2] & MessageFlag.AGENT_FINAL)  # next is tool
        assert flags[4] & MessageFlag.AGENT_FINAL      # next is human

    def test_agent_final_look_ahead_uses_raw_next(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        # An empty (text-less) ai is not embedded, but it IS the raw next entry
        # after the preceding ai — so that preceding ai is NOT final (its real
        # next message is the empty ai, not a human).
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_entries(summaries / "a.jsonl", [
            _entry("ai", "thinking"),                       # 0: next(raw)=empty ai -> not final
            _entry("ai", "", tool_calls=[{"name": "t"}]),  # 1: empty -> NOT embedded
            _entry("tool", "out"),                          # 2
            _entry("ai", "final answer"),                   # 3: next=human -> final
            _entry("human", "ok"),                          # 4
        ])
        build_message_embeddings(summaries, embed_model=fake_embed)
        build_message_metadata(summaries)
        _, flags = load_message_metadata(summaries, as_torch=False)
        # embedded rows: [ai "thinking", tool "out", ai "final answer", human "ok"]
        assert not (flags[0] & MessageFlag.AGENT_FINAL)  # raw next was the empty ai
        assert flags[2] & MessageFlag.AGENT_FINAL        # raw next is human

    def test_metadata_requires_embeddings(self, tmp_path: Path) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        with pytest.raises(FileNotFoundError):
            build_message_metadata(summaries)

    def test_load_metadata_missing_raises(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_entries(summaries / "a.jsonl", [_entry("human", "hi")])
        build_message_embeddings(summaries, embed_model=fake_embed)
        # embeddings exist but metadata not built yet
        assert not message_metadata_exist(summaries)
        with pytest.raises(FileNotFoundError):
            load_message_metadata(summaries)

    def test_rebuild_embeddings_invalidates_metadata(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write_entries(f, [_entry("human", "one"), _entry("ai", "two")])
        build_message_embeddings(summaries, embed_model=fake_embed)
        build_message_metadata(summaries)
        assert message_metadata_exist(summaries)

        # Append a message -> embeddings change -> metadata must be invalidated
        _write_entries(f, [_entry("human", "one"), _entry("ai", "two"), _entry("human", "three")])
        build_message_embeddings(summaries, embed_model=fake_embed)
        assert not message_metadata_exist(summaries)

        # Rebuild metadata aligns with new row count
        stats = build_message_metadata(summaries)
        assert stats.total_messages == 3
        idx, _ = load_message_metadata(summaries, as_torch=False)
        assert list(idx) == [0, 1, 2]

    def test_metadata_files_are_not_world_accessible(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        import stat

        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_entries(summaries / "a.jsonl", [_entry("human", "hi")])
        build_message_embeddings(summaries, embed_model=fake_embed)
        build_message_metadata(summaries)
        d = store_dir(summaries)
        for name in ("message_indices.npy", "flags.npy"):
            mode = (d / name).stat().st_mode
            assert not (mode & (stat.S_IRWXG | stat.S_IRWXO)), name


class TestPermissions:
    def test_store_files_are_not_world_accessible(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        import stat

        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write_jsonl(summaries / "a.jsonl", ["hello world"])
        build_message_embeddings(summaries, embed_model=fake_embed)

        d = store_dir(summaries)
        for name in ("embeddings.npy", "token_lengths.npy", "manifest.json"):
            mode = (d / name).stat().st_mode
            # no group/other permission bits (policy: no 0666/0777)
            assert not (mode & (stat.S_IRWXG | stat.S_IRWXO)), name
        # no stray temp file left behind
        assert not list(d.glob("*.tmp*"))


class TestFileEntryCounts:
    def test_watermark_records_parsed_entry_count(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write_jsonl(f, ["hello world", "second message here"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        counts = store_file_entry_counts(summaries)
        # Keyed by the path the build recorded (str(glob result)); value is the
        # parsed-entry count (2), the same unit count_jsonl_lines uses.
        assert counts.get(str(f)) == 2

    def test_watermark_frozen_after_append(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write_jsonl(f, ["one", "two"])
        build_message_embeddings(summaries, embed_model=fake_embed)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "summarized_at": "2026-01-02T00:00:00",
                "message_type": "human",
                "message": {"data": {"content": "three"}},
            }, ensure_ascii=False) + "\n")
        # The store was not rebuilt -> watermark still reflects the frozen 2 rows.
        assert store_file_entry_counts(summaries).get(str(f)) == 2

    def test_missing_store_returns_empty(self, tmp_path: Path) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        assert store_file_entry_counts(summaries) == {}
