from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def _write_history_jsonl(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 4,
        "kind": "deepagents.summarization_history.message",
        "summarized_at": "2026-01-01T00:00:00",
        "message_index": 0,
        "message_type": "human",
        "message": {"data": {"content": content}},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reindex_offline_rebuilds_index_from_jsonl(monkeypatch, tmp_path, embed_model) -> None:
    from deep_worker_cli.cli import agent_command
    from semantic_search.history_indexer import index_all_history_files
    from semantic_search.history_indexer import history_index_exists

    summaries_dir = tmp_path / Path("~/.deep_worker").name / "ag" / "summaries"
    _write_history_jsonl(summaries_dir / "thread_1.jsonl", "Remember Paris.")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        agent_command,
        "index_all_history_files",
        lambda path, on_progress=None, on_message_progress=None,
        on_analyze_progress=None: index_all_history_files(
            path, embed_model=embed_model, on_progress=on_progress,
            on_message_progress=on_message_progress,
            on_analyze_progress=on_analyze_progress,
        ),
    )

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert history_index_exists(summaries_dir)
    assert (summaries_dir / ".history_index" / "line_hashes.json").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reindex_offline_fixes_stale_hash_desync(monkeypatch, tmp_path, embed_model) -> None:
    from deep_worker_cli.cli import agent_command
    from semantic_search.history_indexer import index_all_history_files
    from semantic_search.history_indexer import history_index_exists

    summaries_dir = tmp_path / Path("~/.deep_worker").name / "ag" / "summaries"
    _write_history_jsonl(summaries_dir / "thread_1.jsonl", "Remember Berlin.")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        agent_command,
        "index_all_history_files",
        lambda path, on_progress=None, on_message_progress=None,
        on_analyze_progress=None: index_all_history_files(
            path, embed_model=embed_model, on_progress=on_progress,
            on_message_progress=on_message_progress,
            on_analyze_progress=on_analyze_progress,
        ),
    )

    assert await agent_command.reindex_history_command("ag") == 0
    lancedb_dir = summaries_dir / ".history_index" / "lancedb"
    assert lancedb_dir.exists()
    shutil.rmtree(lancedb_dir)
    assert (summaries_dir / ".history_index" / "line_hashes.json").exists()
    assert not history_index_exists(summaries_dir)

    assert await agent_command.reindex_history_command("ag") == 0
    assert history_index_exists(summaries_dir)
