"""Unit tests: the embedding model is loaded STRICTLY offline.

Incident & design:
    semantic_search/docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md

Invariant under test
--------------------
The single ``HuggingFaceEmbedding(...)`` construction in
``semantic_search.retriever.search`` MUST force local-only model resolution by
passing ``model_kwargs={"local_files_only": True}``. This makes the
``project_memory`` tool never hit the network: ``huggingface_hub`` performs no
HEAD/metadata/download request, so a transient HF Hub outage can never hang an
agent turn.

These tests patch the *boundary* (the ``HuggingFaceEmbedding`` symbol imported
inside ``search``) and assert on the kwargs it receives — they do NOT mock the
unit under test (``search``/the retriever logic). The cached-embedder branch is
exercised by clearing the process-wide ``_embedding_cache`` first.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_index.core.embeddings.mock_embed_model import MockEmbedding

import semantic_search.retriever as retriever
from semantic_search.retriever import search


def _make_indexed_repo(tmp_path: Path) -> Path:
    """Create the minimal on-disk shape ``search`` needs to get past the
    ``IndexNotFoundError`` gate (a ``.vector_index`` directory)."""
    (tmp_path / ".vector_index").mkdir()
    return tmp_path


def _patched_search(tmp_path: Path, embed_spy: MagicMock) -> MagicMock:
    """Run ``search`` through real retriever code with only the heavy
    externals (vector store / index / retriever / docstore) patched out, and
    the ``HuggingFaceEmbedding`` symbol replaced by ``embed_spy``.

    Returns the ``embed_spy`` (constructor) for kwargs assertions.
    """
    _make_indexed_repo(tmp_path)

    # Force the embedder-construction branch: empty the process-wide cache so
    # ``search`` does NOT reuse a previously built embedder.
    retriever._embedding_cache.clear()
    retriever._index_cache.clear()

    node_with_score = MagicMock()
    inner = MagicMock()
    inner.get_content.return_value = "def hello(): pass"
    inner.metadata = {"file_path": str(tmp_path / "main.py"), "language": "python"}
    inner.relationships = {}
    node_with_score.node = inner
    node_with_score.score = 0.9

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [node_with_score]

    # ``HuggingFaceEmbedding`` is imported INSIDE ``search`` (lazy import), so we
    # patch it at its source module, not on ``retriever``.
    with patch(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding", embed_spy
    ), \
         patch.object(retriever, "_build_retriever", return_value=mock_retriever), \
         patch.object(retriever, "StorageContext"), \
         patch.object(retriever, "VectorStoreIndex"), \
         patch.object(retriever, "LanceDBVectorStore"), \
         patch.object(retriever, "SimpleDocumentStore"):
        search("hello function", repo_path=tmp_path, top_k=1)
    return embed_spy


class TestOfflineEmbedderConstruction:
    def test_embedder_built_with_local_files_only(self, tmp_path: Path) -> None:
        """The embedder MUST be constructed with
        ``model_kwargs={"local_files_only": True}`` — the scoped offline switch."""
        embed_spy = MagicMock(return_value=MockEmbedding(embed_dim=16))
        _patched_search(tmp_path, embed_spy)

        assert embed_spy.call_count == 1, "embedder should be built exactly once"
        _, kwargs = embed_spy.call_args
        assert "model_kwargs" in kwargs, "must pass model_kwargs"
        assert kwargs["model_kwargs"].get("local_files_only") is True, (
            "model_kwargs must force local_files_only=True so project_memory "
            "never hits the network"
        )

    def test_offline_switch_is_scoped_not_global_env(self, tmp_path: Path) -> None:
        """The fix must NOT set any global HF offline env var — the offline
        behaviour is scoped to our embedder instance only (user constraint)."""
        import os

        before = {
            k: os.environ.get(k)
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        embed_spy = MagicMock(return_value=MockEmbedding(embed_dim=16))
        _patched_search(tmp_path, embed_spy)
        after = {
            k: os.environ.get(k)
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        assert before == after, (
            "fix must not mutate global HF offline env vars — offline must be "
            "scoped via model_kwargs, not process-global"
        )

    def test_no_explicit_cache_folder_relies_on_default(self, tmp_path: Path) -> None:
        """Per the resolved design we do NOT pin cache_folder; the llama_index
        default is used (verified to find the model with local_files_only alone).
        Guard against a future refactor silently pinning a wrong directory."""
        embed_spy = MagicMock(return_value=MockEmbedding(embed_dim=16))
        _patched_search(tmp_path, embed_spy)
        _, kwargs = embed_spy.call_args
        assert "cache_folder" not in kwargs, (
            "design decision: do not pin cache_folder; rely on llama_index default"
        )
