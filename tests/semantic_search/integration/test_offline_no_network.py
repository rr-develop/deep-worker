"""Integration test (NO mocks): project_memory's engine never touches the network.

Incident & design:
    semantic_search/docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md

What this proves
----------------
With a real, locally-cached embedding model and a real on-disk lance index,
``search()`` (the engine behind the agents' ``project_memory`` tool) completes
successfully while ALL outbound network connections are hard-blocked at the
socket layer. This is a behavioural proof ("zero network"), not a check of a
particular flag: we install a real network kill-switch (``socket.socket.connect``
raises on any outbound connect) and verify the search still returns results.

Crucially, the process-wide embedder cache is cleared *before* the blocked
section, so the embedder is (re)constructed UNDER the kill-switch — exactly the
load path that hung agent ``test_v3`` when HF Hub was unreachable. If the fix
(``model_kwargs={"local_files_only": True}``) regressed, building the embedder
would attempt a network HEAD/metadata/download and this test would fail with the
kill-switch's RuntimeError instead of hanging.

No mocks of the unit under test: the embedder, SentenceTransformer, lancedb and
the retriever are all real. Only the OS network boundary is patched.
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("llama_index")  # semantic-search extra; absent in CI

import deep_worker.providers.semantic_search.retriever as retriever
from deep_worker.providers.semantic_search import _embed_cache
from deep_worker.providers.semantic_search.indexer import index_repository
from deep_worker.providers.semantic_search.retriever import search

pytestmark = pytest.mark.integration


def _model_cached() -> bool:
    """Same guard the existing no-leak test uses: only run when the real model
    is already present locally (this test must never download anything)."""
    hf = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    li = Path(os.path.expanduser("~/.cache/llama_index"))
    return hf.exists() or li.exists()


@pytest.fixture(scope="module")
def indexed_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real git repo, indexed once with the REAL embedder.

    Indexing happens BEFORE the network is blocked — it is the legitimate
    warm-up. The kill-switch is applied only around ``search`` in the test body.
    """
    if not _model_cached():
        pytest.skip(
            "Real model cache not present — this test needs "
            "intfloat/multilingual-e5-small already downloaded."
        )

    repo = tmp_path_factory.mktemp("offline_repo")
    (repo / "auth.py").write_text(
        "def authenticate(user, password):\n"
        '    """Authenticate a user against the database."""\n'
        "    return verify(user, password)\n\n"
        "def verify(u, p):\n"
        "    return True\n"
    )
    (repo / "README.md").write_text(
        "# Tiny Auth Service\n\nAuthenticate users and hash passwords.\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@x", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    index_repository(repo)
    return repo


def test_search_succeeds_with_network_hard_blocked(
    indexed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """search() returns results while EVERY outbound socket connect raises.

    Clearing the embedder cache forces the embedder to be rebuilt *under* the
    kill-switch, exercising the exact model-load path from the incident.
    """
    attempted: list[object] = []
    real_connect = socket.socket.connect

    def blocked_connect(self: socket.socket, address: object) -> None:  # type: ignore[override]
        # Allow AF_UNIX / loopback (local IPC) but block any real outbound
        # network connection — that is what a HF Hub call would use.
        try:
            host = address[0] if isinstance(address, tuple) else None
        except Exception:  # noqa: BLE001
            host = None
        if host not in (None, "127.0.0.1", "::1", "localhost"):
            attempted.append(address)
            raise RuntimeError(f"NETWORK BLOCKED: outbound connect to {address!r}")
        return real_connect(self, address)  # type: ignore[arg-type]

    # Force a cold embedder build under the kill-switch. The embedder cache
    # lives in the shared resolver (``_embed_cache``) after the §7.1 hoist;
    # ``retriever._index_cache`` is a separate per-repo LanceDB cache.
    _embed_cache._reset_cache_for_tests()
    retriever._index_cache.clear()

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    results = search("how is the user authenticated", repo_path=indexed_repo, top_k=2)

    # Behavioural proof: we got real results AND no outbound connect happened.
    assert isinstance(results, list)
    assert len(results) >= 1, "offline search must still return results"
    assert results[0].file_path.endswith("auth.py")
    assert attempted == [], (
        f"project_memory engine attempted outbound network connections: {attempted}"
    )
