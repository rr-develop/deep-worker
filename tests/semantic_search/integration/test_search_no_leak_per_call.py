"""Regression test: ``search()`` must not allocate fresh embedding/index
machinery on every call.

Bug (production, deep_worker_web 2026-04-25 incident):

    ``semantic_search.retriever.search`` instantiates a new
    ``HuggingFaceEmbedding(...)`` and a new ``VectorStoreIndex`` /
    ``LanceDBVectorStore`` *every* invocation. The previous instances
    are not freed because:

      * ``LanceDBVectorStore`` keeps a reference to its embed_model.
      * ``lancedb`` spawns a tokio thread pool (visible as
        ``tokio-runtime-w`` threads) that holds C++-side state
        referencing the Python embed_model.
      * ``llama_index.core.Settings.embed_model`` is a global, but the
        previous Python objects survive in the Connection-graph
        spanning lance C++.

    Net effect measured on a real agent (model multilingual-e5-small,
    ~117M params): each call adds one fresh ``mmap`` of the model blob
    (~450 MB VA) and one fresh ~128 MB anonymous arena (torch storage
    of the weight tensor). Real RSS grows ~390 MB per call.

This test exercises the leak with a small repo and 5 repeated searches,
and asserts that RSS growth after warmup stays below a small bound.
The contract is: amortised post-warmup memory cost per search must be
O(retrieval working set), not O(model size).
"""
from __future__ import annotations

import gc
import os
import resource
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("llama_index")  # semantic-search extra; absent in CI

from deep_worker.providers.semantic_search.indexer import index_repository
from deep_worker.providers.semantic_search.retriever import search


def _rss_kb() -> int:
    """Process resident set size in KB.

    On Linux ``ru_maxrss`` is reported in KB. We use ``ru_maxrss``
    because it is monotonic-non-decreasing — exactly what we want for
    'has the process committed more memory since baseline?'.
    Reading /proc/self/status would also work; we pick getrusage for
    portability and to avoid file I/O between samples.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.fixture(scope="module")
def small_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny repo we can index quickly with a real embedder.

    Module-scoped: we want exactly ONE indexing pass for the whole
    test module so the leak signal is dominated by per-search work,
    not per-index work.
    """
    repo = tmp_path_factory.mktemp("leak_repo")
    (repo / "main.py").write_text(
        "def authenticate(user, password):\n"
        "    \"\"\"Authenticate user against database.\"\"\"\n"
        "    return verify(user, password)\n\n"
        "def verify(u, p):\n"
        "    return True\n"
    )
    (repo / "utils.py").write_text(
        "def hash_password(pw: str) -> str:\n"
        "    \"\"\"Hash a password using bcrypt.\"\"\"\n"
        "    import bcrypt\n"
        "    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()\n"
    )
    (repo / "README.md").write_text(
        "# Tiny Auth Service\n\nAuthenticate users; hash passwords.\n"
    )
    # semantic_search.scanner requires the repo to be a git repository
    # (it uses git ls-files to honour .gitignore). Initialise an empty
    # repo and add the seed files so the indexer sees them.
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@x", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    return repo


@pytest.fixture(scope="module")
def indexed_repo(small_repo: Path) -> Path:
    """Index the tiny repo once with a real HuggingFaceEmbedding.

    We need a real embedder (not MockEmbedding) for the leak signal:
    the bug lives in HuggingFaceEmbedding instantiation and its
    interaction with lancedb's tokio-backed C++ retainers. Mocks would
    paper over exactly the path we are guarding.
    """
    # Avoid network: only proceed if the model is already cached locally.
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    llamaindex_cache = Path(os.path.expanduser("~/.cache/llama_index"))
    if not (cache_dir.exists() or llamaindex_cache.exists()):
        pytest.skip(
            "Real model cache not present — this test needs "
            "multilingual-e5-small to be already downloaded."
        )

    index_repository(small_repo)
    return small_repo


# Tolerance:
# Even with the cache fix in place there will be SOME residual growth
# from per-call working set (query encoding, retrieval result Python
# objects, garbage that hasn't been collected yet). We pick 50 MB as a
# generous upper bound — well below the 390 MB/call we observed in the
# leaking version, and well above any legitimate per-call working set
# for a tiny corpus + top_k=1.
_MAX_GROWTH_PER_CALL_KB = 50 * 1024  # 50 MB
_NUM_PROBE_CALLS = 5


@pytest.mark.integration
def test_search_does_not_leak_per_call(indexed_repo: Path) -> None:
    """5 repeated searches must not grow RSS by more than 50 MB total.

    Without the fix this test sees ~5 × 390 MB ≈ 2 GB growth.
    With the fix RSS plateaus quickly after the first call.
    """
    # Warmup: the FIRST call always pays for one-time costs we cannot
    # avoid — model load, tokio pool spin-up, lance store open.
    # Whether the fix is in place or not, the first call costs ~1 GB.
    # The leak signal is in calls 2..N: with the bug, each call adds
    # another ~390 MB; with the fix, post-warmup RSS plateaus.
    search("authenticate user", repo_path=indexed_repo, top_k=1)
    gc.collect()  # let any transient allocations settle

    baseline_kb = _rss_kb()

    for i in range(_NUM_PROBE_CALLS):
        search(f"hash password attempt {i}", repo_path=indexed_repo, top_k=1)

    gc.collect()
    after_kb = _rss_kb()
    growth_kb = after_kb - baseline_kb

    # Diagnostic output the failing case wants.
    msg = (
        f"\n  baseline RSS after warmup: {baseline_kb:,} kB"
        f"\n  RSS after {_NUM_PROBE_CALLS} more searches: {after_kb:,} kB"
        f"\n  growth: {growth_kb:,} kB ({growth_kb / 1024:.1f} MB)"
        f"\n  budget: {_MAX_GROWTH_PER_CALL_KB:,} kB ({_MAX_GROWTH_PER_CALL_KB / 1024:.1f} MB)"
    )
    assert growth_kb < _MAX_GROWTH_PER_CALL_KB, (
        f"LEAK: search() grew RSS by {growth_kb:,} kB over "
        f"{_NUM_PROBE_CALLS} calls — expected < "
        f"{_MAX_GROWTH_PER_CALL_KB:,} kB.{msg}\n"
        f"Each call probably re-instantiates HuggingFaceEmbedding / "
        f"VectorStoreIndex / LanceDBVectorStore instead of reusing "
        f"the cached singletons. See semantic_search/retriever.py."
    )
