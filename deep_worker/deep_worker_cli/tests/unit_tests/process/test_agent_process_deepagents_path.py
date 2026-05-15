"""Regression test for the deepagents-path resolution bug.

Background
==========
``agent_process.py`` had a hard-coded line

    deepagents_path = PathLib("deep_worker/deepagents")
    sys.path.insert(0, str(deepagents_path))

that defeats the --dev flag: even when the bash launcher correctly
re-execs into the dev tree, this line yanks the production
``deepagents`` package onto sys.path with TOP priority, so the
dev-tree agent ends up running production middleware code.

Fix: derive the path from ``__file__`` (the same pattern already used
in agent.py, config.py, tools.py).

This test asserts the file does NOT contain the literal hard-coded
prod path. It is intentionally a textual test: stepping through the
real daemon-forking path that runs the offending line is infeasible
in a unit test, and the surface we want to protect is "no one types
that absolute path back into this file".

Design doc: docs/DESIGN_DEV_FLAG.md
"""

from __future__ import annotations

from pathlib import Path


# Path under test resolves relative to this test file so each tree's
# copy of the test validates its own copy of agent_process.py.
_THIS_FILE = Path(__file__).resolve()
_AGENT_PROCESS = (
    _THIS_FILE.parent.parent.parent.parent  # → deep_worker_cli/
    / "deep_worker_cli" / "process" / "agent_process.py"
)


def test_agent_process_exists() -> None:
    assert _AGENT_PROCESS.exists(), f"not found: {_AGENT_PROCESS}"


def test_no_hardcoded_prod_deepagents_path() -> None:
    """``agent_process.py`` must not contain the hard-coded
    deep_worker/deepagents string.
    """
    text = _AGENT_PROCESS.read_text()
    bad = "deep_worker/deepagents"
    assert bad not in text, (
        f"agent_process.py still contains the hard-coded prod-tree path "
        f"{bad!r}.  This breaks --dev (the agent in the dev tree will "
        f"still load deepagents from the prod tree, defeating the whole "
        f"point of --dev).  Replace with a __file__-relative path."
    )


_DW = "deep_worker"


def test_no_hardcoded_dev_deepagents_path() -> None:
    """Symmetric check: the dev path must not be hard-coded either.
    Both directions must be derived from __file__."""
    text = _AGENT_PROCESS.read_text()
    bad = f"/path/to/dev-tree/{_DW}/deepagents"
    assert bad not in text, (
        f"agent_process.py contains the hard-coded dev-tree path "
        f"{bad!r}.  Derive it from __file__ instead."
    )


def test_deepagents_path_is_file_relative() -> None:
    """Sanity: the file must contain a __file__-relative computation
    of the deepagents path, OR rely entirely on PYTHONPATH set by the
    launcher.  We accept both, but we want SOMETHING that proves the
    intent — the easiest signal is the literal substring."""
    text = _AGENT_PROCESS.read_text()
    has_file_relative = (
        "Path(__file__)" in text or "PathLib(__file__)" in text
    )
    # The file uses many __file__ references already; this is loose
    # but cheap and catches the regression "someone moved back to a
    # hard-coded string".
    assert has_file_relative, (
        "agent_process.py no longer derives any path from __file__. "
        "This is suspicious — was the --dev fix reverted?"
    )
