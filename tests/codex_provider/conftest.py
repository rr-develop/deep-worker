"""Make this suite independent of a locally installed Codex CLI.

``CodexChatModel``'s construction path resolves the binary eagerly
(``exec_runner._find_codex_path``: ``CODEX_PATH`` env → ``shutil.which``
→ ``FileNotFoundError``). Every test here drives the model through fakes
and never spawns the process, so the resolution is the ONLY thing that
needs a path to exist — and on a machine without the CLI (a CI runner)
it raised at construction time, failing 41 of these tests while the same
suite was green on a developer box that happened to have ``codex``
installed via npm.

So point ``CODEX_PATH`` at a deliberately non-existent file: resolution
succeeds, and if any test ever actually tries to EXEC it, that is a bug
this fixture makes loud (ENOENT on spawn) instead of silently exercising
the developer's real binary.

A pre-set ``CODEX_PATH`` is honoured, so an integration run can still
point the suite at a real binary.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, name="stub_codex_path")
def fixture_stub_codex_path(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.environ.get("CODEX_PATH"):
        monkeypatch.setenv("CODEX_PATH", "/nonexistent/codex-not-spawned-by-unit-tests")
