"""Let a live provider CLI use the operator's login — without exposing the rest.

``tests/conftest.py`` points ``HOME`` at a sandbox for every test (owner
directive 2026-08-17: "Tests must not mess with production configs.
NEVER!!!"). That is right, and it must stay — but it also hides
``~/.grok/auth.json``, ``~/.codex/auth.json`` and ``~/.claude`` from the
CLIs, so a live provider test cannot authenticate. Symptom, measured
2026-08-26 while writing these tests:

  * grok — skipped ("~/.grok/auth.json missing"),
  * codex — ``401 Unauthorized`` against ``wss://api.openai.com``,
  * claude-code-sdk — ``Not logged in · Please run /login``,

while all three worked perfectly from a plain shell in the same second.
Every one of those reads as "the provider is broken", and none of them was.

The bridge below symlinks ONLY the provider home directories into the
sandbox. What that does and does not expose:

  * ``~/.deep_worker`` stays sandboxed — the runtime tree the directive is
    about (``provider_availability.json``, per-agent state, token stats) is
    untouched, which is the whole point of the fixture;
  * the CLIs read their own credentials and write their own session logs in
    their own homes, exactly as they do for a real agent. That is not a side
    effect to be avoided: those logs ARE the telemetry the live context gauge
    reads, so a test that redirected them would be testing nothing.

Nothing here is written by the test itself, and no ``allow_live_home`` marker
is needed: HOME remains the sandbox.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path

import pytest

# The provider CLIs' own homes. Kept explicit — never a glob over dotfiles.
_PROVIDER_HOMES = (".grok", ".codex", ".claude", ".claude.json")

# Installed programs that live inside the runtime tree rather than beside the
# other CLIs: the ACP adapters are npm packages under
# ``~/.deep_worker/acp/<name>``. They are dependencies, not state — the same
# category as the ``claude`` binary itself — so the sandbox links them while
# everything else under ``~/.deep_worker`` (agents/, providers.json, token
# stats) stays isolated. Without this a claude-code agent cannot start at
# all: "claude-agent-acp adapter not installed".
_RUNTIME_DEPENDENCIES = ((".deep_worker", "acp"),)


def real_home() -> Path:
    """The operator's home, independent of ``HOME``.

    ``Path.home()`` and ``expanduser`` both consult ``HOME``, which the
    sandbox fixture has already moved by the time any test fixture runs.
    The passwd database has not moved.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


@pytest.fixture
def provider_login() -> list[str]:
    """Expose the provider CLIs' logins inside the sandboxed HOME.

    Returns the names actually linked, so a test can report what it found.
    Skips rather than silently running against the live tree if the sandbox
    fixture is ever removed.
    """
    sandbox = Path(os.environ.get("HOME", ""))
    real = real_home()
    if not sandbox or sandbox == real:
        pytest.skip(
            "HOME is not sandboxed — refusing to run a live test against the "
            "operator's runtime tree"
        )
    linked: list[str] = []
    for name in _PROVIDER_HOMES:
        source = real / name
        target = sandbox / name
        if not source.exists() or target.exists():
            continue
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError:  # pragma: no cover — hostile filesystem
            continue
        linked.append(name)
    for parent, child in _RUNTIME_DEPENDENCIES:
        source = real / parent / child
        target = sandbox / parent / child
        if not source.exists() or target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError:  # pragma: no cover — hostile filesystem
            continue
        linked.append(f"{parent}/{child}")
    return linked
