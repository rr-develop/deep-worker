"""POLICIES §16 guard: nothing may reference the `.deepagents` config dir.

This has regressed twice. Value-level assertions on the resolvers (see
``test_no_deepagents_config_paths.py``) catch a broken resolver, but they
cannot catch a NEW module that builds the literal itself — which is exactly
how the last regression survived. This test scans the tree instead.

Scope note: the vendored Python package ``deep_worker/deepagents/`` and the
``deepagents-cli`` product name are a different thing that merely shares the
word. Only the leading-dot *config directory* path is banned.

Tests that seed a foreign tree to prove it stays invisible are allowed to
name it, but per-line rather than per-file: the mention must cite the policy
(``§16``) within a few lines. A blanket file exemption would let a genuine
regression hide inside a file that legitimately mentions the path once.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

#: The banned thing: the config directory, always written with a leading dot
#: and followed by a separator or a quote/end (so ``deepagents.middleware``
#: and ``deep_worker/deepagents/`` do not match).
_BANNED = re.compile(r"\.deepagents(?=[/\\\"'`\s]|$)")

#: Files allowed to name the directory, and why.
_ALLOWED = {
    # States the ban itself.
    "docs/POLICIES.md",
    # Raise-on-use guards + the comments explaining them.
    "deep_worker/deep_worker_cli/deep_worker_cli/storage_paths.py",
    # One-line comments recording that the path is deliberately NOT consulted.
    "deep_worker/deep_worker_cli/deep_worker_cli/config.py",
    "deep_worker/deep_worker_cli/deep_worker_cli/project_utils.py",
    "deep_worker/deep_worker_cli/deep_worker_cli/runtime/provider_registry.py",
    "deep_worker/deep_worker_cli/deep_worker_cli/external_mcp/catalog.py",
    "deep_worker/deep_worker_cli/deep_worker_cli/process/env_loader.py",
    "deep_worker/deep_worker_cli/deep_worker_cli/sessions.py",
    # Keeps a stray directory from ever being committed if one reappears.
    ".gitignore",
    # These two assert the ban, so they must name the thing being banned.
    "deep_worker/deep_worker_cli/tests/unit_tests/test_no_deepagents_config_paths.py",
    "deep_worker/deep_worker_cli/tests/unit_tests/"
    "test_no_deepagents_references_repo_wide.py",
}

#: Historical docs ABOUT the upstream ancestor — not our configuration.
_ALLOWED_PREFIXES = ("docs/deepagents/",)

#: Tests may seed a foreign tree to prove it stays invisible, so a blanket
#: file-level exemption would hide a genuine regression in the same file.
#: Instead each such line must carry the policy reference — a plain
#: ``tmp_path / ".deepagents"`` with no ``§16`` nearby is still a defect.
#: Only an explicit policy reference counts. Synonyms like "foreign" or
#: "ancestor" were tried and removed: 123 lines across the repo carry them
#: without meaning anything about this ban, so any future regression landing
#: near one would pass silently — exactly the blind spot this guard exists
#: to close. Citing §16 is what proves the mention is deliberate.
_POLICY_REF = re.compile(r"§16")


#: Start of the enclosing test/function — the unit a mention belongs to.
_DEF = re.compile(r"^\s*(?:async\s+)?def\s|^\s*class\s")


def _line_is_pinning_the_ban(lines: list[str], idx: int) -> bool:
    """True when the enclosing test (or the module header) cites the policy.

    The reference usually sits in the test's docstring or in the assertion
    message, which can be well above or below the path itself. So the window
    is the enclosing ``def``/``class`` body rather than a fixed line count,
    plus the module docstring — a file-wide claim about what it is pinning.
    """
    start = 0
    for i in range(idx, -1, -1):
        if _DEF.match(lines[i]):
            start = i
            break
    end = len(lines)
    for i in range(idx + 1, len(lines)):
        if _DEF.match(lines[i]):
            end = i
            break
    scope = lines[start:end]
    header = lines[:40]
    return any(_POLICY_REF.search(line) for line in (*scope, *header))


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    pytest.skip("not in a git checkout")


def _tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        pytest.skip("git ls-files unavailable")
    return [p for p in out.stdout.split("\0") if p]


def test_no_tracked_file_references_the_deepagents_config_dir() -> None:
    root = _repo_root()
    offenders: list[str] = []

    for rel in _tracked_files(root):
        if rel in _ALLOWED or rel.startswith(_ALLOWED_PREFIXES):
            continue
        # The vendored package is a legitimate import path, not config.
        if rel.startswith("deep_worker/deepagents/"):
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable — nothing to scan
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if not _BANNED.search(line):
                continue
            if _line_is_pinning_the_ban(lines, lineno - 1):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()[:110]}")

    assert not offenders, (
        "POLICIES §16: these files reference the banned `.deepagents` config "
        "directory. Use ~/.deep_worker (see storage_paths.config_home / "
        "agent_config_dir / project_config_dir).\n\n" + "\n".join(offenders)
    )
