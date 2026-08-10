"""Anti-recurrence: no repository-provided command may invoke ``uv`` bare.

This is the test that stops the *defect class* rather than the one instance.

A bare ``uv run`` / ``uv sync`` / ``uv venv`` creates a project ``.venv`` and
leaves it behind. The pre-commit hook then blocks the next commit on the venv
check — pointing at virtual environments the developer never knowingly created,
and hiding whatever the real problem was. An earlier incident (a stale ``rpyc``
in a ``.venv`` shadowing a sibling async library project, 21 GB RSS, a day of
debugging) came out of exactly this loop, and the hook names ``uv run`` in the
Makefiles as a contributing cause.

Every such command must go through ``deep_worker/scripts/uv-run.sh``, which owns
the cleanup.

Allow-listing is **occurrence-level, not path-level**: the expectation below is
an exact table of (path, normalized line, count), compared as a whole. A
path-level list would let a future bare command added *inside* an already-allowed
file — ``pre-commit-type-check.sh`` above all — pass silently. Here, any extra
occurrence goes RED even in an allowed file, and a *removed* one goes RED too,
so the table cannot quietly rot.

Design: deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return Path(_git("rev-parse", "--show-toplevel", cwd=here).strip())


ROOT = _repo_root()

#: Commands that create a project-local ``.venv`` as a side effect.
#:
#: Two spellings are matched, because both really occur in this repo:
#:   * shell/Makefile form  -- ``uv run …``
#:   * argv-list form       -- ``["uv", "run", …]`` / ``('uv', 'sync', …)``,
#:     which a Python service passes straight to subprocess and which the
#:     shell-only pattern would miss entirely.
UV_COMMAND = re.compile(
    r"""\buv\s+(run|sync|venv)\b"""
    r"""|["']uv["']\s*,\s*["'](run|sync|venv)["']"""
)

#: Files whose *content* we scan. Executables and anything a human reads as
#: instructions. Kept broad on purpose: the point is to catch a new surface.
SCANNED_SUFFIXES = {
    ".sh",
    ".bash",
    ".md",
    ".toml",
    ".cfg",
    ".yaml",
    ".yml",
    # .py included deliberately: a service that shells out to `uv run` from
    # Python leaks a .venv exactly like a Makefile recipe does. Excluding it
    # would leave the largest code surface in the repo unscanned.
    ".py",
}
SCANNED_NAMES = {"Makefile", "makefile", "GNUmakefile", "pre-commit", "post-commit"}

#: Directories that are not repository-provided instructions.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}

# ---------------------------------------------------------------------------
# The expected occurrences, as an exact MULTISET.
#
# Keyed by (repo-relative path, normalized line) -> expected count. Counts
# matter: a second, identical invocation added to an already-allowed file must
# go RED, and a set would silently swallow it.
#
# Only EXECUTABLE invocations are listed. Prose -- policy banners that forbid
# the command, agent prompts, the design doc's verbatim reproduction, the
# historical transcript -- is excluded structurally rather than by heuristic
# (see EXCLUDED_PATHS and _executable_occurrences below), because guessing
# "is this line a command or a mention?" from punctuation is exactly the kind
# of rule that false-greens.
#
# Adding an entry here is a deliberate act a reviewer will see in the diff.
# ---------------------------------------------------------------------------
EXPECTED_EXEC: dict[tuple[str, str], int] = {
    # The pre-commit hook's own type check. Compliant ONLY because of its
    # `trap cleanup_venvs EXIT INT TERM`; see that script's header.
    (
        "deep_worker/scripts/pre-commit-type-check.sh",
        "if uv run --all-groups ty check deep_worker_cli; then",
    ): 1,
    (
        "deep_worker/scripts/pre-commit-type-check.sh",
        "if uv run --all-groups ty check deepagents; then",
    ): 1,
    # The single seam every other command routes through. `setsid` puts the
    # payload in its own process group so the whole group can be killed before
    # cleanup -- see the script's header.
    ("deep_worker/scripts/uv-run.sh", 'setsid uv run "$@" &'): 1,
}

#: Only the test fixtures themselves are excluded wholesale: theyquote these
#: commands in assertions and stub bodies, and self-reference is unavoidable.
#: EVERY other file -- including policy banners, agent prompts and config
#: comments -- is pinned occurrence by occurrence in PROSE_EXPECTED below, so a
#: newly added actionable command inside one of them goes RED instead of hiding
#: behind a path-level exemption.
EXCLUDED_PATHS = {
    "tests/test_no_bare_uv_invocations.py",
    "tests/test_uv_run_wrapper.py",
}

#: Non-executable mentions: policy text that FORBIDS the command, agent prompts,
#: the design doc's verbatim reproduction, a historical transcript, config
#: comments. Pinned as an exact multiset for the same reason as EXPECTED_EXEC --
#: a new line in any of these files changes the count and must be reviewed.
PROSE_EXPECTED: dict[tuple[str, str], int] = {
    (
        'deep_worker/deep_worker_cli/Makefile',
        '# All `uv run` invocations go through this wrapper, which creates the project',
    ): 1,
    (
        'deep_worker/deep_worker_cli/Makefile',
        '# EXIT/INT/TERM). A bare `uv run` here would leave a .venv behind and block the',
    ): 1,
    (
        'deep_worker/deep_worker_cli/WORKING_DIR_FIX.md',
        '$ uv run --all-groups pytest tests/unit_tests/cli/ -v',
    ): 1,
    (
        'deep_worker/deep_worker_cli/WORKING_DIR_FIX.md',
        '$ uv run --all-groups pytest tests/unit_tests/cli/test_agent_command_working_dir.py -v',
    ): 1,
    (
        'deep_worker/deep_worker_cli/pyproject.toml',
        '# Pre-commit hook uses: uv run --all-groups ty check deep_worker_cli',
    ): 1,
    (
        'deep_worker/deepagents/Makefile',
        '# All `uv run` invocations go through this wrapper, which creates the project',
    ): 1,
    (
        'deep_worker/deepagents/Makefile',
        '# EXIT/INT/TERM). A bare `uv run` here would leave a .venv behind and block the',
    ): 1,
    (
        'deep_worker/deepagents/Makefile',
        '# [ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ty check $(PYTHON_FILES)',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '$ uv run --all-groups ty check deepagents',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '- cd .../deep_worker_cli && uv run --all-groups ty check deep_worker_cli',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '- cd .../deepagents && uv run --all-groups ty check deepagents',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '- works for any `uv run` payload — `ty`, `pytest`, `ruff`, `python` — so it',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '> But `uv run` in the Makefile and scripts/pre-commit-type-check.sh KEPT',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'Add `deep_worker/scripts/uv-run.sh` — a thin wrapper that runs `uv run "$@"` in',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'D. **`uv run --no-project` / `uvx`.**',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'Every *other* repository-provided command that shells out to `uv run` has no',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'In both `Makefile`s, replace bare `uv run …` with `$(UV_RUN) …` where',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'R5. `uv run` stays confined to cleanup-safe sites. After this change the',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'README/instruction Markdown — for bare `uv run` / `uv sync` / `uv venv`.',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'Tracking only the immediate `uv` PID is **also** insufficient. `uv run` launches',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '`deep_worker/scripts/pre-commit-type-check.sh` is a sanctioned `uv run` site.',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        "an internal service's prompt strings that *forbid* `uv run`, a comment in",
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        '`uv run` anywhere else remains forbidden and is caught by a test.',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'bare `uv run` in `test`, `tests`, `coverage`, `integration_test(s)`,',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'cleanup at all. `git grep -n "uv run"` over tracked files finds three classes:',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'executable occurrences are exactly: the two `uv run --all-groups ty check`',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'lines in `pre-commit-type-check.sh`, and the single `uv run "$@"` in',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'pre-existence check → `uv run` → cleanup → release. `flock` blocks in the',
    ): 1,
    (
        'deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md',
        'reference the wrapper and contain no bare `uv run` (runtime assertion,',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# - DO NOT remove the cleanup_venvs trap (it is what makes `uv run` here',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# 1. `uv run` creates a temporary `.venv` for the type check.',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# CLEANUP: remove any `.venv` `uv run` leaves behind.',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# Remove project-local .venv directories created by `uv run`.',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# `uv run --all-groups ty check` (below) NEEDS a project venv to resolve',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# wrong — the lock file is what makes ``uv run`` deterministic, so',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# This is NOT permission to use `uv run` anywhere else. The Makefile,',
    ): 1,
    (
        'deep_worker/scripts/pre-commit-type-check.sh',
        '# where `uv run` is allowed. The reason: `ty check` without installed',
    ): 1,
    (
        'deep_worker/scripts/type-check.sh',
        '# cd .../deepagents && uv run --all-groups ty check deepagents',
    ): 1,
    (
        'deep_worker/scripts/uv-run.sh',
        '# CLEANUP-SAFE `uv run` SEAM',
    ): 1,
    (
        'deep_worker/scripts/uv-run.sh',
        '# This is NOT a licence to use `uv run` elsewhere. Bare invocations remain',
    ): 1,
    (
        'deep_worker/scripts/uv-run.sh',
        '# forbidden. `uv run` needs a project venv. The compromise the repo already made',
    ): 1,
}

#: Within a SCANNED file, a comment line is a mention rather than a command.
#: This is the only structural rule, and it is unambiguous: a shell/Makefile
#: comment cannot execute. Markdown prose files are handled by EXCLUDED_PATHS.
COMMENT_PREFIXES = ("#",)


def _normalize(line: str) -> str:
    return " ".join(line.split())


def _iter_scanned_files() -> list[Path]:
    tracked = _git("ls-files", "-z", cwd=ROOT).split("\0")
    out = []
    for rel in tracked:
        if not rel:
            continue
        path = ROOT / rel
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SCANNED_NAMES or path.suffix in SCANNED_SUFFIXES:
            if path.is_file():
                out.append(path)
    return out


def _executable_occurrences() -> "Counter[tuple[str, str]]":
    """Multiset of (path, normalized line) for every occurrence NOT pinned as prose.

    Anything not explicitly listed in PROSE_EXPECTED is treated as an
    executable invocation and must therefore appear in EXPECTED_EXEC. That
    direction matters: an unclassified new line fails LOUDLY rather than being
    guessed at, so there is no heuristic left to false-green.

    A Counter rather than a set: duplicates are the interesting case. A second
    identical `uv run` line added to an allowed file would vanish into a set.
    """
    found: Counter[tuple[str, str]] = Counter()
    prose = Counter(PROSE_EXPECTED)
    for path in _iter_scanned_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if not UV_COMMAND.search(line):
                continue
            key = (rel, _normalize(line))
            if prose[key] > 0:
                prose[key] -= 1  # consume one pinned prose occurrence
                continue
            found[key] += 1
    return found


def _prose_occurrences() -> "Counter[tuple[str, str]]":
    """Multiset of (path, normalized line) for every NON-executable mention."""
    found: Counter[tuple[str, str]] = Counter()
    for path in _iter_scanned_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if not UV_COMMAND.search(line):
                continue
            key = (rel, _normalize(line))
            if key in EXPECTED_EXEC:
                continue
            found[key] += 1
    return found


def _occurrence_lines() -> dict[tuple[str, str], list[int]]:
    """Line numbers per occurrence, for readable failure messages."""
    lines: dict[tuple[str, str], list[int]] = {}
    for path in _iter_scanned_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if UV_COMMAND.search(line):
                lines.setdefault((rel, _normalize(line)), []).append(lineno)
    return lines


def test_no_unexpected_bare_uv_invocations() -> None:
    """The exact MULTISET of executable ``uv run/sync/venv`` calls is pinned.

    RED when a new bare invocation appears ANYWHERE -- including a second copy
    inside a file that already contains a sanctioned one.
    """
    actual = _executable_occurrences()
    expected = Counter(EXPECTED_EXEC)
    lines_by_key = _occurrence_lines()

    def _where(key: tuple[str, str]) -> str:
        rel, line = key
        spots = ",".join(str(n) for n in lines_by_key.get(key, []))
        return f"  {rel}:{spots or '?'}: {line}"

    extra = actual - expected
    missing = expected - actual

    assert not extra, (
        "New or duplicated bare `uv` invocation(s) found. These create a "
        "project .venv and leave it behind, which blocks the NEXT commit on "
        "the venv check and restarts the same failure cycle.\n\n"
        + "\n".join(
            f"{_where(k)}  (found {actual[k]}x, expected {expected.get(k, 0)}x)"
            for k in sorted(extra)
        )
        + "\n\nRoute it through deep_worker/scripts/uv-run.sh, which owns the "
        "cleanup (trap on EXIT/INT/TERM). Only add to EXPECTED_EXEC if the call "
        "site genuinely owns equivalent cleanup itself."
    )
    assert not missing, (
        "Expected sanctioned invocation(s) disappeared or changed:\n"
        + "\n".join(
            f"  {k[0]}: {k[1]}  (expected {expected[k]}x, found {actual.get(k, 0)}x)"
            for k in sorted(missing)
        )
        + "\n\nIf this was intentional, update EXPECTED_EXEC. This assertion "
        "exists so the allow-list cannot silently rot."
    )


def test_prose_occurrences_are_pinned_exactly() -> None:
    """The non-executable mentions are pinned as a multiset too.

    Without this, PROSE_EXPECTED could quietly list a line that no longer
    exists, which would let a genuinely new command take its place and be
    consumed as "prose" by the scanner above.
    """
    actual = _prose_occurrences()
    expected = Counter(PROSE_EXPECTED)
    lines_by_key = _occurrence_lines()

    stale = expected - actual
    assert not stale, (
        "PROSE_EXPECTED pins mention(s) that are no longer present:\n"
        + "\n".join(
            f"  {rel}: {line}  (expected {expected[(rel, line)]}x, "
            f"found {actual.get((rel, line), 0)}x)"
            for rel, line in sorted(stale)
        )
        + "\n\nRemove them. A stale prose entry is a free slot a new command "
        "could slip into."
    )
    # The other direction (new prose) is already reported by
    # test_no_unexpected_bare_uv_invocations, which treats anything unpinned as
    # executable -- deliberately the louder failure.
    unpinned = actual - expected
    assert not unpinned, (
        "New mention(s) of a venv-creating uv command:\n"
        + "\n".join(
            f"  {rel}:{','.join(str(n) for n in lines_by_key.get((rel, line), []))}: "
            f"{line}"
            for rel, line in sorted(unpinned)
        )
        + "\n\nIf it is prose (policy text, documentation, a transcript), add it "
        "to PROSE_EXPECTED. If it is a command, route it through "
        "deep_worker/scripts/uv-run.sh."
    )


def test_excluded_paths_all_exist() -> None:
    """A stale entry in EXCLUDED_PATHS would silently widen the scan's blind spot."""
    missing = sorted(rel for rel in EXCLUDED_PATHS if not (ROOT / rel).is_file())
    assert not missing, (
        "EXCLUDED_PATHS lists files that no longer exist:\n"
        + "\n".join(f"  {rel}" for rel in missing)
        + "\n\nRemove them, or the exclusion list drifts out of sync with the "
        "tree and stops meaning anything."
    )


def test_makefiles_route_through_the_wrapper() -> None:
    """Both package Makefiles must call the seam, not ``uv`` directly.

    ``make test`` / ``make type`` are advertised in each Makefile's help target,
    so they are commands the repository actively recommends -- and they were the
    largest leaking surface.
    """
    for rel in (
        "deep_worker/deep_worker_cli/Makefile",
        "deep_worker/deepagents/Makefile",
    ):
        path = ROOT / rel
        text = path.read_text()
        assert "UV_RUN" in text, (
            f"{rel} does not define UV_RUN. Its recipes must go through "
            f"deep_worker/scripts/uv-run.sh so they cannot leak .venv."
        )
        for lineno, line in enumerate(text.splitlines(), start=1):
            if UV_COMMAND.search(line) and not line.lstrip().startswith("#"):
                pytest.fail(
                    f"{rel}:{lineno} invokes uv directly: {line.strip()}\n"
                    f"Use $(UV_RUN) instead."
                )


def test_uv_run_discovery_is_makefile_relative() -> None:
    """Wrapper discovery must not depend on the caller's cwd or on git.

    ``:=`` is required: MAKEFILE_LIST grows as makefiles are included, so a
    recursive ``=`` would resolve against whatever was last included.
    """
    for rel in (
        "deep_worker/deep_worker_cli/Makefile",
        "deep_worker/deepagents/Makefile",
    ):
        text = (ROOT / rel).read_text()
        match = re.search(r"^UV_RUN\s*(:?=)\s*(.+)$", text, re.M)
        assert match, f"{rel} has no UV_RUN assignment"
        assert match.group(1) == ":=", (
            f"{rel} uses recursive '=' for UV_RUN; use ':=' so MAKEFILE_LIST is "
            f"evaluated at definition time."
        )
        value = match.group(2)
        assert "MAKEFILE_LIST" in value, (
            f"{rel} UV_RUN={value!r} does not derive from MAKEFILE_LIST, so it "
            f"depends on the caller's cwd."
        )
        assert "git " not in value, (
            f"{rel} UV_RUN={value!r} shells out to git; discovery must work in "
            f"a plain source tree too."
        )


def test_remediation_output_points_at_the_wrapper() -> None:
    """The hook's failure message must not tell humans to run a bare command.

    This is the loop that started it: the check fails, the developer is told to
    run a bare ``uv run``, and the resulting .venv blocks the next commit.
    """
    text = (ROOT / "deep_worker" / "scripts" / "pre-commit-type-check.sh").read_text()
    advice = text.split("To fix this issue:", 1)
    assert len(advice) == 2, "Could not locate the remediation block in the hook"
    block = advice[1]

    assert "type-check.sh" in block, (
        "The hook's remediation advice must point at "
        "deep_worker/scripts/type-check.sh, which cleans up after itself."
    )
    for lineno, line in enumerate(block.splitlines(), start=1):
        if UV_COMMAND.search(line):
            pytest.fail(
                f"Remediation advice still suggests a bare uv command: "
                f"{line.strip()}"
            )


def test_scripts_readme_points_at_the_wrapper() -> None:
    readme = ROOT / "deep_worker" / "scripts" / "README.md"
    text = readme.read_text()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if UV_COMMAND.search(line) and "uv-run.sh" not in line:
            pytest.fail(
                f"{readme.relative_to(ROOT)}:{lineno} recommends a bare uv "
                f"command: {line.strip()}\nPoint at scripts/type-check.sh."
            )


# ---------------------------------------------------------------------------
# Trap/scope pins. The hook and the wrapper deliberately have DIFFERENT
# cleanup scopes, so each is asserted against its own documented contract --
# neither is derived from the other.
# ---------------------------------------------------------------------------


def test_hook_keeps_its_own_trap_and_two_package_scope() -> None:
    """The hook cleans both fixed package paths, regardless of cwd.

    It is deliberately NOT routed through uv-run.sh: that would add a runtime
    dependency to the commit gate, so a broken wrapper would break every commit.
    """
    text = (ROOT / "deep_worker" / "scripts" / "pre-commit-type-check.sh").read_text()
    assert "trap cleanup_venvs EXIT INT TERM" in text, (
        "The hook lost its cleanup trap. That trap is the ONLY reason `uv run` "
        "is permissible there at all."
    )
    for pkg in ("DEEP_WORKER_CLI_DIR", "DEEPAGENTS_DIR"):
        assert f'"${{{pkg}}}/.venv"' in text, (
            f"The hook no longer cleans {pkg}/.venv. It always checks both "
            f"packages, so it must always clean both."
        )


@pytest.mark.skipif(
    not (ROOT / "deep_worker" / "scripts" / "uv-run.sh").exists(),
    reason="uv-run.sh does not exist yet (RED phase of TDD)",
)
def test_wrapper_has_trap_and_single_resolved_root_scope() -> None:
    """The wrapper cleans exactly ONE path: the resolved root it was invoked for."""
    text = (ROOT / "deep_worker" / "scripts" / "uv-run.sh").read_text()
    assert re.search(r"^\s*trap\s+\S+\s+EXIT INT TERM", text, re.M), (
        "uv-run.sh must trap EXIT, INT and TERM so cleanup runs on every exit "
        "path, not just the happy one."
    )
    assert "flock" in text, (
        "uv-run.sh must serialize per project root with flock; without it two "
        "invocations both claim ownership of one .venv."
    )
    assert "sha256" in text.lower(), (
        "The lock filename must be a full SHA-256 digest of the resolved root."
    )
