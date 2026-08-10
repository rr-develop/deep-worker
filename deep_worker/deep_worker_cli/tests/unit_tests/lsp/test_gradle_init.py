"""Unit tests for installing the Gradle init script for Android/Kotlin LSP.

Contract — see lsp/docs/DESIGN_KOTLIN_LSP_ANDROID_SOURCE_SETS.md §7.1 / §10.

Why this file exists: without the init script, Kotlin LSP cross-file
navigation silently returns empty on any AGP project (§1 of the design).
The script must land in the GLOBAL ``~/.gradle/init.d`` (Gradle ignores
project init scripts — verified, §7 limitation A), so an error in
permissions/idempotency affects all of the user's builds.

Tests T3/T4/T5 check the CONTENT of the resource, not the behavior of the
Python code. This is intentional: "guard comes before any mutation",
"reflection is wrapped in try/catch" and "only mark tasks with outcome ==
null" live in Groovy; they cannot be checked by an ordinary function call,
and violating them returns the defects from §4 and §6.1 of the design.
"""
from __future__ import annotations

import re
import stat
from pathlib import Path

from deep_worker_cli.lsp.gradle_init import (
    GRADLE_INIT_SCRIPT_NAME,
    InstallState,
    gradle_init_state,
    gradle_init_script_source,
    install_gradle_init_script,
    is_gradle_init_installed,
    uninstall_gradle_init_script,
)


# --------------------------------------------------------------------------
# T1 — installation: path, permissions, content
# --------------------------------------------------------------------------

def test_installs_script_with_expected_name_and_content(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    changed = install_gradle_init_script(init_d)

    target = init_d / GRADLE_INIT_SCRIPT_NAME
    assert changed is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == gradle_init_script_source()


def test_file_mode_is_0644_and_dir_0755(tmp_path: Path) -> None:
    """§7.1: 0644/0755 — not a secret, but also no 777/666 (project policy).

    The difference from mcp.json (0600/0700) is intentional: the script is
    read by the Gradle daemon and contains no secrets.
    """
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    target = init_d / GRADLE_INIT_SCRIPT_NAME

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(init_d.stat().st_mode) == 0o755


# --------------------------------------------------------------------------
# T2 — idempotency and protection of foreign edits
# --------------------------------------------------------------------------

def test_second_install_is_noop(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    assert install_gradle_init_script(init_d) is True
    assert install_gradle_init_script(init_d) is False


def test_does_not_overwrite_modified_script_without_force(tmp_path: Path) -> None:
    """The user may have tweaked the script — silently overwriting is not allowed."""
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    target = init_d / GRADLE_INIT_SCRIPT_NAME
    target.write_text("// hand-edited\n", encoding="utf-8")

    assert install_gradle_init_script(init_d) is False
    assert target.read_text(encoding="utf-8") == "// hand-edited\n"

    assert install_gradle_init_script(init_d, force=True) is True
    assert target.read_text(encoding="utf-8") == gradle_init_script_source()


def test_is_installed_reports_state(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    assert is_gradle_init_installed(init_d) is False
    install_gradle_init_script(init_d)
    assert is_gradle_init_installed(init_d) is True


# --------------------------------------------------------------------------
# T3 — guard before any mutation (§6.3 of the design)
# --------------------------------------------------------------------------

def test_resource_guards_on_lsp_plugin_before_any_mutation() -> None:
    """Outside a Kotlin LSP import the script must be a no-op.

    It lives in the GLOBAL init.d and applies to all of the user's builds, so
    the "is this an LSP import?" check must come earlier along the EXECUTION
    PATH than anything mutating (recollectVariants /
    installSourceSetsBuilderWrapper).

    We check exactly the single entry point — the body of ``withPlugin`` in
    ``allprojects``: the textual order of function/class declarations says
    nothing about execution order.
    """
    src = _strip_comments(gradle_init_script_source())

    entry = re.search(
        r"withPlugin\(pluginId\)\s*\{(.*?)\n\s*\}\n\s*\}\n\s*\}", src, flags=re.S
    )
    assert entry is not None, "allprojects/withPlugin entry point not found"
    body = entry.group(1)

    guard = re.search(r"lspPlugin\s*==\s*null\s*\)\s*return", body)
    assert guard is not None, "no early return in the entry point when the plugin is absent"

    for mutator in ("recollectVariants", "installSourceSetsBuilderWrapper"):
        pos = body.find(mutator)
        assert pos != -1, f"mutator {mutator!r} is not invoked from the entry point"
        assert guard.start() < pos, (
            f"mutation {mutator!r} runs before the guard for IdeaGradleLspPlugin"
        )


# --------------------------------------------------------------------------
# T4 — all reflection is wrapped in try/catch (§6.2 of the design)
# --------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def test_every_reflective_call_is_guarded_by_try_catch() -> None:
    """A version mismatch in the Kotlin LSP must not break the import.

    The worst acceptable outcome is falling back to the current (broken)
    behavior, not a build failure. We check that every reflective operation
    sits inside a try block.
    """
    src = _strip_comments(gradle_init_script_source())
    reflective = ("loadClass", "getDeclaredMethod", "getMethod", "setOutcome")

    for op in reflective:
        for m in re.finditer(re.escape(op), src):
            before = src[: m.start()]
            assert before.count("try") > before.count("catch"), (
                f"reflective call {op!r} outside try/catch (position {m.start()})"
            )


def test_resource_catches_assertion_and_illegal_state() -> None:
    """§6.1 + reviewer requirement: an assert in setOutcome under -ea and an
    IllegalStateException from an unconfigured task must not escape."""
    src = gradle_init_script_source()
    assert "Throwable" in src, (
        "catch must be on Throwable — otherwise AssertionError (-ea) "
        "and IllegalStateException would slip through"
    )


# --------------------------------------------------------------------------
# T5 — only mark tasks with outcome == null (§6.1 of the design)
# --------------------------------------------------------------------------

def test_only_marks_tasks_whose_outcome_is_unset() -> None:
    """`setOutcome` contains `assert outcome == null`: setting non-null over
    non-null again fails under -ea. Plus restoration must return the ORIGINAL
    outcome, not a made-up one."""
    src = _strip_comments(gradle_init_script_source())
    assert "!st.executed" in src or "!state.executed" in src, (
        "no !executed check before setOutcome — the assert precondition is violated"
    )
    assert "finally" in src, "outcome restoration must be in finally"


def test_restores_the_captured_previous_outcome_not_a_literal() -> None:
    """Restoration must use the CAPTURED value, not a made-up ``null``.

    ``getExecuted()`` is ``outcome != null``, but the reverse is not a valid
    code invariant: if the guard weakens (or the task turns out to be
    SKIPPED/NO_SOURCE), a hard-coded ``invoke(st, null)`` would clobber the
    task's real state instead of restoring it. Capturing ``prev`` makes
    restoration correct regardless of the guard.
    """
    src = _strip_comments(gradle_init_script_source())

    assert re.search(r"def\s+prev\s*=\s*\w+\.outcome", src), (
        "the previous outcome is not captured before setOutcome"
    )
    assert re.search(r"invoke\(\s*\w+\.state\s*,\s*\w+\.prev\s*\)", src), (
        "restoration does not use the captured prev"
    )
    assert not re.search(r"setOutcome\"?\s*,\s*outcomeClass\s*\)\s*\.?\s*invoke\(\s*\w+\s*,\s*null\s*\)", src), (
        "restoring with a literal null — a regression versus the verified patch"
    )


def test_marks_only_buildconfig_tasks() -> None:
    """The blast radius is limited: only generate*BuildConfig."""
    src = _strip_comments(gradle_init_script_source())
    assert "BuildConfig" in src
    assert "generate" in src


# --------------------------------------------------------------------------
# T6 — deletion only of "our" file
# --------------------------------------------------------------------------

def test_uninstall_removes_our_script(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    assert uninstall_gradle_init_script(init_d) is True
    assert not (init_d / GRADLE_INIT_SCRIPT_NAME).exists()


def test_uninstall_keeps_foreign_file(tmp_path: Path) -> None:
    """A file without our version marker is foreign — it must not be deleted."""
    init_d = tmp_path / ".gradle" / "init.d"
    init_d.mkdir(parents=True)
    target = init_d / GRADLE_INIT_SCRIPT_NAME
    target.write_text("// somebody else's script\n", encoding="utf-8")

    assert uninstall_gradle_init_script(init_d) is False
    assert target.exists()


def test_uninstall_is_noop_when_absent(tmp_path: Path) -> None:
    assert uninstall_gradle_init_script(tmp_path / ".gradle" / "init.d") is False


def test_resource_carries_version_marker() -> None:
    """The marker distinguishes "our" file from a foreign one without a
    byte-by-byte comparison and keeps future migrations possible (§7.1)."""
    first_line = gradle_init_script_source().splitlines()[0]
    assert re.match(r"^// deep-worker:kotlin-android-source-sets v\d+$", first_line), (
        f"the first line of the resource is not a version marker: {first_line!r}"
    )


# --------------------------------------------------------------------------
# T7 — atomicity
# --------------------------------------------------------------------------

def test_no_tmp_leftovers_after_install(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    leftovers = [p.name for p in init_d.iterdir() if p.name != GRADLE_INIT_SCRIPT_NAME]
    assert leftovers == [], f"leftover files from the write: {leftovers}"


# --------------------------------------------------------------------------
# Install state: "already current" ≠ "ours but different" ≠ "foreign"
#
# Reviewer note (non-blocker 1): when the script version was upgraded or after
# a user edit, the CLI printed "already installed" — the message was wrong and
# harmful because it hid the fact that an OLD version was in place.
# --------------------------------------------------------------------------

def test_state_absent(tmp_path: Path) -> None:
    assert gradle_init_state(tmp_path / ".gradle" / "init.d") is InstallState.ABSENT


def test_state_current_after_install(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    assert gradle_init_state(init_d) is InstallState.CURRENT


def test_state_ours_but_differs(tmp_path: Path) -> None:
    """Marker is ours, content is not: either a user edit or an old version."""
    init_d = tmp_path / ".gradle" / "init.d"
    install_gradle_init_script(init_d)
    target = init_d / GRADLE_INIT_SCRIPT_NAME
    marker = gradle_init_script_source().splitlines()[0]
    target.write_text(marker + "\n// locally tweaked\n", encoding="utf-8")

    assert gradle_init_state(init_d) is InstallState.OURS_OUTDATED


def test_state_foreign(tmp_path: Path) -> None:
    init_d = tmp_path / ".gradle" / "init.d"
    init_d.mkdir(parents=True)
    (init_d / GRADLE_INIT_SCRIPT_NAME).write_text("// alien\n", encoding="utf-8")

    assert gradle_init_state(init_d) is InstallState.FOREIGN
