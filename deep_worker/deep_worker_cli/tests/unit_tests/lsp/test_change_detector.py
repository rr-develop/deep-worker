"""RED tests for the event-driven change detector (DESIGN §9, commit 5).

The detector is the "economical update" half of the design: it must
refresh the project model when the module STRUCTURE changes, and stay
completely silent when it does not.

Contract pinned here (all from §9 of the signed-off design):

* provider rules are selected by marker files in the project root;
* edits inside existing sources never trigger a refresh (the LSP already
  handles those via ``didChangeWatchedFiles``);
* a structural change (e.g. a new ``include(":lib")``) does trigger one;
* a burst of events debounces to a single refresh (300 ms);
* polyglot projects merge rule sets, each provider firing independently;
* an unchanged structural snapshot is a no-op even if a trigger file was
  touched — this is the economy rule the whole design exists for.

These are written RED first: the module does not exist yet.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.change_detector import (
    ChangeDetector,
    rules_for_project,
    snapshot_for,
)


# ── provider dispatch ────────────────────────────────────────────────


def test_provider_dispatcher_picks_kotlin_for_gradle_root(tmp_path: Path):
    (tmp_path / "settings.gradle.kts").write_text('include(":app")\n')
    rules = rules_for_project(tmp_path)
    assert [r.name for r in rules] == ["kotlin-gradle"]


def test_provider_dispatcher_picks_python_for_pyproject_root(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    rules = rules_for_project(tmp_path)
    assert [r.name for r in rules] == ["python"]


def test_default_rules_only_fires_without_specific_provider(tmp_path: Path):
    """§9: defaults are a fallback and must never run alongside a
    specific provider — that would cause a double refresh."""
    (tmp_path / "README.md").write_text("nothing recognisable\n")
    assert [r.name for r in rules_for_project(tmp_path)] == ["default"]

    (tmp_path / "go.mod").write_text("module x\n")
    names = [r.name for r in rules_for_project(tmp_path)]
    assert "go" in names
    assert "default" not in names, "default must not co-fire with a provider"


def test_polyglot_project_merges_rule_sets(tmp_path: Path):
    """Android + a TS web module in one repo → both providers active,
    each watching its own paths, no duplicates."""
    (tmp_path / "settings.gradle.kts").write_text('include(":app")\n')
    (tmp_path / "package.json").write_text('{"name":"web"}\n')

    rules = rules_for_project(tmp_path)
    names = sorted(r.name for r in rules)
    assert names == ["kotlin-gradle", "typescript"]
    assert "default" not in names

    # Watched paths must not overlap into duplicate kernel watches.
    all_paths = [p for r in rules for p in r.trigger_globs]
    assert len(all_paths) == len(set(all_paths)), "duplicate trigger globs"


# ── refresh decisions ────────────────────────────────────────────────


def test_change_detector_ignores_intra_module_edits(tmp_path: Path):
    """Editing a .kt inside an existing module must NOT refresh."""
    (tmp_path / "settings.gradle.kts").write_text('include(":app")\n')
    src = tmp_path / "app" / "src" / "main" / "kotlin"
    src.mkdir(parents=True)
    (src / "Main.kt").write_text("fun main() {}\n")

    det = ChangeDetector(tmp_path)
    before = det.snapshot()

    (src / "Main.kt").write_text("fun main() { println(1) }\n")

    assert det.should_refresh([src / "Main.kt"]) is False
    assert det.snapshot() == before, "intra-module edit changed the snapshot"


def test_non_trigger_event_does_not_consume_a_pending_structural_change(
    tmp_path: Path,
):
    """The trigger gate is load-bearing, not just an optimisation.

    ``should_refresh`` commits the recomputed snapshot when it reports
    True. If a non-trigger event (a plain ``.kt`` save) were allowed to
    reach the snapshot comparison, it would observe a structural change
    that landed on disk moments earlier, refresh nothing useful, and
    *absorb* the delta — so the real ``settings.gradle.kts`` event that
    follows would compare equal and return False. The project model
    would stay stale until some unrelated later change happened to move
    it again.
    """
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')
    src = tmp_path / "app" / "src" / "main" / "kotlin"
    src.mkdir(parents=True)
    source = src / "Main.kt"
    source.write_text("fun f() {}\n")

    det = ChangeDetector(tmp_path)

    # A structural change lands on disk, but the event we happen to
    # receive first is for the unrelated source file.
    settings.write_text('include(":app")\ninclude(":lib")\n')

    assert det.should_refresh([source]) is False, (
        "a non-trigger path must never reach the snapshot comparison"
    )
    # The delta must still be pending, not silently consumed.
    assert det.should_refresh([settings]) is True, (
        "structural change was swallowed by an unrelated event"
    )


def test_build_gradle_kts_is_a_trigger_despite_kts_being_a_source_suffix(
    tmp_path: Path,
):
    """``.kts`` is both a source and a manifest suffix — manifests win."""
    (tmp_path / "settings.gradle.kts").write_text('include(":app")\n')
    module = tmp_path / "app"
    module.mkdir()
    build = module / "build.gradle.kts"
    build.write_text("plugins { id(\"x\") }\n")

    det = ChangeDetector(tmp_path)
    assert det._is_trigger(build) is True, (
        "**/build.gradle.kts must match the Gradle trigger globs"
    )


def test_change_detector_triggers_on_settings_include_change(tmp_path: Path):
    """Adding a module to settings.gradle.kts MUST refresh."""
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')

    det = ChangeDetector(tmp_path)
    before = det.snapshot()

    settings.write_text('include(":app")\ninclude(":lib")\n')

    assert det.should_refresh([settings]) is True
    assert det.snapshot() != before


def test_same_structural_change_does_not_refresh_twice(tmp_path: Path):
    """A refresh must COMMIT the new snapshot as the baseline.

    If it did not, every subsequent trigger event would keep comparing
    against the stale original and re-refresh forever — the opposite of
    the economy rule, and a busy-loop against the language server.
    """
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')

    det = ChangeDetector(tmp_path)
    settings.write_text('include(":app")\ninclude(":lib")\n')

    assert det.should_refresh([settings]) is True
    # Same file touched again, structure unchanged since the refresh.
    assert det.should_refresh([settings]) is False, (
        "snapshot was not committed — the same change refreshes forever"
    )


def test_unchanged_snapshot_is_a_noop_even_if_trigger_file_touched(
    tmp_path: Path,
):
    """The economy rule: a trigger file being written is not enough —
    the structural snapshot has to actually differ."""
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')

    det = ChangeDetector(tmp_path)
    before = det.snapshot()

    # Rewrite identical content (formatter run, touch, etc.)
    settings.write_text('include(":app")\n')

    assert det.should_refresh([settings]) is False
    assert det.snapshot() == before


# ── debounce ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_change_detector_debounces_burst(tmp_path: Path):
    """§9: 50 events in a burst must collapse to exactly one refresh."""
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')

    refreshes: list[str] = []
    evaluations = {"n": 0}

    det = ChangeDetector(
        tmp_path,
        debounce_s=0.05,
        on_refresh=lambda reason: refreshes.append(reason),
    )
    # Count how many times the (disk-touching) structural evaluation
    # runs, via the detector's own observability hook.
    det.on_evaluate = lambda: evaluations.__setitem__("n", evaluations["n"] + 1)

    await det.start()
    try:
        for i in range(50):
            settings.write_text(
                "\n".join(f'include(":m{n}")' for n in range(i + 1)) + "\n"
            )
            await det.notify([settings])
        await det.drain()
        # Give any NON-coalesced timers that a broken implementation
        # left running enough time to fire, so this test fails loudly
        # instead of racing past them.
        await asyncio.sleep(0.25)
    finally:
        await det.stop()

    assert len(refreshes) == 1, f"expected 1 coalesced refresh, got {refreshes}"

    # Refresh count alone is a weak assertion: the snapshot gate would
    # absorb the extra timers even with no debouncing at all. What
    # debouncing actually buys is not re-running the (filesystem-
    # touching) snapshot computation 50 times, so pin that directly.
    assert evaluations["n"] == 1, (
        f"burst was evaluated {evaluations['n']} times — debounce did not "
        f"coalesce; each evaluation re-reads the manifests from disk"
    )


@pytest.mark.asyncio
async def test_debounce_window_does_not_drop_a_later_distinct_change(
    tmp_path: Path,
):
    """Coalescing must not swallow a change that arrives after the
    window closes — that would leave the model permanently stale."""
    settings = tmp_path / "settings.gradle.kts"
    settings.write_text('include(":app")\n')

    refreshes: list[str] = []
    det = ChangeDetector(
        tmp_path,
        debounce_s=0.05,
        on_refresh=lambda reason: refreshes.append(reason),
    )
    await det.start()
    try:
        settings.write_text('include(":app")\ninclude(":a")\n')
        await det.notify([settings])
        await det.drain()

        settings.write_text('include(":app")\ninclude(":a")\ninclude(":b")\n')
        await det.notify([settings])
        await det.drain()
    finally:
        await det.stop()

    assert len(refreshes) == 2, f"second change lost: {refreshes}"


# ── snapshot economy ─────────────────────────────────────────────────


def test_snapshot_is_cheap_and_structural_only(tmp_path: Path):
    """The snapshot must describe STRUCTURE, not file contents — it is
    recomputed on every trigger event and must stay cheap."""
    (tmp_path / "settings.gradle.kts").write_text(
        'include(":app")\ninclude(":lib")\n'
    )
    rules = rules_for_project(tmp_path)
    snap = snapshot_for(tmp_path, rules)

    assert ":app" in str(snap) and ":lib" in str(snap)

    # Must not embed whole file bodies: the snapshot is recomputed on
    # every trigger event, so its size must not scale with the size of
    # the manifests it summarises.
    big = "x" * 200_000
    (tmp_path / "gradle").mkdir(exist_ok=True)
    (tmp_path / "gradle" / "libs.versions.toml").write_text(big)
    grown = snapshot_for(tmp_path, rules_for_project(tmp_path))

    assert len(str(grown)) < 4096, (
        "snapshot grew with manifest size — it must store digests, "
        "not file contents"
    )
    assert big not in str(grown)
