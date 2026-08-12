"""Unit tests for the provider staging hook + two-way manifest invariant.

Covers ``deep_worker_cli/hooks/stage_providers.py``
:

- two-way manifest invariant: eligible source set == manifest set EXACTLY
  (new / missing / extra / mismatched / case-collision / symlink-escape /
  deny violations all hard-fail);
- staging lifecycle: ``stage_providers_from_source`` payload layout,
  ``build_staging_manifest`` / ``verify_staging`` positive + corrupt/extra/
  missing hard-fails;
- extracted-mode resolution: verified ``vendor_staging`` + marker used as-is
  (no ``../../``), ambiguous half-states hard-fail;
- hook ``initialize``/``finalize`` temp lifecycle (temp lives until
  ``finalize()`` removes it).

The hook module is stdlib-only at import time (the hatchling import is
guarded), so it is loaded here via ``importlib`` without a build backend.
Fixtures use the required layout ``project_root.parent/providers`` with all
five provider packages.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# deep_worker/deep_worker_cli/tests/unit_tests/ → deep_worker_cli/
_DWC = Path(__file__).resolve().parents[2]
HOOK_PATH = _DWC / "hooks" / "stage_providers.py"

PROVIDER_PACKAGES = [
    "claude_code_provider",
    "codex_provider",
    "grok_provider",
    "semantic_search",
    "deepworker_a2a",
]


def _load_hook():
    spec = importlib.util.spec_from_file_location("stage_providers", HOOK_PATH)
    assert spec and spec.loader, f"cannot load hook module: {HOOK_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_hook()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_package(providers_root: Path, pkg: str, files: dict[str, str]) -> None:
    """Create a provider package dir with the given {relpath: content} files."""
    pkg_dir = providers_root / pkg
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        f = pkg_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")


def _build_standard_tree(root: Path) -> tuple[Path, Path]:
    """Build a canonical project_root + providers tree, return both paths.

    Layout mirrors production: ``<project_root>.parent/providers/`` with the
    five packages each holding a couple of ``*.py`` files + ``py.typed``.
    """
    providers_root = root / "providers"
    providers_root.mkdir(parents=True)
    (providers_root / "__init__.py").write_text("", encoding="utf-8")
    # `deep_worker/__init__.py` (parent of the providers tree) — staged by
    # stage_providers_from_source into payload/__init__/deep_worker.
    (root / "__init__.py").write_text("", encoding="utf-8")
    for pkg in PROVIDER_PACKAGES:
        _make_package(
            providers_root,
            pkg,
            {
                "__init__.py": f"VALUE_{pkg}\n",
                "py.typed": "",
                "alpha.py": f"def alpha_{pkg}() -> int:\n    return 1\n",
                "beta.py": f"def beta_{pkg}() -> str:\n    return '{pkg}'\n",
            },
        )
    project_root = root / "project"
    project_root.mkdir()
    return project_root, providers_root


def _write_manifest(project_root: Path, providers_root: Path) -> None:
    """Generate providers_manifest.json via the CLI --update path."""
    rc = H._cli(["--update", "--project-root", str(project_root)])
    assert rc == 0
    assert (project_root / H.PROVIDERS_MANIFEST).is_file()


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """Standard canonical tree with an up-to-date manifest."""
    project_root, providers_root = _build_standard_tree(tmp_path)
    _write_manifest(project_root, providers_root)
    return project_root, providers_root


# ---------------------------------------------------------------------------
# collect_eligible
# ---------------------------------------------------------------------------


def test_collect_eligible_includes_only_py_and_pytyped(tree: tuple[Path, Path]) -> None:
    _, providers_root = tree
    eligible = H.collect_eligible(providers_root)
    assert set(eligible) == set(PROVIDER_PACKAGES)
    for pkg in PROVIDER_PACKAGES:
        assert set(eligible[pkg]) == {
            "__init__.py",
            "py.typed",
            "alpha.py",
            "beta.py",
        }, pkg


def test_collect_eligible_excludes_non_eligible_files(tree: tuple[Path, Path]) -> None:
    _, providers_root = tree
    # README.md / .txt / docs are NOT eligible even if present.
    (providers_root / "claude_code_provider" / "README.md").write_text(
        "docs", encoding="utf-8"
    )
    (providers_root / "claude_code_provider" / "notes.txt").write_text(
        "x", encoding="utf-8"
    )
    eligible = H.collect_eligible(providers_root)
    assert "README.md" not in eligible["claude_code_provider"]
    assert "notes.txt" not in eligible["claude_code_provider"]


def test_collect_eligible_missing_package_hard_fails(tmp_path: Path) -> None:
    _, providers_root = _build_standard_tree(tmp_path)
    (providers_root / "grok_provider").rename(
        providers_root / "grok_provider_renamed"
    )
    with pytest.raises(H.ManifestError, match="provider package missing"):
        H.collect_eligible(providers_root)


def test_collect_eligible_case_collision_hard_fails(tmp_path: Path) -> None:
    _, providers_root = _build_standard_tree(tmp_path)
    (providers_root / "claude_code_provider" / "Alpha.py").write_text(
        "x", encoding="utf-8"
    )
    with pytest.raises(H.ManifestError, match="case collision"):
        H.collect_eligible(providers_root)


def test_collect_eligible_symlink_escape_hard_fails(tmp_path: Path) -> None:
    _, providers_root = _build_standard_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("x", encoding="utf-8")
    os.symlink(outside, providers_root / "claude_code_provider" / "evil.py")
    with pytest.raises(H.ManifestError, match="symlink"):
        H.collect_eligible(providers_root)


def test_collect_eligible_symlink_dir_hard_fails(tmp_path: Path) -> None:
    """A symlink DIRECTORY inside a provider package must hard-fail.

    With ``followlinks=False`` a symlink dir would otherwise be silently
    skipped — its files dropped from the eligible set — which the escape guard
    (DESIGN §3.4b) forbids.
    """
    _, providers_root = _build_standard_tree(tmp_path)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "sneaky.py").write_text("x", encoding="utf-8")
    os.symlink(
        outside_dir, providers_root / "claude_code_provider" / "evil_dir"
    )
    with pytest.raises(H.ManifestError, match="symlink directory"):
        H.collect_eligible(providers_root)


def test_collect_eligible_symlink_package_root_hard_fails(tmp_path: Path) -> None:
    """A symlinked package ROOT must hard-fail (traversal outside the tree)."""
    project_root, providers_root = _build_standard_tree(tmp_path)
    (providers_root / "grok_provider").rename(
        tmp_path / "grok_provider_real"
    )
    os.symlink(tmp_path / "grok_provider_real", providers_root / "grok_provider")
    with pytest.raises(H.ManifestError, match="symlink provider package root"):
        H.collect_eligible(providers_root)


# ---------------------------------------------------------------------------
# Two-way manifest invariant (validate_providers_manifest)
# ---------------------------------------------------------------------------


def test_manifest_check_passes_on_canonical_tree(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    H.validate_providers_manifest(providers_root, project_root / H.PROVIDERS_MANIFEST)


def test_new_source_file_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    (providers_root / "codex_provider" / "gamma.py").write_text(
        "def gamma() -> int:\n    return 3\n", encoding="utf-8"
    )
    with pytest.raises(H.ManifestError, match="new source file not in manifest"):
        H.validate_providers_manifest(
            providers_root, project_root / H.PROVIDERS_MANIFEST
        )


def test_missing_source_file_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    (providers_root / "grok_provider" / "beta.py").unlink()
    with pytest.raises(H.ManifestError, match="no source file on disk"):
        H.validate_providers_manifest(
            providers_root, project_root / H.PROVIDERS_MANIFEST
        )


def test_extra_manifest_entry_fails(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packages"]["deepworker_a2a"]["files"]["ghost.py"] = (
        "0" * 64
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="no source file on disk"):
        H.validate_providers_manifest(project_root.parent / "providers", manifest_path)


def test_hash_mismatch_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    (providers_root / "semantic_search" / "alpha.py").write_text(
        "# changed\n", encoding="utf-8"
    )
    with pytest.raises(H.ManifestError, match="hash mismatch"):
        H.validate_providers_manifest(
            providers_root, project_root / H.PROVIDERS_MANIFEST
        )


def test_unknown_package_in_manifest_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packages"]["not_a_provider"] = {"source_subpath": "x", "files": {}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="unknown package"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_schema_mismatch_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="schema mismatch"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_invalid_json_fails(tmp_path: Path) -> None:
    project_root, providers_root = _build_standard_tree(tmp_path)
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(H.ManifestError, match="not valid JSON"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_missing_manifest_fails(tmp_path: Path) -> None:
    project_root, providers_root = _build_standard_tree(tmp_path)
    with pytest.raises(H.ManifestError, match="provider manifest missing"):
        H.validate_providers_manifest(
            providers_root, project_root / H.PROVIDERS_MANIFEST
        )


# -- deny list ---------------------------------------------------------------


def test_valid_deny_entry_passes(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Remove beta.py from files + add deny with a reason.
    del manifest["packages"]["grok_provider"]["files"]["beta.py"]
    manifest["deny"]["grok_provider/beta.py"] = "approved exclusion"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    H.validate_providers_manifest(providers_root, manifest_path)


def test_deny_without_reason_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deny"]["grok_provider/beta.py"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="missing a reason comment"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_deny_in_manifest_files_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deny"]["grok_provider/beta.py"] = "reason"  # still in files
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="also present in manifest files"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_deny_non_eligible_path_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deny"]["grok_provider/ghost.py"] = "reason"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="not an eligible source file"):
        H.validate_providers_manifest(providers_root, manifest_path)


def test_deny_bad_path_shape_fails(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    manifest_path = project_root / H.PROVIDERS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deny"]["no-slash"] = "reason"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H.ManifestError, match="'<pkg>/<rel>'"):
        H.validate_providers_manifest(providers_root, manifest_path)


# ---------------------------------------------------------------------------
# Staging lifecycle
# ---------------------------------------------------------------------------


def test_stage_providers_from_source_layout(tree: tuple[Path, Path], tmp_path: Path) -> None:
    project_root, _ = tree
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    for pkg in PROVIDER_PACKAGES:
        pkg_dir = payload / pkg
        assert (pkg_dir / "__init__.py").is_file(), pkg
        assert (pkg_dir / "py.typed").is_file(), pkg
        assert (pkg_dir / "alpha.py").is_file(), pkg
    # init files staged for the canonical deep_worker package.
    assert (payload / "__init__" / "deep_worker").is_file()
    assert (payload / "__init__" / "providers").is_file()


def test_stage_providers_from_source_requires_valid_manifest(
    tmp_path: Path,
) -> None:
    project_root, providers_root = _build_standard_tree(tmp_path)
    # Drop a file WITHOUT regenerating the manifest → staging must hard-fail.
    (providers_root / "grok_provider" / "beta.py").unlink()
    with pytest.raises(H.ManifestError):
        H.stage_providers_from_source(project_root, tmp_path / "payload")


def test_build_and_verify_staging_roundtrip(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    marker = H.build_staging_manifest(payload)
    marker_path = tmp_path / "staging.manifest.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    H.verify_staging(payload, marker_path)  # must not raise


def test_verify_staging_corrupt_hash_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    marker_path = tmp_path / "staging.manifest.json"
    marker_path.write_text(
        json.dumps(H.build_staging_manifest(payload)), encoding="utf-8"
    )
    (payload / "claude_code_provider" / "alpha.py").write_text(
        "# tampered\n", encoding="utf-8"
    )
    with pytest.raises(H.ManifestError, match="hash mismatch"):
        H.verify_staging(payload, marker_path)


def test_verify_staging_extra_file_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    marker_path = tmp_path / "staging.manifest.json"
    marker_path.write_text(
        json.dumps(H.build_staging_manifest(payload)), encoding="utf-8"
    )
    (payload / "extra.py").write_text("x", encoding="utf-8")
    with pytest.raises(H.ManifestError, match="unexpected file"):
        H.verify_staging(payload, marker_path)


def test_verify_staging_missing_file_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    marker_path = tmp_path / "staging.manifest.json"
    marker_path.write_text(
        json.dumps(H.build_staging_manifest(payload)), encoding="utf-8"
    )
    (payload / "grok_provider" / "beta.py").unlink()
    with pytest.raises(H.ManifestError, match="staged file missing"):
        H.verify_staging(payload, marker_path)


# ---------------------------------------------------------------------------
# Extracted-mode resolution (_resolve_payload)
# ---------------------------------------------------------------------------


def _make_hook(root: Path, target: str):
    hook = object.__new__(H.StageProvidersHook)
    hook.root = str(root)
    hook.target_name = target
    return hook


def _build_data() -> dict:
    """Hatchling pre-seeds ``force_include``; mirror that here."""
    return {"force_include": {}}


def test_resolve_payload_extracted_mode_verified(tmp_path: Path) -> None:
    """vendor_staging + valid marker → payload used as-is (temp is None)."""
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    # Simulate sdist extraction: payload + marker next to project root.
    extracted = project_root / H.VENDOR_STAGING
    payload.rename(extracted)
    marker = H.build_staging_manifest(extracted)
    (project_root / H.STAGING_MANIFEST).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    hook = _make_hook(project_root, "wheel")
    got_payload, got_marker, temp = hook._resolve_payload(project_root)
    assert got_payload == extracted
    assert got_marker == project_root / H.STAGING_MANIFEST
    assert temp is None  # extracted mode never creates a temp dir


def test_resolve_payload_extracted_corrupt_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    payload = tmp_path / "payload"
    H.stage_providers_from_source(project_root, payload)
    extracted = project_root / H.VENDOR_STAGING
    payload.rename(extracted)
    marker = H.build_staging_manifest(extracted)
    (project_root / H.STAGING_MANIFEST).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    (extracted / "deepworker_a2a" / "alpha.py").write_text(
        "# tampered\n", encoding="utf-8"
    )
    hook = _make_hook(project_root, "wheel")
    with pytest.raises(H.ManifestError, match="hash mismatch"):
        hook._resolve_payload(project_root)


def test_resolve_payload_half_state_vendor_only_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    (project_root / H.VENDOR_STAGING).mkdir()
    hook = _make_hook(project_root, "wheel")
    with pytest.raises(H.ManifestError, match="refusing ambiguous"):
        hook._resolve_payload(project_root)


def test_resolve_payload_half_state_marker_only_fails(tmp_path: Path) -> None:
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    (project_root / H.STAGING_MANIFEST).write_text(
        "{}", encoding="utf-8"
    )
    hook = _make_hook(project_root, "wheel")
    with pytest.raises(H.ManifestError, match="refusing ambiguous"):
        hook._resolve_payload(project_root)


def test_resolve_payload_monorepo_creates_temp(tmp_path: Path) -> None:
    """Monorepo mode: unique temp dir created and tracked, marker written."""
    project_root, _ = _build_standard_tree(tmp_path)
    _write_manifest(project_root, project_root.parent / "providers")
    hook = _make_hook(project_root, "wheel")
    payload, marker, temp = hook._resolve_payload(project_root)
    assert temp is not None
    assert temp.is_dir()
    assert temp.name.startswith(H._TEMP_PREFIX)
    assert payload.is_dir()
    assert marker.is_file()
    assert temp in H._CREATED_TEMPS
    # payload lives under the temp dir.
    assert payload.parent == temp
    # hygiene: unregister + remove (mirrors finalize()).
    H._CREATED_TEMPS.discard(temp)
    shutil.rmtree(temp, ignore_errors=True)


def test_resolve_payload_monorepo_failure_cleans_temp(tmp_path: Path) -> None:
    """Injected staging failure → hard-fail with NO leftover temp.

    A manifest-listed source file disappears after ``--update``, so
    ``stage_providers_from_source`` fails inside ``_resolve_payload`` (monorepo
    branch); the temp dir must be removed and unregistered, never left behind.
    """
    project_root, providers_root = _build_standard_tree(tmp_path)
    _write_manifest(project_root, providers_root)
    (providers_root / "grok_provider" / "beta.py").unlink()

    temp_root = Path(tempfile.gettempdir())
    before_dirs = set(temp_root.glob(H._TEMP_PREFIX + "*"))
    before_temps = set(H._CREATED_TEMPS)

    hook = _make_hook(project_root, "wheel")
    with pytest.raises(H.ManifestError):
        hook._resolve_payload(project_root)

    assert set(temp_root.glob(H._TEMP_PREFIX + "*")) == before_dirs, (
        "staging failure must leave no dw-provider-stage.* dirs behind"
    )
    assert H._CREATED_TEMPS == before_temps, (
        "failed temp must be unregistered from _CREATED_TEMPS"
    )


# ---------------------------------------------------------------------------
# Hook initialize / finalize
# ---------------------------------------------------------------------------


def test_initialize_wheel_force_include(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    hook = _make_hook(project_root, "wheel")
    build_data = _build_data()
    hook.initialize(version="0.0.0", build_data=build_data)
    fi = build_data["force_include"]
    assert len(fi) == 7, fi
    for pkg in PROVIDER_PACKAGES:
        assert f"deep_worker/providers/{pkg}" in fi.values(), pkg
    assert "deep_worker/__init__.py" in fi.values()
    assert "deep_worker/providers/__init__.py" in fi.values()
    hook.finalize(version="0.0.0", build_data={}, artifact="x.whl")  # hygiene


def test_initialize_never_writes_source_tree_staging(tree: tuple[Path, Path]) -> None:
    """Staging lives ONLY in the temp dir, never in the tree.

    After ``initialize`` the temp payload is still alive (removal happens in
    ``finalize()``, not in ``initialize()``), and the source tree contains no
    ``vendor_staging/`` or ``staging.manifest.json``.
    """
    project_root, _ = tree
    hook = _make_hook(project_root, "wheel")
    hook.initialize(version="0.0.0", build_data=_build_data())
    temp = hook._temp
    assert temp is not None and temp.is_dir()
    assert temp.name.startswith(H._TEMP_PREFIX)
    assert (temp / "payload" / "claude_code_provider" / "__init__.py").is_file()
    # temp lives until finalize() — initialize() must NOT have removed it.
    assert not (project_root / H.VENDOR_STAGING).exists(), (
        "source tree must not gain vendor_staging/ (no source-tree staging)"
    )
    assert not (project_root / H.STAGING_MANIFEST).exists(), (
        "source tree must not gain staging.manifest.json"
    )
    hook.finalize(version="0.0.0", build_data={}, artifact="x.whl")
    assert not temp.exists(), "finalize() must remove the staging temp dir"


def test_parallel_resolve_payload_no_conflict(tmp_path: Path) -> None:
    """Two concurrent monorepo staging cycles on ONE tree.

    Each process stages into its own unique ``dw-provider-stage.*`` temp dir —
    no cross-process conflict, no source-tree staging, no leftover temps.
    """
    project_root, providers_root = _build_standard_tree(tmp_path)
    _write_manifest(project_root, providers_root)

    script = (
        "import importlib.util, shutil, sys\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('stage_providers', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "root = Path(sys.argv[2])\n"
        "hook = object.__new__(m.StageProvidersHook)\n"
        "hook.root = str(root)\n"
        "hook.target_name = 'wheel'\n"
        "payload, marker, temp = hook._resolve_payload(root)\n"
        "assert temp is not None and temp.name.startswith(m._TEMP_PREFIX)\n"
        "m.verify_staging(payload, marker)\n"
        "m._CREATED_TEMPS.discard(temp)\n"
        "shutil.rmtree(temp, ignore_errors=True)\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # keep subprocesses free of the sandbox tree
    temp_root = Path(tempfile.gettempdir())
    before = set(temp_root.glob(H._TEMP_PREFIX + "*"))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(HOOK_PATH), str(project_root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(2)
    ]
    results = [p.communicate(timeout=300) for p in procs]
    for proc, (out, err) in zip(procs, results):
        assert proc.returncode == 0, f"parallel staging failed:\n{out}\n{err}"
        assert "OK" in out

    assert not (project_root / H.VENDOR_STAGING).exists(), (
        "source tree must not gain vendor_staging/"
    )
    assert not (project_root / H.STAGING_MANIFEST).exists(), (
        "source tree must not gain staging.manifest.json"
    )
    # snapshot-compare: concurrent staging must not LEAK new temp dirs (other
    # tests may legitimately hold dw-provider-stage.* temps of their own).
    after = set(temp_root.glob(H._TEMP_PREFIX + "*"))
    assert after == before, f"parallel staging leaked temps: {sorted(after - before)}"


def test_initialize_sdist_force_include(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    hook = _make_hook(project_root, "sdist")
    build_data = _build_data()
    hook.initialize(version="0.0.0", build_data=build_data)
    fi = build_data["force_include"]
    # payload → vendor_staging, marker → staging.manifest.json, hook → hooks/
    assert H.VENDOR_STAGING in fi.values()
    assert H.STAGING_MANIFEST in fi.values()
    assert "hooks/stage_providers.py" in fi.values()


def test_initialize_unknown_target_fails(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    hook = _make_hook(project_root, "bogus")
    temp_root = Path(tempfile.gettempdir())
    before_dirs = set(temp_root.glob(H._TEMP_PREFIX + "*"))
    before_temps = set(H._CREATED_TEMPS)
    with pytest.raises(H.ManifestError, match="unexpected build target"):
        hook.initialize(version="0.0.0", build_data={})
    # The monorepo temp created by _resolve_payload must be removed and
    # unregistered IMMEDIATELY — Hatch only calls finalize() on success, so
    # any leak here would live until process exit.
    assert set(temp_root.glob(H._TEMP_PREFIX + "*")) == before_dirs, (
        "initialize failure must leave no dw-provider-stage.* dirs behind"
    )
    assert H._CREATED_TEMPS == before_temps, (
        "failed initialize must unregister its temp"
    )


def test_finalize_removes_temp_and_unregisters(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    hook = _make_hook(project_root, "wheel")
    build_data = _build_data()
    hook.initialize(version="0.0.0", build_data=build_data)
    temp = hook._temp
    assert temp is not None and temp.is_dir()
    assert temp in H._CREATED_TEMPS
    hook.finalize(version="0.0.0", build_data={}, artifact="x.whl")
    assert not temp.exists(), "finalize() must remove the staging temp dir"
    assert temp not in H._CREATED_TEMPS


def test_finalize_no_temp_is_noop(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    hook = _make_hook(project_root, "wheel")
    hook.finalize(version="0.0.0", build_data={}, artifact="x.whl")  # no temp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_check_ok(tree: tuple[Path, Path]) -> None:
    project_root, _ = tree
    assert H._cli(["--check", "--project-root", str(project_root)]) == 0


def test_cli_check_fails_on_violation(tree: tuple[Path, Path]) -> None:
    project_root, providers_root = tree
    (providers_root / "claude_code_provider" / "new_file.py").write_text(
        "x", encoding="utf-8"
    )
    assert H._cli(["--check", "--project-root", str(project_root)]) == 1


def test_cli_update_regenerates_manifest(tmp_path: Path) -> None:
    project_root, providers_root = _build_standard_tree(tmp_path)
    # No manifest yet → check fails.
    assert H._cli(["--check", "--project-root", str(project_root)]) == 1
    assert H._cli(["--update", "--project-root", str(project_root)]) == 0
    assert H._cli(["--check", "--project-root", str(project_root)]) == 0
    manifest = json.loads(
        (project_root / H.PROVIDERS_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == H.SCHEMA_VERSION
    assert manifest["packages"].keys() == set(PROVIDER_PACKAGES)
