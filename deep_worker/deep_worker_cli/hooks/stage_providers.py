"""Hatchling build hook that stages the five vendored provider packages.

The deep-worker wheel and sdist ship the in-tree providers
(``deep_worker/providers/<pkg>``) via a custom build hook — there are NO
static force-include entries for providers in ``pyproject.toml``.  This module
is that hook, and it is also the maintainer of the two-way provider manifest
invariant (``providers_manifest.json``).

Design contract:

* Monorepo builds ALWAYS regenerate staging from the current sources into a
  UNIQUE temp directory (``tempfile.mkdtemp(prefix="dw-provider-stage.")``),
  never into the source tree.  The temp lives until Hatch has consumed it —
  removal happens in ``finalize()``, not in ``initialize()``.
* sdist-extracted rebuilds are detected by the explicit integrity marker
  ``staging.manifest.json`` (SHA-256 per payload file) sitting NEXT TO the
  ``vendor_staging/`` payload.  If the marker is present and every hash
  matches (and there are no extra/missing files) the payload is used as-is —
  no access to ``../../``.  Any mismatch is a hard fail, never a silent
  regeneration.
* ``providers_manifest.json`` is a tracked, two-way invariant: the eligible
  source set (every production ``*.py`` + ``py.typed`` under each of the five
  provider packages, minus the explicit deny list) must EXACTLY equal the
  manifest set.  New / missing / extra / mismatched / case-collision /
  symlink-escape entries all hard-fail.

The module is stdlib-only at import time (the hatchling import is guarded), so
the same file doubles as a pure-stdlib CLI for the pre-commit hook and tests:

    python3 hooks/stage_providers.py --check   # validate the invariant
    python3 hooks/stage_providers.py --update  # regenerate the manifest

(NB: the design text suggests ``python -m deep_worker_cli.tools.
check_provider_manifest`` as an alternative mechanism; that module path is
unavailable because ``deep_worker_cli.tools`` is an existing module, so the
checker lives here — "the same hook" branch of the design.)
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

try:  # hatchling is present only inside a build (isolation / uv build)
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # pragma: no cover - dev / pre-commit / test context
    BuildHookInterface = object  # type: ignore[assignment,misc]

PROJECT_NAME = "deep-worker"
SCHEMA_VERSION = 1

# The vendored provider packages (canonical import parent:
# ``deep_worker.providers.<pkg>``). Order is stable — it defines the
# manifest's package order. ``acp`` is the shared ACP layer; the provider
# packages import from it (claude_code_provider imports ..acp.*).
PROVIDER_PACKAGES = [
    "codex_provider",
    "grok_provider",
    "semantic_search",
    "deepworker_a2a",
    "acp",
    "claude_code_provider",
]

# Tracked invariant manifest (project root) and the per-build integrity marker
# shipped next to the sdist payload.
PROVIDERS_MANIFEST = "providers_manifest.json"
STAGING_MANIFEST = "staging.manifest.json"
VENDOR_STAGING = "vendor_staging"

# ---------------------------------------------------------------------------
# Web payload.
#
# ``deep_worker_web`` is the web UI server. It must ship inside the
# deep-worker wheel, because a user who installed from PyPI has no sources and
# `deep-worker web` has to work for them.
#
# It is staged exactly like the providers — copied from OUTSIDE the hatch root
# into a unique temp dir, covered by a sha256 marker — and for one specific
# reason it is NOT moved into the package tree instead: three path expressions
# in the web code derive the checkout root by counting directory levels
#
#   app/config.py                 Path(__file__).resolve().parents[2]
#   app/routers/agents.py         Path(__file__).resolve().parent x4
#   app/services/agent_service.py Path(__file__).resolve().parent x4
#
# and moving the files changes that count. The result is not an exception but
# a DIFFERENT yet existing directory (…/deep_worker_cli instead of the tree
# root), so the failure is silent: ``.exists()`` turns False somewhere far
# from the cause. Staging keeps the source tree byte-for-byte identical, which
# is what lets this host keep running deep_worker from source unchanged.
WEB_SOURCE_DIRNAME = "deep_worker_web"
WEB_STAGING = "web_staging"
WEB_DEST = "deep_worker_cli/web"

# Eligible web payload = every production ``*.py`` under ``app/`` plus every
# asset under ``static/``. ``static/`` ships wholesale (html/css/js and any
# future asset type), so it is matched by subtree, not by suffix.
WEB_CODE_SUBDIR = "app"
WEB_ASSET_SUBDIR = "static"

_TEMP_PREFIX = "dw-provider-stage."
_CREATED_TEMPS: set[Path] = set()


class ManifestError(RuntimeError):
    """Raised on any provider-manifest / staging invariant violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_payload(payload: Path) -> dict[str, str]:
    """Map every regular file under ``payload`` to its SHA-256 (posix relpath).

    Hard-fails on symlinks — both symlink files AND symlink directories
    (escape guard) — so a payload can never reach outside its root through a
    link, and a symlink dir is never silently skipped.
    """
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(payload, followlinks=False):
        dirpath_p = Path(dirpath)
        if dirpath_p.is_symlink():
            raise ManifestError(
                f"symlink directory inside payload (escape guard): {dirpath_p}"
            )
        for d in list(dirnames):
            full_d = dirpath_p / d
            if full_d.is_symlink():
                raise ManifestError(
                    f"symlink directory inside payload (escape guard): {full_d}"
                )
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            full = dirpath_p / name
            if full.is_symlink():
                raise ManifestError(f"symlink inside payload (escape guard): {full}")
            rel = full.relative_to(payload).as_posix()
            files[rel] = _sha256(full)
    return files


def collect_eligible(providers_root: Path) -> dict[str, dict[str, str]]:
    """Return the eligible source set: ``{pkg: {relpath: sha256}}``.

    Eligible = every production ``*.py`` plus the mandatory ``py.typed`` under
    each of the five provider packages.  Non-eligible files (``__pycache__``,
    docs, tests, …) are not part of the set by definition.  Hard-fails on a
    missing package, symlinks (escape guard) or case collisions.
    """
    result: dict[str, dict[str, str]] = {}
    for pkg in PROVIDER_PACKAGES:
        pkg_root = providers_root / pkg
        if not pkg_root.is_dir():
            raise ManifestError(f"provider package missing: {pkg_root}")
        if pkg_root.is_symlink():
            raise ManifestError(
                f"symlink provider package root (escape guard): {pkg_root}"
            )
        files: dict[str, str] = {}
        seen_lower: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(pkg_root, followlinks=False):
            dirpath_p = Path(dirpath)
            if dirpath_p.is_symlink():
                raise ManifestError(
                    f"symlink directory in provider package (escape guard): {dirpath_p}"
                )
            # Reject symlink dirs explicitly — with followlinks=False a symlink
            # dir would otherwise be silently skipped (its files dropped from
            # the eligible set), which the escape guard forbids.
            for d in list(dirnames):
                full_d = dirpath_p / d
                if full_d.is_symlink():
                    raise ManifestError(
                        f"symlink directory in provider package (escape guard): {full_d}"
                    )
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in sorted(filenames):
                full = dirpath_p / name
                if full.is_symlink():
                    raise ManifestError(
                        f"symlink in provider package (escape guard): {full}"
                    )
                if not (name.endswith(".py") or name == "py.typed"):
                    continue
                rel = full.relative_to(pkg_root).as_posix()
                lower = rel.lower()
                if lower in seen_lower and seen_lower[lower] != rel:
                    raise ManifestError(
                        f"case collision in {pkg}: {seen_lower[lower]} vs {rel}"
                    )
                seen_lower[lower] = rel
                files[rel] = _sha256(full)
        result[pkg] = files
    return result


def collect_web_eligible(web_root: Path) -> dict[str, str]:
    """Return the eligible web payload set: ``{relpath: sha256}``.

    Eligible = every ``*.py`` under ``app/`` plus every file under
    ``static/``.  ``__pycache__`` is skipped; tree-only files
    (``requirements.txt``, ``pytest.ini``, docs, tests) are not part of the
    shipped set by definition.

    Hard-fails on a missing tree and on symlinks — both symlink files AND
    symlink directories (escape guard) — so the payload can never reach
    outside ``web_root`` through a link, and a symlink dir is never silently
    skipped.  Same guarantees as :func:`collect_eligible` for providers.
    """
    if not web_root.is_dir():
        raise ManifestError(f"web tree missing: {web_root}")
    if web_root.is_symlink():
        raise ManifestError(f"symlink web tree root (escape guard): {web_root}")

    files: dict[str, str] = {}
    seen_lower: dict[str, str] = {}
    for subdir, py_only in ((WEB_CODE_SUBDIR, True), (WEB_ASSET_SUBDIR, False)):
        sub_root = web_root / subdir
        if not sub_root.is_dir():
            raise ManifestError(f"web payload subtree missing: {sub_root}")
        if sub_root.is_symlink():
            raise ManifestError(
                f"symlink directory in web tree (escape guard): {sub_root}"
            )
        for dirpath, dirnames, filenames in os.walk(sub_root, followlinks=False):
            dirpath_p = Path(dirpath)
            if dirpath_p.is_symlink():
                raise ManifestError(
                    f"symlink directory in web tree (escape guard): {dirpath_p}"
                )
            for d in list(dirnames):
                full_d = dirpath_p / d
                if full_d.is_symlink():
                    raise ManifestError(
                        f"symlink directory in web tree (escape guard): {full_d}"
                    )
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in sorted(filenames):
                full = dirpath_p / name
                if full.is_symlink():
                    raise ManifestError(
                        f"symlink in web tree (escape guard): {full}"
                    )
                if py_only and not name.endswith(".py"):
                    continue
                rel = full.relative_to(web_root).as_posix()
                lower = rel.lower()
                if lower in seen_lower and seen_lower[lower] != rel:
                    raise ManifestError(
                        f"case collision in web payload: "
                        f"{seen_lower[lower]} vs {rel}"
                    )
                seen_lower[lower] = rel
                files[rel] = _sha256(full)
    return files


def _web_root_for(project_root: Path) -> Path:
    """Return the ``deep_worker_web`` tree for a given hatch project root.

    ``project_root`` is ``<tree>/deep_worker/deep_worker_cli``, so the web tree
    is its grandparent's child — the same relative position the providers tree
    occupies one level up.
    """
    return project_root.parent.parent / WEB_SOURCE_DIRNAME


def stage_web_from_source(project_root: Path, payload: Path) -> None:
    """Copy the eligible web payload from the sources into ``payload``.

    The source tree is only READ: every write goes to ``payload`` (a unique
    temp dir in monorepo builds).  That is what keeps this host's from-source
    launch working untouched.
    """
    web_root = _web_root_for(project_root)
    files = collect_web_eligible(web_root)
    payload.mkdir(parents=True, exist_ok=False)
    _stage_files(web_root, payload, sorted(files))


def build_providers_manifest(providers_root: Path) -> dict[str, Any]:
    """Build a manifest document for the current eligible set."""
    packages = collect_eligible(providers_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "_comment": (
            "Two-way provider manifest invariant: the eligible source set "
            "— every *.py plus py.typed under "
            "deep_worker/providers/<pkg>, minus the deny list — must EXACTLY "
            "equal the files below. Regenerate with "
            "`python3 hooks/stage_providers.py --update` after any provider "
            "source change."
        ),
        "packages": {
            pkg: {
                "source_subpath": f"deep_worker/providers/{pkg}",
                "files": files,
            }
            for pkg, files in packages.items()
        },
        "deny": {},
    }


def validate_providers_manifest(providers_root: Path, manifest_path: Path) -> None:
    """Enforce the two-way invariant: eligible set == manifest set exactly.

    Raises :class:`ManifestError` on any violation (never returns a status).
    """
    if not manifest_path.is_file():
        raise ManifestError(f"provider manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"provider manifest is not valid JSON: {manifest_path}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"provider manifest schema mismatch: {manifest_path} "
            f"(expected {SCHEMA_VERSION}, got {manifest.get('schema_version')!r})"
        )

    eligible = collect_eligible(providers_root)
    deny = manifest.get("deny", {})
    if not isinstance(deny, dict):
        raise ManifestError("provider manifest 'deny' must be an object")

    errors: list[str] = []
    for pkg, expected_files in eligible.items():
        pkg_entry = manifest.get("packages", {}).get(pkg)
        if not isinstance(pkg_entry, dict):
            errors.append(f"manifest missing package entry: {pkg}")
            continue
        manifest_files = pkg_entry.get("files", {})
        if not isinstance(manifest_files, dict):
            errors.append(f"manifest package entry '{pkg}' has no files object")
            continue
        for rel, sha in expected_files.items():
            deny_reason = deny.get(f"{pkg}/{rel}")
            if deny_reason is not None:
                continue  # explicitly approved exclusion
            if rel not in manifest_files:
                errors.append(f"new source file not in manifest: {pkg}/{rel}")
            elif manifest_files[rel] != sha:
                errors.append(f"hash mismatch for {pkg}/{rel} (run --update)")
        for rel in manifest_files:
            if rel not in expected_files:
                errors.append(f"manifest entry has no source file on disk: {pkg}/{rel}")

    for pkg in manifest.get("packages", {}):
        if pkg not in PROVIDER_PACKAGES:
            errors.append(f"manifest has unknown package: {pkg}")

    for denied in deny:
        parts = denied.split("/", 1)
        if len(parts) != 2 or parts[0] not in PROVIDER_PACKAGES:
            errors.append(f"deny path must be '<pkg>/<rel>': {denied!r}")
            continue
        pkg, rel = parts
        eligible_file = eligible.get(pkg, {}).get(rel)
        if eligible_file is None:
            errors.append(f"deny path is not an eligible source file: {denied}")
        elif rel in manifest.get("packages", {}).get(pkg, {}).get("files", {}):
            errors.append(f"deny path also present in manifest files: {denied}")
        if not deny[denied]:
            errors.append(f"deny path missing a reason comment: {denied}")

    if errors:
        raise ManifestError(
            "provider manifest invariant violated:\n  " + "\n  ".join(sorted(errors))
        )


def _stage_files(src_root: Path, dst_root: Path, relpaths: list[str]) -> None:
    for rel in relpaths:
        src = src_root / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage_providers_from_source(project_root: Path, payload: Path) -> None:
    """Regenerate the full staging payload from the current sources.

    Validates the tracked manifest first (hard-fail on invariant violation),
    then copies exactly the manifest-listed files into ``payload``:

        payload/<pkg>/...                     the five provider packages
        payload/__init__/deep_worker          deep_worker/__init__.py
        payload/__init__/providers            deep_worker/providers/__init__.py
    """
    providers_root = project_root.parent / "providers"
    manifest_path = project_root / PROVIDERS_MANIFEST
    validate_providers_manifest(providers_root, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    payload.mkdir(parents=True, exist_ok=False)
    for pkg in PROVIDER_PACKAGES:
        relpaths = sorted(manifest["packages"][pkg]["files"])
        _stage_files(providers_root / pkg, payload / pkg, relpaths)

    init_root = payload / "__init__"
    init_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(project_root.parent / "__init__.py", init_root / "deep_worker")
    shutil.copy2(providers_root / "__init__.py", init_root / "providers")


def build_staging_manifest(payload: Path) -> dict[str, Any]:
    """Build the per-build integrity marker for a staged payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "files": _walk_payload(payload),
    }


def verify_staging(payload: Path, manifest_path: Path) -> None:
    """Verify an existing payload against its staging manifest (hard-fail)."""
    if not payload.is_dir():
        raise ManifestError(f"staging payload missing: {payload}")
    if not manifest_path.is_file():
        raise ManifestError(f"staging manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"staging manifest is not valid JSON: {manifest_path}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"staging manifest schema mismatch: {manifest_path} "
            f"(expected {SCHEMA_VERSION}, got {manifest.get('schema_version')!r})"
        )
    expected = manifest.get("files", {})
    if not isinstance(expected, dict):
        raise ManifestError("staging manifest 'files' must be an object")

    actual = _walk_payload(payload)
    errors: list[str] = []
    for rel, sha in sorted(expected.items()):
        if rel not in actual:
            errors.append(f"staged file missing: {rel}")
        elif actual[rel] != sha:
            errors.append(f"staged file hash mismatch: {rel}")
    for rel in sorted(actual):
        if rel not in expected:
            errors.append(f"unexpected file in payload: {rel}")
    if errors:
        raise ManifestError(
            "staging verification failed:\n  " + "\n  ".join(errors)
        )


def _emergency_cleanup() -> None:
    for temp in list(_CREATED_TEMPS):
        shutil.rmtree(temp, ignore_errors=True)


atexit.register(_emergency_cleanup)


class StageProvidersHook(BuildHookInterface):
    """Custom hatchling hook wiring the provider payload into both targets.

    Registered via ``[tool.hatch.build.hooks.custom] path = "hooks/stage_providers.py"``.
    """

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        payload, staging_manifest_path, temp = self._resolve_payload(root)
        self._temp = temp  # kept until finalize(); temp lives until Hatch consumes it

        # Any failure after the temp is created (unknown target, mapping
        # errors, …) must clean it up IMMEDIATELY — Hatch calls finalize()
        # only on success, so a leak here would live until process exit.
        try:
            web_payload = self._resolve_web_payload(root, temp)
            if self.target_name == "wheel":
                for pkg in PROVIDER_PACKAGES:
                    build_data["force_include"][str(payload / pkg)] = (
                        f"deep_worker/providers/{pkg}"
                    )
                build_data["force_include"][str(payload / "__init__" / "deep_worker")] = (
                    "deep_worker/__init__.py"
                )
                build_data["force_include"][str(payload / "__init__" / "providers")] = (
                    "deep_worker/providers/__init__.py"
                )
                # Web UI: app/ + static/ land under deep_worker_cli/web/ so a
                # pip user gets `deep-worker web` with no sources. Mapped as
                # two subtrees (not file-by-file) to keep force_include small.
                build_data["force_include"][str(web_payload / WEB_CODE_SUBDIR)] = (
                    f"{WEB_DEST}/{WEB_CODE_SUBDIR}"
                )
                build_data["force_include"][str(web_payload / WEB_ASSET_SUBDIR)] = (
                    f"{WEB_DEST}/{WEB_ASSET_SUBDIR}"
                )
            elif self.target_name == "sdist":
                if temp is not None:
                    # Monorepo: ship the staged payload + its integrity marker so a
                    # later wheel rebuild from this sdist is self-sufficient.
                    build_data["force_include"][str(payload)] = VENDOR_STAGING
                    build_data["force_include"][str(staging_manifest_path)] = STAGING_MANIFEST
                    # Same for the web payload — without it an sdist rebuild
                    # would have to reach ../../deep_worker_web, which is
                    # exactly what extracted mode must never do.
                    build_data["force_include"][str(web_payload)] = WEB_STAGING
                # In extracted mode both already live inside the project root and
                # the default sdist file selection picks them up.
                build_data["force_include"][str(root / "hooks" / "stage_providers.py")] = (
                    "hooks/stage_providers.py"
                )
            else:
                raise ManifestError(f"unexpected build target: {self.target_name!r}")
        except BaseException:
            if temp is not None:
                _CREATED_TEMPS.discard(temp)
                shutil.rmtree(temp, ignore_errors=True)
                self._temp = None
            raise

    def finalize(self, version: str, build_data: dict[str, Any], artifact: str) -> None:
        temp = getattr(self, "_temp", None)
        if temp is not None:
            _CREATED_TEMPS.discard(temp)
            shutil.rmtree(temp, ignore_errors=True)

    # -- internals ----------------------------------------------------------

    def _resolve_web_payload(self, root: Path, temp: Path | None) -> Path:
        """Return the staged web payload directory.

        * monorepo (``temp is not None``): staged fresh next to the provider
          payload by :meth:`_resolve_payload`.
        * sdist-extracted (``temp is None``): ``web_staging/`` shipped inside
          the sdist, used as-is — never reaching ``../../`` for sources that
          an extracted tree does not have.
        """
        if temp is not None:
            return temp / "web"
        extracted = root / WEB_STAGING
        if not extracted.is_dir():
            raise ManifestError(
                f"{WEB_STAGING} missing from extracted sdist: {extracted}\n"
                "The web payload must ship inside the sdist; a rebuild cannot "
                "fall back to the source tree."
            )
        return extracted

    def _resolve_payload(self, root: Path) -> tuple[Path, Path, Path | None]:
        """Return ``(payload, staging_manifest_path, temp_or_None)``.

        * sdist-extracted: ``vendor_staging/`` + ``staging.manifest.json``
          present and verified → use as-is (``temp is None``).
        * Ambiguous half-states (one of the two without the other) → hard fail.
        * monorepo: stage fresh into a unique temp dir (``temp is not None``).
        """
        vendored = root / VENDOR_STAGING
        manifest = root / STAGING_MANIFEST
        if vendored.is_dir() and manifest.is_file():
            verify_staging(vendored, manifest)
            return vendored, manifest, None
        if vendored.is_dir():
            raise ManifestError(
                f"{VENDOR_STAGING} exists without {STAGING_MANIFEST}: "
                "refusing ambiguous payload (leftover sdist extraction?)"
            )
        if manifest.is_file():
            raise ManifestError(
                f"{STAGING_MANIFEST} exists without {VENDOR_STAGING}: "
                "refusing ambiguous payload"
            )

        temp = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
        _CREATED_TEMPS.add(temp)
        try:
            payload = temp / "payload"
            stage_providers_from_source(root, payload)
            # The web payload is staged into the SAME temp, under its own
            # subdir, so one temp lifecycle covers both and finalize() needs
            # no extra bookkeeping.
            stage_web_from_source(root, temp / "web")
            staging_manifest = build_staging_manifest(payload)
            (temp / STAGING_MANIFEST).write_text(
                json.dumps(staging_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return payload, temp / STAGING_MANIFEST, temp
        except BaseException:
            _CREATED_TEMPS.discard(temp)
            shutil.rmtree(temp, ignore_errors=True)
            raise


# ---------------------------------------------------------------------------
# Pure-stdlib CLI (pre-commit hook + tests): python3 hooks/stage_providers.py
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stage_providers",
        description="Manage the deep-worker provider manifest / staging invariant.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Hatch project root (default: this file's parent.parent).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate providers_manifest.json two-way invariant (default).",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="Regenerate providers_manifest.json from the current sources.",
    )
    args = parser.parse_args(argv)

    project_root = (
        Path(args.project_root)
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    providers_root = project_root.parent / "providers"
    manifest_path = project_root / PROVIDERS_MANIFEST

    if args.update:
        document = build_providers_manifest(providers_root)
        manifest_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        total = sum(len(pkg["files"]) for pkg in document["packages"].values())
        print(f"updated {manifest_path} ({total} files)")
        return 0

    try:
        validate_providers_manifest(providers_root, manifest_path)
    except ManifestError as exc:
        print(f"provider manifest check FAILED: {exc}", file=sys.stderr)
        return 1
    print("provider manifest OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
