"""Provider source resolution — single stdlib-only entry point.

Determines where each vendored provider package comes from:

- ``source`` — the repository layout (dev host): the trusted source root is
  added to ``sys.path`` once, and ``deep_worker.providers.<pkg>`` resolves
  from the in-tree ``deep_worker/providers/<pkg>`` directory.
- ``installed`` — the provider is imported from the active environment
  (wheel/site-packages); ``sys.path`` is NEVER mutated in this mode.

This module is called BEFORE any provider import. It is stdlib-only so it is
cheap to import from any startup path (``config.py``,
``agent_process.start()``, ``task_manager``, ``non_interactive.py``,
``agent_command.py``).

Resolution happens once (lazy memoize); switching modes after a provider
module is already in ``sys.modules`` raises ``RuntimeError``.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Descriptors (canonical import names — always prefixed, rev 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderDescriptor:
    pkg: str  # canonical import name: deep_worker.providers.<name>
    source_subpath: str  # import-package path relative to the source root
    import_parent: str  # path relative to source root added to sys.path


PROVIDER_DESCRIPTORS: tuple[ProviderDescriptor, ...] = (
    ProviderDescriptor(
        "deep_worker.providers.claude_code_provider",
        "deep_worker/providers/claude_code_provider",
        ".",
    ),
    ProviderDescriptor(
        "deep_worker.providers.codex_provider",
        "deep_worker/providers/codex_provider",
        ".",
    ),
    ProviderDescriptor(
        "deep_worker.providers.grok_provider",
        "deep_worker/providers/grok_provider",
        ".",
    ),
    ProviderDescriptor(
        "deep_worker.providers.semantic_search",
        "deep_worker/providers/semantic_search",
        ".",
    ),
    ProviderDescriptor(
        "deep_worker.providers.deepworker_a2a",
        "deep_worker/providers/deepworker_a2a",
        ".",
    ),
)

_DESCRIPTOR_BY_PKG = {d.pkg: d for d in PROVIDER_DESCRIPTORS}


class ProviderOrigin(Enum):
    INSTALLED = "installed"
    SOURCE = "source"


@dataclass(frozen=True)
class ProviderResolution:
    pkg: str
    mode: str  # "auto" | "installed" | "source"
    origin: ProviderOrigin
    roots: tuple[str, ...]  # sys.path prefixes added for SOURCE mode
    reason: str
    unavailable: str | None
    remedy: str | None


# ---------------------------------------------------------------------------
# Source-root discovery
# ---------------------------------------------------------------------------

_ENV_SOURCE_ROOT = "DEEP_WORKER_SOURCE_ROOT"
_ENV_PROVIDER_SOURCE = "DEEP_WORKER_PROVIDER_SOURCE"

# Marker files that identify a trusted repository layout root.
_SOURCE_ROOT_MARKERS: tuple[str, ...] = (
    "deep_worker/deep_worker_cli/pyproject.toml",
) + tuple(d.source_subpath for d in PROVIDER_DESCRIPTORS)


class SourceRootError(RuntimeError):
    """Raised when an explicitly configured source root is present but invalid.

    ``DEEP_WORKER_SOURCE_ROOT`` is a trusted declaration (DESIGN §3.1): if it
    is set but does not pass validation, resolution must fail with an
    actionable error instead of silently falling back to a nearby checkout or
    to the installed wheel.
    """


def _missing_source_markers(root: Path) -> list[str]:
    return [marker for marker in _SOURCE_ROOT_MARKERS if not (root / marker).exists()]


def _source_root_from_env() -> Path | None:
    """Return the trusted env-declared source root, or ``None`` when unset.

    Raises :class:`SourceRootError` when the variable is present but the path
    is not a valid repository root — never silently ignores the declaration.
    """
    raw = os.environ.get(_ENV_SOURCE_ROOT)
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute():
        raise SourceRootError(
            f"{_ENV_SOURCE_ROOT}={raw!r} is not an absolute path"
        )
    if not _is_valid_source_root(root):
        missing = _missing_source_markers(root)
        raise SourceRootError(
            f"{_ENV_SOURCE_ROOT}={raw!r} is not a trusted repository root "
            f"(missing marker(s): {', '.join(missing)})"
        )
    return root


def _source_root_near_deep_worker_cli() -> Path | None:
    """Locate the repo root next to the installed/resolved deep_worker_cli.

    ``deep_worker_cli/__init__.py`` lives at
    ``<tree>/deep_worker/deep_worker_cli/deep_worker_cli/__init__.py``;
    four ``parent`` hops reach ``<tree>``.
    """
    try:
        import deep_worker_cli  # noqa: PLC0415
    except Exception:
        return None
    pkg_file = getattr(deep_worker_cli, "__file__", None)
    if not pkg_file:
        return None
    root = Path(pkg_file).resolve().parent.parent.parent.parent
    return root if _is_valid_source_root(root) else None


def _is_valid_source_root(root: Path) -> bool:
    if not root.is_absolute() or not root.is_dir():
        return False
    return all((root / marker).exists() for marker in _SOURCE_ROOT_MARKERS)


def _find_source_root() -> Path | None:
    """Return the trusted source root, or ``None`` when none is declared.

    An explicitly set but invalid ``DEEP_WORKER_SOURCE_ROOT`` propagates
    :class:`SourceRootError` (no silent fallback to nearby/installed).
    """
    env_root = _source_root_from_env()
    if env_root is not None:
        return env_root
    return _source_root_near_deep_worker_cli()


# ---------------------------------------------------------------------------
# Optional heavy-extra capability probes
# ---------------------------------------------------------------------------
#
# In the wheel all 5 vendor packages are ALWAYS present, so the real gate for
# an optional capability is its heavy extra stack (e.g. llama-index for
# semantic-search). ``ensure_provider_importable`` probes these modules
# lazily via ``importlib.util.find_spec`` (no import, no side effects):
# missing extra → the resolution reports ``unavailable`` with the remedy, and
# call sites behave per contract (auto → warn-once + skip; explicit →
# CapabilityError).

_EXTRA_PROBES: dict[str, tuple[tuple[str, ...], str]] = {
    "deep_worker.providers.semantic_search": (
        ("llama_index",),
        "pip install deep-worker[semantic-search]",
    ),
}


def _apply_extra_probe(resolution: ProviderResolution) -> ProviderResolution:
    """If the provider package resolves but its optional heavy extra stack is
    missing, downgrade the resolution to ``unavailable`` with the remedy.

    Classification (DESIGN §3.3/§3.6): ``find_spec(mod) is None`` is the
    normal missing-extra case.  A ``ModuleNotFoundError`` naming exactly the
    probe module (or one of its parents) is also a missing-extra case.
    Anything else — unrelated ``ModuleNotFoundError``, ``ImportError``,
    ``ValueError``, ``AttributeError``, broken import machinery — is an
    internal defect and is re-raised, never masked as "extra not installed".

    A resolution that is already ``unavailable`` (e.g. the provider itself is
    missing) is returned unchanged — the extra probe never overwrites an
    earlier, more fundamental failure.
    """
    probe = _EXTRA_PROBES.get(resolution.pkg)
    if probe is None or resolution.unavailable is not None:
        return resolution
    modules, remedy = probe
    import importlib.util  # noqa: PLC0415

    def _missing() -> ProviderResolution:
        return ProviderResolution(
            pkg=resolution.pkg,
            mode=resolution.mode,
            origin=resolution.origin,
            roots=resolution.roots,
            reason=f"optional extra missing: {mod!r} not importable",
            unavailable=f"optional dependency {mod!r} not installed",
            remedy=remedy,
        )

    for mod in modules:
        try:
            if importlib.util.find_spec(mod) is None:
                return _missing()
        except ModuleNotFoundError as exc:
            # Only an exception that names the probe module (or a parent of
            # it) is the missing-extra case; anything else is a broken-state
            # defect and must not be downgraded.
            if exc.name and (exc.name == mod or mod.startswith(exc.name + ".")):
                return _missing()
            raise
        except (ImportError, ValueError, AttributeError):
            raise
    return resolution


# ---------------------------------------------------------------------------
# Installed-origin verification (purelib/platlib only)
# ---------------------------------------------------------------------------


def _site_roots() -> list[Path]:
    paths = sysconfig.get_paths()
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        raw = paths.get(key)
        if raw:
            roots.append(Path(raw).resolve())
    return roots


def _installed_spec_location(pkg: str) -> Path | None:
    """Return the spec origin/submodule-search-location for ``pkg`` if it
    resolves ENTIRELY under active site roots (purelib/platlib), else ``None``.

    Every applicable ``spec.origin`` / ``submodule_search_locations`` entry
    must lie under an active root (DESIGN §3.1): a namespace or package that
    mixes one site-packages entry with an outside entry (e.g. a ``/repos``
    source checkout) is a forbidden source+installed hybrid and is rejected.

    Never mutates ``sys.path``.
    """
    import importlib.util  # noqa: PLC0415

    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None:
        return None
    site_roots = _site_roots()
    candidates: list[str] = []
    origin = spec.origin
    if origin:
        candidates.append(origin)
    locs = spec.submodule_search_locations
    if locs:
        candidates.extend(str(l) for l in locs)
    if not candidates:
        return None
    resolved = [Path(cand).resolve() for cand in candidates]
    for path in resolved:
        if not any(path.is_relative_to(root) for root in site_roots):
            return None  # mixed origin: at least one entry escapes the roots
    return resolved[0]


def _verify_installed(pkg: str) -> bool:
    """Post-import check: the real module must live under purelib/platlib.

    ALL applicable origins — ``module.__file__`` and every
    ``module.__path__`` entry — must be under the active site roots.  A module
    whose ``__file__`` is under site-packages but whose ``__path__`` reaches
    outside (or vice versa) is a forbidden source+installed hybrid → ``False``.
    """
    module = sys.modules.get(pkg)
    if module is None:
        return False
    file_path = getattr(module, "__file__", None)
    path_attr = getattr(module, "__path__", None)
    site_roots = _site_roots()
    candidates: list[Path] = []
    if file_path:
        candidates.append(Path(file_path).resolve())
    if path_attr:
        candidates.extend(Path(entry).resolve() for entry in path_attr)
    if not candidates:
        return False  # nothing verifiable under the active roots
    return all(
        any(p.is_relative_to(root) for root in site_roots) for p in candidates
    )


# ---------------------------------------------------------------------------
# Resolution (once, memoized)
# ---------------------------------------------------------------------------

_resolved: dict[str, ProviderResolution] = {}


def _mode_from_env() -> str | None:
    raw = os.environ.get(_ENV_PROVIDER_SOURCE)
    if raw in ("source", "installed"):
        return raw
    return None


def resolve_provider(pkg: str, mode: str = "auto") -> ProviderResolution:
    """Resolve the origin of a provider package.

    ``mode``: "auto" (default), "installed", or "source". Resolution is
    memoized per package; changing the effective origin after the module is
    already in ``sys.modules`` raises ``RuntimeError``.
    """
    if pkg not in _DESCRIPTOR_BY_PKG:
        raise ValueError(f"unknown provider package: {pkg!r}")
    cached = _resolved.get(pkg)
    if cached is not None:
        _assert_mode_stable(cached, pkg, mode)
        if pkg in sys.modules:
            return _post_import_checked(cached)
        return cached

    effective = mode
    if mode == "auto":
        effective = _mode_from_env() or "auto"

    if effective == "source":
        resolution = _resolve_source(pkg)
    elif effective == "installed":
        resolution = _resolve_installed(pkg)
    else:  # auto
        resolution = _resolve_auto(pkg)

    _resolved[pkg] = resolution
    if pkg in sys.modules:
        return _post_import_checked(resolution)
    return resolution


def _assert_mode_stable(resolution: ProviderResolution, pkg: str, mode: str) -> None:
    if pkg not in sys.modules:
        # The provider is not imported yet; the origin may still change.
        return
    effective = mode
    if mode == "auto":
        effective = _mode_from_env() or "auto"
    if resolution.mode != effective:
        # The provider is already imported under its canonical name; the
        # origin cannot change any more.
        raise RuntimeError(
            f"provider {pkg!r} is already imported and resolved as "
            f"{resolution.mode!r}; refusing mode switch to {effective!r}"
        )


def _post_import_checked(resolution: ProviderResolution) -> ProviderResolution:
    """Post-import origin verification (rev 4 precision-fix).

    Once the module is actually imported, its physical location must lie under
    the active purelib/platlib in INSTALLED mode — otherwise the origin is
    unreliable (forbidden source+installed mixing) and the provider is
    reported as unavailable.
    """
    if resolution.origin is ProviderOrigin.INSTALLED and not _verify_installed(resolution.pkg):
        return ProviderResolution(
            pkg=resolution.pkg,
            mode=resolution.mode,
            origin=resolution.origin,
            roots=(),
            reason="post-import check failed: module not under active site roots",
            unavailable="provider origin mismatch",
            remedy="install deep-worker (wheel) or point DEEP_WORKER_SOURCE_ROOT "
            "at the repository root",
        )
    return resolution


def _resolve_source(pkg: str) -> ProviderResolution:
    desc = _DESCRIPTOR_BY_PKG[pkg]
    try:
        root = _find_source_root()
    except SourceRootError as exc:
        return ProviderResolution(
            pkg=pkg,
            mode="source",
            origin=ProviderOrigin.SOURCE,
            roots=(),
            reason=str(exc),
            unavailable=f"invalid {_ENV_SOURCE_ROOT}",
            remedy=f"unset {_ENV_SOURCE_ROOT} or point it at the repository root",
        )
    if root is None:
        return ProviderResolution(
            pkg=pkg,
            mode="source",
            origin=ProviderOrigin.SOURCE,
            roots=(),
            reason="source mode requested but no trusted source root found",
            unavailable="source root not found",
            remedy=f"set {_ENV_SOURCE_ROOT} to the repository root",
        )
    return ProviderResolution(
        pkg=pkg,
        mode="source",
        origin=ProviderOrigin.SOURCE,
        roots=(str(root),),
        reason=f"source root {root}",
        unavailable=None,
        remedy=None,
    )


def _resolve_installed(pkg: str) -> ProviderResolution:
    loc = _installed_spec_location(pkg)
    if loc is None:
        return ProviderResolution(
            pkg=pkg,
            mode="installed",
            origin=ProviderOrigin.INSTALLED,
            roots=(),
            reason="provider not found under active site roots",
            unavailable="provider not installed",
            remedy="install deep-worker (wheel) or point DEEP_WORKER_SOURCE_ROOT "
            "at the repository root",
        )
    return ProviderResolution(
        pkg=pkg,
        mode="installed",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason=f"installed at {loc}",
        unavailable=None,
        remedy=None,
    )


def _resolve_auto(pkg: str) -> ProviderResolution:
    try:
        root = _find_source_root()
    except SourceRootError as exc:
        # An explicitly set but invalid root is an error, never a fallback to
        # the nearby checkout or the installed wheel (DESIGN §3.1).
        return ProviderResolution(
            pkg=pkg,
            mode="auto",
            origin=ProviderOrigin.SOURCE,
            roots=(),
            reason=str(exc),
            unavailable=f"invalid {_ENV_SOURCE_ROOT}",
            remedy=f"unset {_ENV_SOURCE_ROOT} or point it at the repository root",
        )
    if root is not None:
        return ProviderResolution(
            pkg=pkg,
            mode="auto",
            origin=ProviderOrigin.SOURCE,
            roots=(str(root),),
            reason=f"full repository layout detected at {root}",
            unavailable=None,
            remedy=None,
        )
    loc = _installed_spec_location(pkg)
    if loc is not None:
        return ProviderResolution(
            pkg=pkg,
            mode="auto",
            origin=ProviderOrigin.INSTALLED,
            roots=(),
            reason=f"installed at {loc}",
            unavailable=None,
            remedy=None,
        )
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="neither source layout nor installed provider found",
        unavailable="provider unavailable",
        remedy="install deep-worker or set DEEP_WORKER_SOURCE_ROOT",
    )


def ensure_provider_importable(pkg: str, mode: str = "auto") -> ProviderResolution:
    """Resolve ``pkg`` and, for SOURCE origin, idempotently put the source
    root on ``sys.path`` so ``import deep_worker.providers.<name>`` works.

    Returns the resolution (with ``unavailable``/``remedy`` populated on
    failure). Callers decide how to surface the failure (log-once vs
    CapabilityError). For providers with an optional heavy extra
    (``_EXTRA_PROBES``), availability also requires the extra stack —
    regardless of the package origin.
    """
    resolution = resolve_provider(pkg, mode=mode)
    resolution = _apply_extra_probe(resolution)
    if resolution.origin is ProviderOrigin.SOURCE:
        for root in resolution.roots:
            if root and root not in sys.path:
                sys.path.insert(0, root)
    return resolution


def ensure_source_root_on_path() -> bool:
    """Ensure the source root is on ``sys.path`` (source mode only).

    Returns ``True`` when a trusted source root was found and is on
    ``sys.path``; ``False`` when no source root exists (installed-only
    environment — callers should use ``installed`` resolution).  An explicitly
    set but invalid ``DEEP_WORKER_SOURCE_ROOT`` raises :class:`SourceRootError`
    instead of silently degrading to installed.
    """
    root = _find_source_root()
    if root is None:
        return False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return True


__all__ = [
    "PROVIDER_DESCRIPTORS",
    "ProviderDescriptor",
    "ProviderOrigin",
    "ProviderResolution",
    "SourceRootError",
    "ensure_provider_importable",
    "ensure_source_root_on_path",
    "resolve_provider",
]
