"""Event-driven change detection for the shared LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §9. This module answers one
question as cheaply as possible:

    given a filesystem event, does the PROJECT STRUCTURE differ from
    what the language server was last told about?

Only a structural difference justifies a refresh. Editing the body of
an existing source file does not — the language server already learns
about that through ``didChangeWatchedFiles``, and refreshing the whole
project model on every keystroke is exactly the waste this design
exists to avoid.

Design shape:

* **Provider rules** are selected by marker files in the project root
  (``settings.gradle.kts`` → Gradle, ``pyproject.toml`` → Python, …).
  Several can be active at once for polyglot repos; each watches its
  own paths and refreshes independently. The ``default`` rule set is a
  fallback and never co-fires with a specific provider (that would
  cause a double refresh).
* **Structural snapshot** is a small dict per provider — module
  includes, workspace members, a hash of the dependency manifest. It
  never embeds whole file bodies, because it is recomputed on every
  trigger event.
* **Debounce** coalesces a burst of saves into one refresh.

Everything here is event-driven: :meth:`ChangeDetector.notify` is
called by the watcher, and the debounce timer is an ``asyncio`` task
that sleeps exactly once per burst. There is no polling loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Sequence

#: §9: coalesce a burst of saves into a single refresh.
DEFAULT_DEBOUNCE_S: Final = 0.3

#: Source extensions whose *content* edits never imply a structural
#: change. Listed explicitly rather than "anything not a trigger" so a
#: new manifest type cannot be silently swallowed.
_SOURCE_SUFFIXES: Final = frozenset(
    {
        ".kt", ".kts", ".java", ".scala", ".groovy",
        ".py", ".pyi",
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".rs", ".go", ".c", ".h", ".cc", ".cpp", ".hpp",
    }
)
# ``.kts`` is a source suffix AND a manifest suffix (build.gradle.kts).
# Trigger matching runs first, so manifests win — see ``_is_trigger``.


@dataclass(frozen=True)
class ProviderRules:
    """One provider's watch/snapshot contract (§9 table)."""

    name: str
    markers: tuple[str, ...]
    trigger_globs: tuple[str, ...]
    refresh_action: str

    def matches_marker(self, root: Path) -> bool:
        return any((root / m).exists() for m in self.markers)


#: The §9 provider table. Order is stable so ``rules_for_project``
#: returns a deterministic list.
PROVIDERS: Final[tuple[ProviderRules, ...]] = (
    ProviderRules(
        name="kotlin-gradle",
        markers=("settings.gradle.kts", "settings.gradle"),
        trigger_globs=(
            "settings.gradle.kts",
            "settings.gradle",
            "**/build.gradle.kts",
            "**/build.gradle",
            "gradle.properties",
            "gradle/libs.versions.toml",
            "gradle/wrapper/gradle-wrapper.properties",
        ),
        refresh_action="serena.refresh_project+didChangeWorkspaceFolders",
    ),
    ProviderRules(
        name="maven",
        markers=("pom.xml",),
        trigger_globs=("pom.xml", "**/pom.xml", ".mvn/**"),
        refresh_action="serena.refresh_project+didChangeWorkspaceFolders",
    ),
    ProviderRules(
        name="python",
        markers=("pyproject.toml", "setup.py"),
        trigger_globs=(
            "pyproject.toml",
            "uv.lock",
            "poetry.lock",
            "requirements*.txt",
            "setup.py",
            "setup.cfg",
        ),
        refresh_action="pyright.reindex",
    ),
    ProviderRules(
        name="typescript",
        markers=("package.json",),
        trigger_globs=(
            "package.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "package-lock.json",
            "tsconfig*.json",
            "pnpm-workspace.yaml",
            "nx.json",
            "turbo.json",
        ),
        refresh_action="tsserver.project_reload",
    ),
    ProviderRules(
        name="rust",
        markers=("Cargo.toml",),
        trigger_globs=("Cargo.toml", "Cargo.lock", "**/Cargo.toml"),
        refresh_action="rust-analyzer.reloadWorkspace",
    ),
    ProviderRules(
        name="go",
        markers=("go.mod",),
        trigger_globs=("go.mod", "go.sum", "go.work"),
        refresh_action="gopls.didChangeConfiguration",
    ),
    ProviderRules(
        name="cpp",
        markers=("CMakeLists.txt", "compile_commands.json"),
        trigger_globs=(
            "compile_commands.json",
            "CMakeLists.txt",
            "**/CMakeLists.txt",
            "meson.build",
            "Makefile",
        ),
        refresh_action="clangd.reload_compile_db",
    ),
    ProviderRules(
        name="scala",
        markers=("build.sbt",),
        trigger_globs=("build.sbt", "project/**.scala", "project/**.sbt"),
        refresh_action="metals.import_build",
    ),
)

DEFAULT_RULES: Final = ProviderRules(
    name="default",
    markers=(),
    trigger_globs=(".git/HEAD", "*.toml", "*.json", "*.yml"),
    refresh_action="serena.refresh_project",
)


def rules_for_project(root: Path) -> list[ProviderRules]:
    """Return every provider whose marker is present in ``root``.

    Falls back to :data:`DEFAULT_RULES` only when nothing matched —
    §9 forbids the default set co-firing with a specific provider.
    """
    matched = [p for p in PROVIDERS if p.matches_marker(root)]
    return matched or [DEFAULT_RULES]


# ── structural snapshots ─────────────────────────────────────────────

_GRADLE_INCLUDE = re.compile(r"""include\s*\(?\s*['"]([^'"]+)['"]""")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _snapshot_one(root: Path, rules: ProviderRules) -> dict[str, object]:
    """Small structural fingerprint for one provider."""
    if rules.name == "kotlin-gradle":
        settings = root / "settings.gradle.kts"
        if not settings.exists():
            settings = root / "settings.gradle"
        text = _read(settings)
        return {
            "includes": sorted(set(_GRADLE_INCLUDE.findall(text))),
            "toml_hash": _digest(_read(root / "gradle" / "libs.versions.toml")),
        }
    if rules.name == "python":
        return {
            "manifest_hash": _digest(_read(root / "pyproject.toml")),
            "deps_hash": _digest(_read(root / "uv.lock")),
        }
    if rules.name == "typescript":
        return {"manifest_hash": _digest(_read(root / "package.json"))}
    if rules.name == "rust":
        return {"manifest_hash": _digest(_read(root / "Cargo.toml"))}
    if rules.name == "go":
        return {"manifest_hash": _digest(_read(root / "go.mod"))}
    if rules.name == "maven":
        return {"manifest_hash": _digest(_read(root / "pom.xml"))}
    if rules.name == "cpp":
        return {"manifest_hash": _digest(_read(root / "CMakeLists.txt"))}
    if rules.name == "scala":
        return {"manifest_hash": _digest(_read(root / "build.sbt"))}
    # default
    return {"git_head": _digest(_read(root / ".git" / "HEAD"))}


def snapshot_for(
    root: Path, rules: Sequence[ProviderRules]
) -> dict[str, object]:
    """Merge each active provider's structural snapshot."""
    return {r.name: _snapshot_one(root, r) for r in rules}


# ── the detector ─────────────────────────────────────────────────────


class ChangeDetector:
    """Decides whether a filesystem event warrants a project refresh."""

    def __init__(
        self,
        root: Path,
        *,
        debounce_s: float = DEFAULT_DEBOUNCE_S,
        on_refresh: Callable[[str], None] | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._rules = rules_for_project(self._root)
        self._debounce_s = debounce_s
        self._on_refresh = on_refresh
        #: Optional observability hook fired once per structural
        #: evaluation. Lets callers (and tests) see how often the
        #: disk-touching snapshot computation actually runs, which is
        #: what debouncing exists to minimise.
        self.on_evaluate: Callable[[], None] | None = None
        self._snapshot = snapshot_for(self._root, self._rules)
        self._pending: asyncio.Task[None] | None = None
        #: Timers cancelled by a newer event; awaited by ``drain``.
        self._superseded: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def rules(self) -> list[ProviderRules]:
        return list(self._rules)

    def snapshot(self) -> dict[str, object]:
        """Current structural snapshot (recomputed on demand)."""
        return snapshot_for(self._root, self._rules)

    # ── decision ─────────────────────────────────────────────────────

    def _is_trigger(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self._root)
        except (ValueError, OSError):
            return False
        rel_str = rel.as_posix()
        for r in self._rules:
            for pattern in r.trigger_globs:
                if rel.match(pattern) or _glob_match(rel_str, pattern):
                    return True
        return False

    def should_refresh(self, changed: Sequence[Path]) -> bool:
        """True iff a change is structural.

        Two gates, in order:

        1. at least one changed path must be a trigger file for an
           active provider (a ``.kt`` edit is not);
        2. the recomputed snapshot must actually differ — a formatter
           rewriting identical bytes is a no-op.
        """
        if self.on_evaluate is not None:
            self.on_evaluate()
        paths = [Path(p) for p in changed]
        if not any(self._is_trigger(p) for p in paths):
            return False
        current = self.snapshot()
        if current == self._snapshot:
            return False
        self._snapshot = current
        return True

    # ── debounced, event-driven delivery ─────────────────────────────

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for task in [*self._superseded, self._pending]:
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._superseded.clear()
        self._pending = None

    async def notify(self, changed: Sequence[Path]) -> None:
        """Record an event; schedule a debounced evaluation.

        A burst restarts the timer, so N events within the window cost
        exactly one evaluation and at most one refresh. Superseded
        timers are tracked until they finish unwinding so ``drain`` can
        prove none of them fired.
        """
        paths = [Path(p) for p in changed]
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
            self._superseded.append(self._pending)
        self._pending = asyncio.create_task(self._settle(paths))

    async def _settle(self, paths: list[Path]) -> None:
        try:
            await asyncio.sleep(self._debounce_s)
        except asyncio.CancelledError:
            return
        if self.should_refresh(paths) and self._on_refresh is not None:
            self._on_refresh(",".join(r.name for r in self._rules))

    async def drain(self) -> None:
        """Await every scheduled debounce, not just the newest one.

        Awaiting only the latest timer would let a broken (non-
        coalescing) implementation look correct: the older timers would
        still be sleeping when the caller inspected the results.
        """
        tasks = [*self._superseded]
        if self._pending is not None:
            tasks.append(self._pending)
        self._superseded.clear()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


def _glob_match(rel_str: str, pattern: str) -> bool:
    """``Path.match`` does not handle a leading ``**/`` the way §9
    means it, so widen the comparison for those patterns."""
    from fnmatch import fnmatch

    if pattern.startswith("**/"):
        tail = pattern[3:]
        return fnmatch(rel_str, tail) or fnmatch(rel_str, f"*/{tail}") or fnmatch(
            rel_str, pattern
        )
    return fnmatch(rel_str, pattern)
