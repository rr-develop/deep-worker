"""Regression tests for tree-local path resolution in deep_worker_web.

Background
==========
deep_worker_web can be checked out at TWO locations on the same host:

  * deep_worker_web — prod checkout  (port 7887)
  * deep_worker_web — dev checkout   (port 7888)

For --dev to work end-to-end, the dev web server MUST spawn dev-tree
agents (so they share the same deep_worker_cli / deepagents codebase).
Previously the code hard-coded an absolute prod-checkout path in the
spawn helper and the registry-loader, which meant the dev web server
silently spawned prod-tree agents — defeating the whole point of
operational isolation.

These tests are textual: they assert the files do not contain the
literal prod path. The fix is to derive paths from ``__file__``.

Design doc: docs/DESIGN_DEV_FLAG.md
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ``deep_worker_web`` root = parent x3 of this file.
_THIS_FILE = Path(__file__).resolve()
_WEB_ROOT = _THIS_FILE.parent.parent.parent  # → deep_worker_web/

AGENTS_ROUTER = _WEB_ROOT / "app" / "routers" / "agents.py"
AGENT_SERVICE = _WEB_ROOT / "app" / "services" / "agent_service.py"
CONFIG = _WEB_ROOT / "app" / "config.py"
MAIN = _WEB_ROOT / "app" / "main.py"
PATHS_MODULE = _WEB_ROOT / "app" / "paths.py"


_DW = "deep_worker"
_BAD_PROD = "/path/to/prod-tree/deep_worker/deep_worker_cli"
_BAD_PROD_DEEPAGENTS = f"/path/to/prod-tree/{_DW}/deepagents"
_BAD_DEV = "/path/to/dev-tree/deep_worker/deep_worker_cli"
_BAD_DEV_DEEPAGENTS = f"/path/to/dev-tree/{_DW}/deepagents"


def test_agents_router_exists() -> None:
    assert AGENTS_ROUTER.exists(), f"not found: {AGENTS_ROUTER}"


def test_agent_service_exists() -> None:
    assert AGENT_SERVICE.exists(), f"not found: {AGENT_SERVICE}"


def test_agents_router_has_no_hardcoded_tree_paths() -> None:
    """``app/routers/agents.py`` builds PYTHONPATH for spawned agents.
    It must NOT contain a hard-coded prod or dev path — it must derive
    the path from its own location so that the dev web server spawns
    dev-tree agents and the prod web server spawns prod-tree agents.
    """
    text = AGENTS_ROUTER.read_text()
    for bad in (_BAD_PROD, _BAD_PROD_DEEPAGENTS, _BAD_DEV, _BAD_DEV_DEEPAGENTS):
        assert bad not in text, (
            f"{AGENTS_ROUTER.name} still hard-codes {bad!r}. "
            f"This breaks operational isolation: the dev web server "
            f"would silently spawn agents on the prod codebase (or "
            f"vice versa). Derive the path from __file__ instead. "
            f"See docs/DESIGN_DEV_FLAG.md."
        )


def test_agent_service_has_no_hardcoded_tree_paths() -> None:
    """``app/services/agent_service.py`` adds deep_worker_cli to sys.path
    so it can import ProcessRegistry. Same rule: derive from __file__,
    don't hard-code a specific tree."""
    text = AGENT_SERVICE.read_text()
    for bad in (_BAD_PROD, _BAD_PROD_DEEPAGENTS, _BAD_DEV, _BAD_DEV_DEEPAGENTS):
        assert bad not in text, (
            f"{AGENT_SERVICE.name} still hard-codes {bad!r}. "
            f"Derive from __file__ so the import follows the tree the "
            f"web server lives in. See docs/DESIGN_DEV_FLAG.md."
        )


def test_agents_router_derives_paths_from_file() -> None:
    """Sanity check: the file should reference ``__file__`` (which is
    the fix). Loose, but catches a wholesale revert."""
    text = AGENTS_ROUTER.read_text()
    assert "__file__" in text, (
        f"{AGENTS_ROUTER.name} no longer mentions __file__ — was the "
        f"tree-local path fix reverted?"
    )


def test_agent_service_derives_paths_from_file() -> None:
    """Same guard as above, one level of indirection later.

    This file used to compute the tree root itself from ``__file__``, and
    this test asserted that mention as proof the prod literal was gone.
    The computation now lives in the shared resolver (``app/paths.py``,
    which does anchor on the import system), so requiring ``__file__``
    *here* would force the duplication back. The invariant it was really
    protecting — "this module does not name a tree, it asks" — is checked
    by requiring the resolver import instead.
    """
    text = AGENT_SERVICE.read_text()
    assert "__file__" in text or _imports_resolver(text), (
        f"{AGENT_SERVICE.name} neither derives paths from __file__ nor "
        f"imports the shared resolver — was the tree-local path fix "
        f"reverted?"
    )


# ===========================================================================
# Shipping the web app inside the deep_worker wheel
# ===========================================================================
# The textual tests above fixed hard-coded ABSOLUTE paths. They do not and
# cannot catch the next failure mode: the ``parents[N]`` / ``.parent`` chains
# that REPLACED those literals are themselves tied to the from-source layout
#
#     <repo>/deep_worker_web/app/<module>.py          (N parents → <repo>)
#
# Once the app ships inside the wheel it lives at
#
#     <site-packages>/deep_worker_cli/web/app/<module>.py
#
# where the same chain lands on ``.../deep_worker_cli`` — a directory that
# EXISTS, so every ``.exists()``-guarded consumer takes the happy path with a
# wrong root and the server degrades silently instead of crashing. Three sites
# fail that way (measured, see ``_SILENT_SITES`` below).
#
# The fix is a single resolver, ``app/paths.py``, that distinguishes the two
# layouts by an INSTALLATION MARKER (how the ``app`` package is named /
# located via ``importlib.resources``) rather than by counting parents in two
# branches. Counting parents in two branches is exactly the bug restated: it
# hard-codes the very layout assumption that is supposed to vary.


def test_paths_module_exists() -> None:
    """A single resolver module must exist (condition 1)."""
    assert PATHS_MODULE.exists(), (
        f"not found: {PATHS_MODULE}. The three silent path sites "
        f"(app/config.py, app/routers/agents.py, "
        f"app/services/agent_service.py) each recompute the tree root with "
        f"their own .parent chain. They must share ONE resolver."
    )


def test_paths_module_uses_installation_marker_not_parent_arithmetic() -> None:
    """The resolver must branch on an installation marker, not on a
    different ``parents[N]`` count per layout.

    Two ``parents[N]`` expressions with different N inside the resolver mean
    the layout assumption was merely duplicated, not removed.
    """
    if not PATHS_MODULE.exists():
        pytest.fail(
            f"{PATHS_MODULE} does not exist — see "
            f"test_paths_module_exists. This test is about the resolver's "
            f"SHAPE and cannot speak to a missing file."
        )
    source = PATHS_MODULE.read_text()
    assert "importlib.resources" in source or "importlib" in source, (
        "paths.py must use importlib.resources to locate the app package — "
        "that is the installation marker. Found no importlib reference."
    )

    tree = ast.parse(source)

    # Collect every parent-walking expression: parents[N] subscripts and
    # chained .parent attribute accesses.
    parent_index_counts: list[int] = []
    parent_chain_lengths: list[int] = []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            parent_index_counts.append(node.slice.value)

    def _chain_len(node: ast.AST) -> int:
        n = 0
        while isinstance(node, ast.Attribute) and node.attr == "parent":
            n += 1
            node = node.value
        return n

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            # Only count the OUTERMOST link of each chain.
            parent_chain_lengths.append(_chain_len(node))

    distinct_depths = set(parent_index_counts) | {
        n for n in parent_chain_lengths if n >= 2
    }
    assert len(distinct_depths) <= 1, (
        f"paths.py walks parents at MORE THAN ONE depth {sorted(distinct_depths)}. "
        f"That is the from-source layout assumption duplicated per branch, "
        f"which is the defect this resolver exists to remove. Distinguish the "
        f"layouts by the installation marker and derive both roots from the "
        f"SAME anchor."
    )


# --------------------------------------------------------------------------
# The three silent sites, identified by the ACTUAL SYNTAX of the line.
# --------------------------------------------------------------------------
# Deliberately NOT keyed off a comment label: a comment can be edited to
# agree with a test while the expression keeps the old behaviour (that is a
# live defect in this very tree — the ``<TREE>`` label at agent_service.py
# claims "parent x3" while the code does "parent x4"). Each entry below is
# the exact source text that must disappear.

_SILENT_SITES: list[tuple[Path, str]] = [
    (CONFIG, "Path(__file__).resolve().parents[2]"),
    (
        AGENTS_ROUTER,
        "Path(__file__).resolve().parent.parent.parent.parent",
    ),
    (
        AGENT_SERVICE,
        "Path(__file__).resolve().parent.parent.parent.parent",
    ),
]


@pytest.mark.parametrize(
    "module_path,bad_expression",
    _SILENT_SITES,
    ids=[p.name for p, _ in _SILENT_SITES],
)
def test_silent_site_no_longer_computes_its_own_tree_root(
    module_path: Path, bad_expression: str
) -> None:
    """Each silent site must delegate to the shared resolver.

    ``bad_expression`` is matched as literal source text, so a comment
    rewrite cannot make this pass.
    """
    source = module_path.read_text()
    assert bad_expression not in source, (
        f"{module_path.name} still computes the tree root itself via "
        f"{bad_expression!r}. Under the installed layout "
        f"(<site-packages>/deep_worker_cli/web/app/) that chain lands on "
        f"deep_worker_cli — a directory that EXISTS — so the .exists() guard "
        f"passes with a wrong root and the failure is SILENT. Call the "
        f"shared resolver in app/paths.py instead."
    )


def _imports_resolver(source: str) -> bool:
    """True when *source* imports the shared ``paths`` resolver module.

    Matched on the import STATEMENT via the AST, never on the substring
    ``"paths"``: these modules already import ``deep_worker_cli.
    storage_paths`` and mention "paths" in prose, so a substring check
    would pass vacuously.

    Relative imports are what the two layouts actually require (the
    package is ``app`` from source and ``deep_worker_cli.web.app`` when
    installed), and for those ``ImportFrom.module`` is None — the name
    sits in ``names`` instead. Both forms count.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "paths" or mod.endswith(".paths"):
                return True
            # ``from . import paths`` / ``from .. import paths``
            if node.level > 0 and any(a.name == "paths" for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(
                a.name == "paths" or a.name.endswith(".paths")
                for a in node.names
            ):
                return True
    return False


@pytest.mark.parametrize(
    "module_path",
    [p for p, _ in _SILENT_SITES],
    ids=[p.name for p, _ in _SILENT_SITES],
)
def test_silent_site_imports_the_shared_resolver(module_path: Path) -> None:
    """Positive control for the test above: removing the bad expression is
    not enough, the site must actually import from the resolver.

    Matched on the import STATEMENT via the AST, not on the substring
    ``"paths"`` — these modules already import ``deep_worker_cli.
    storage_paths`` and mention "paths" in prose, so a substring check would
    pass vacuously.
    """
    assert _imports_resolver(module_path.read_text()), (
        f"{module_path.name} no longer contains the old parent chain but "
        f"has no `import`/`from ... import` of the shared resolver module "
        f"`paths`. Deleting the computation without delegating would leave "
        f"the site with no root at all."
    )


# --------------------------------------------------------------------------
# Mutation-grade proof: run the resolver under BOTH layouts.
# --------------------------------------------------------------------------
# A test that only exercises the tree it lives in proves nothing about the
# installed layout — that is the whole failure this work addresses. These two
# tests materialise each layout on disk and assert the resolver's answer with
# the CONCRETE expected path in the assertion message.

_RESOLVER_PROBE = textwrap.dedent(
    """
    import json
    import sys

    from {pkg} import paths

    print(json.dumps({{
        "installed": bool(paths.is_installed()),
        "assets": str(paths.web_asset_dir()),
        "tree_root": (
            None if paths.source_tree_root() is None
            else str(paths.source_tree_root())
        ),
    }}))
    """
)


def _run_probe(sys_path_root: Path, pkg: str) -> dict[str, object]:
    """Import the resolver from *sys_path_root* and return its answers."""
    import json

    code = _RESOLVER_PROBE.format(pkg=pkg)
    env = {
        "PYTHONPATH": str(sys_path_root),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        # Byte-code from a previous layout must never answer for this one.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        # ``python -c`` puts the CWD on sys.path ahead of PYTHONPATH. Run
        # from the layout root itself so the real checkout can never answer
        # for the materialised one.
        cwd=str(sys_path_root),
    )
    assert proc.returncode == 0, (
        f"resolver probe failed under {pkg!r} rooted at {sys_path_root}:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _materialise_from_source(dest: Path) -> Path:
    """Build the from-source layout: <repo>/deep_worker_web/app/."""
    import shutil

    web = dest / "deep_worker_web"
    web.mkdir(parents=True)
    shutil.copytree(
        _WEB_ROOT / "app",
        web / "app",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(_WEB_ROOT / "static", web / "static")
    # The markers a real checkout has and an installed tree does not.
    (dest / "deep_worker" / "deep_worker_cli").mkdir(parents=True)
    (dest / "deep_worker" / "deepagents").mkdir(parents=True)
    return web


def _materialise_installed(dest: Path) -> Path:
    """Build the installed layout:
    <site-packages>/deep_worker_cli/web/app/."""
    import shutil

    cli = dest / "deep_worker_cli"
    web = cli / "web"
    web.mkdir(parents=True)
    (cli / "__init__.py").write_text("")
    shutil.copytree(
        _WEB_ROOT / "app",
        web / "app",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(_WEB_ROOT / "static", web / "static")
    return dest


def test_resolver_answers_correctly_under_from_source_layout(
    tmp_path: Path,
) -> None:
    root = _materialise_from_source(tmp_path / "repo")
    answers = _run_probe(root, "app")

    expected_assets = root / "static"
    assert answers["assets"] == str(expected_assets), (
        f"under the from-source layout the resolver must report the static "
        f"assets at {expected_assets}, got {answers['assets']!r}"
    )
    assert answers["installed"] is False, (
        f"the from-source checkout at {root} must NOT be reported as an "
        f"installed distribution; the installation marker misfired"
    )
    expected_tree = tmp_path / "repo"
    assert answers["tree_root"] == str(expected_tree), (
        f"under the from-source layout the sibling-package tree root must be "
        f"{expected_tree} (it holds deep_worker/deep_worker_cli), got "
        f"{answers['tree_root']!r}"
    )


def test_resolver_answers_correctly_under_installed_layout(
    tmp_path: Path,
) -> None:
    site = _materialise_installed(tmp_path / "site-packages")
    answers = _run_probe(site, "deep_worker_cli.web.app")

    expected_assets = site / "deep_worker_cli" / "web" / "static"
    assert answers["assets"] == str(expected_assets), (
        f"under the installed layout the resolver must report the static "
        f"assets at {expected_assets}, got {answers['assets']!r}. A "
        f".parent chain tuned for the checkout lands on "
        f"{site / 'deep_worker_cli'} instead — which EXISTS, so the failure "
        f"is silent."
    )
    assert answers["installed"] is True, (
        f"the installed layout at {site} must be reported as installed; the "
        f"installation marker failed to fire"
    )
    assert answers["tree_root"] is None, (
        f"an installed distribution has no sibling source tree, so the tree "
        f"root must be None — reporting {answers['tree_root']!r} would make "
        f"callers add a nonexistent path to sys.path (or, worse, a path that "
        f"exists but is the wrong root)"
    )


def test_resolver_is_right_when_installed_app_is_imported_top_level(
    tmp_path: Path,
) -> None:
    """The installed layout, imported the way ``deep-worker web`` imports it.

    The subcommand puts the wheel's ``web/`` directory on ``sys.path`` so
    that ``app`` resolves top-level — the same thing ``run.py`` does from
    source, and what keeps ~20 absolute ``from app.services …`` imports
    working without touching them. So under the wheel the package can be
    named EITHER ``deep_worker_cli.web.app`` or plain ``app``.

    That makes the package name useless as the installation marker, and
    this is the case that proves it: a name-based marker reports "from
    source" here and hands callers ``<site-packages>/deep_worker_cli`` as a
    tree root. That directory exists, its expected subdirectories do not,
    and the spawn path would be built from nonexistent entries — silently.
    """
    site = _materialise_installed(tmp_path / "site-packages")
    web_dir = site / "deep_worker_cli" / "web"

    # Exactly what the subcommand does: web/ on sys.path, not site-packages.
    answers = _run_probe(web_dir, "app")

    expected_assets = web_dir / "static"
    assert answers["assets"] == str(expected_assets), (
        f"assets must still resolve to {expected_assets} when the installed "
        f"app is imported top-level, got {answers['assets']!r}"
    )
    assert answers["tree_root"] is None, (
        f"there is no source tree under {site}: "
        f"{site / 'deep_worker_cli' / 'deep_worker'} does not exist. The "
        f"resolver reported {answers['tree_root']!r} — a directory that "
        f"exists at the right depth but holds none of the sibling packages, "
        f"which is the silent failure, not a tree root."
    )
    assert answers["installed"] is True, (
        "a wheel imported top-level is still a wheel; the marker must not "
        "depend on how the package was named on sys.path"
    )


def test_static_assets_are_readable_as_package_data(tmp_path: Path) -> None:
    """Every shipped asset must be reachable through the resolver under the
    installed layout — i.e. actually be package data, not a tree-relative
    file that the wheel happens to omit."""
    site = _materialise_installed(tmp_path / "site-packages")
    answers = _run_probe(site, "deep_worker_cli.web.app")
    assets = Path(str(answers["assets"]))

    expected = {
        "index.html",
        "stats.html",
        "js/main.js",
        "js/stats.js",
        "css/style.css",
        "css/colors.css",
    }
    missing = {rel for rel in expected if not (assets / rel).is_file()}
    assert not missing, (
        f"assets unreachable under the installed layout at {assets}: "
        f"{sorted(missing)}"
    )


# --------------------------------------------------------------------------
# Dead template branch
# --------------------------------------------------------------------------
# ``app/templates/`` does not exist in this tree, so ``main.py`` logs a
# warning at every import and binds a ``templates`` variable that no code
# path ever reads. Shipping it would mean declaring a Jinja2 dependency for
# a directory that is never created.


def test_main_has_no_dead_template_branch() -> None:
    source = MAIN.read_text()
    assert "Jinja2Templates" not in source, (
        "app/main.py still imports Jinja2Templates for the app/templates/ "
        "directory, which does not exist in this tree. The bound `templates` "
        "variable is never read, so the import only adds a dependency the "
        "[web] extra would have to declare."
    )
    assert "templates_dir" not in source, (
        "app/main.py still computes templates_dir for a directory that does "
        "not exist; the branch is dead code."
    )


def test_main_mounts_static_through_the_resolver() -> None:
    source = MAIN.read_text()
    assert 'Path(__file__).parent.parent / "static"' not in source, (
        "app/main.py still locates static assets with a tree-relative "
        "parent chain instead of reading them as package data through the "
        "shared resolver."
    )
    assert _imports_resolver(source), (
        "app/main.py no longer has the old static path chain but never "
        "imports the shared resolver module `paths`."
    )
