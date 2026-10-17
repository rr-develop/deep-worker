"""The one place that knows where this application's files live.

Two layouts, one resolver
=========================
``deep_worker_web`` runs from two physically different layouts:

  from-source (a git checkout)
      <repo>/deep_worker_web/app/<module>.py
      <repo>/deep_worker_web/static/
      <repo>/deep_worker/deep_worker_cli/      <- sibling packages
      <repo>/deep_worker/deepagents/

  installed (inside the deep_worker wheel)
      <site-packages>/deep_worker_cli/web/app/<module>.py
      <site-packages>/deep_worker_cli/web/static/
      (no sibling source tree — deep_worker_cli is already importable)

The installed layout is reached under two different package names: as
``deep_worker_cli.web.app`` when imported as a subpackage, and as plain
``app`` under ``deep-worker web``, which puts the wheel's ``web/``
directory on ``sys.path`` exactly as ``run.py`` does from source. Both are
the same files; neither is "from source".

Several modules used to recompute the layout themselves with a ``.parent``
chain tuned for the checkout. Under the installed layout those chains land
on ``<site-packages>/deep_worker_cli`` — a directory that EXISTS — so every
``.exists()``-guarded consumer took the happy path with a wrong root and the
server degraded silently instead of crashing.

Replacing one chain with two chains (``parents[2]`` here, ``parents[4]``
there) would only restate the bug: it hard-codes the very layout assumption
that is supposed to vary, and the depth has to be re-derived by hand at every
call site. So this module derives everything from a SINGLE anchor — the
``app`` package located through ``importlib.resources`` — and branches on an
installation MARKER, never on a parent count.

The marker is whether the sibling source packages are actually THERE
(``_SOURCE_TREE_MARKERS``), not what the package is named. The package name
differs between layouts, but not reliably: ``deep-worker web`` imports the
wheel's app top-level as ``app``, so a name-based marker reports "from
source" while running from site-packages. See the comment on
``_SOURCE_TREE_MARKERS``.

The anchor is the ``app`` package and not its ``web`` parent on purpose. For
a regular package ``importlib.resources.files`` returns a real ``Path``; for
the namespace package that ``deep_worker_cli/web`` is in the wheel it returns
a ``MultiplexedPath``, which has no ``__fspath__`` and therefore cannot be
handed to ``StaticFiles(directory=...)``.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

__all__ = [
    "app_package_dir",
    "is_installed",
    "source_tree_root",
    "web_asset_dir",
    "web_root",
]

# The directories a real checkout has beside deep_worker_web/ and an
# installed distribution does not. These are what ``source_tree_root()``
# exists to point at, so their presence IS the marker: asking "is there a
# source tree here" answers the only question callers actually have.
#
# The package NAME cannot serve as the marker, even though it differs
# between layouts ("app" vs "deep_worker_cli.web.app"). The ``deep-worker
# web`` subcommand puts the ``web/`` directory on ``sys.path`` so that
# ``app`` is importable top-level — the same thing ``run.py`` does from
# source, and what keeps ~20 absolute ``from app.services …`` imports
# working in both layouts. Under that subcommand ``__package__`` is "app"
# while the files sit inside the wheel, so a name-based marker reports
# "from source" and hands callers ``<site-packages>/deep_worker_cli`` as a
# tree root. That path EXISTS and its expected subdirectories do not, which
# is the silent failure this module was written to remove. Measured, not
# assumed.
_SOURCE_TREE_MARKERS = (
    Path("deep_worker") / "deep_worker_cli",
    Path("deep_worker") / "deepagents",
)


def _anchor() -> str:
    """Return the import name of the ``app`` package this module belongs to."""
    # "app" from source and when ``deep-worker web`` puts web/ on sys.path;
    # "deep_worker_cli.web.app" when imported as a subpackage. All three
    # name the directory holding this file, which is all it is used for.
    return __package__ or "app"


def is_installed() -> bool:
    """True when there is no source tree beside us, i.e. an installed wheel.

    This is the installation marker every other function here branches on.
    """
    return source_tree_root() is None


def app_package_dir() -> Path:
    """Filesystem directory of the ``app`` package — the single anchor."""
    return Path(str(importlib.resources.files(_anchor())))


def web_root() -> Path:
    """Directory holding the application package and its assets.

    ``<repo>/deep_worker_web`` from source, ``.../deep_worker_cli/web`` when
    installed. One ``.parent`` in BOTH layouts: ``static/`` is a sibling of
    ``app/`` either way, which is what makes a single anchor sufficient.
    """
    return app_package_dir().parent


def web_asset_dir() -> Path:
    """Directory of the static assets (html / js / css), as package data."""
    return web_root() / "static"


def source_tree_root() -> Path | None:
    """Repo root holding ``deep_worker/deep_worker_cli`` and
    ``deep_worker/deepagents``, or ``None`` when installed.

    Callers use this to put the checkout's own sibling packages on
    ``sys.path`` and to spawn agents against the same tree the server runs
    from. An installed distribution has no such tree: ``deep_worker_cli`` is
    already importable, and there is no directory to point at. Returning a
    path anyway is precisely the silent failure this module exists to
    prevent, so the answer is ``None`` and callers must handle it.

    The marker is the presence of the sibling packages themselves, so a
    directory that merely EXISTS at the right depth is not mistaken for a
    checkout — see ``_SOURCE_TREE_MARKERS``.
    """
    candidate = web_root().parent
    if all((candidate / marker).is_dir() for marker in _SOURCE_TREE_MARKERS):
        return candidate
    return None
