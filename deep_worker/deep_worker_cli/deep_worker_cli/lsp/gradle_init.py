"""Installation of the Gradle init script that fixes Kotlin LSP cross-file navigation.

See ``lsp/docs/DESIGN_KOTLIN_LSP_ANDROID_SOURCE_SETS.md`` §7.1.

Why the file is installed in the GLOBAL ``~/.gradle/init.d`` and not in the
project: Gradle picks up init scripts ONLY from ``$GRADLE_USER_HOME/init.d`` —
a script placed in the project directory is not applied (verified, §7 limit A
of the design). We do not control the Gradle command line that Kotlin LSP
itself builds, so ``--init-script`` is also unavailable. This is the only
working point.

Guarantees (deliberately the same as ``enable.py`` for ``mcp.json``):

* **a copy, not a symlink** — a symlink would point into the venv/wheel and
  break on package reinstall;
* **idempotent** — a repeated install does not touch the file;
* **we do not clobber foreign files** — a user-modified file is not
  overwritten without ``force``; only the file carrying our version marker
  is removed;
* **permissions 0o644 / 0o755** — unlike ``mcp.json`` (0600/0700) there are
  no secrets here, and the Gradle daemon reads the file. No 777/666
  (project policy);
* **atomic write** (tmp in the same directory + ``os.replace``).
"""
from __future__ import annotations

import enum
import os
from pathlib import Path

__all__ = [
    "GRADLE_INIT_SCRIPT_NAME",
    "InstallState",
    "default_init_d_dir",
    "gradle_init_script_source",
    "gradle_init_state",
    "install_gradle_init_script",
    "is_gradle_init_installed",
    "uninstall_gradle_init_script",
]

#: File name in ``init.d``. The ``deep-worker-`` prefix makes the file
#: recognizable among the user's foreign init scripts.
GRADLE_INIT_SCRIPT_NAME = "deep-worker-kotlin-android-source-sets.gradle"

#: First line of the resource. By it (NOT by byte equality) we distinguish
#: "our" file from a foreign one and can do version migrations.
_VERSION_MARKER_PREFIX = "// deep-worker:kotlin-android-source-sets v"

_ASSET_RELPATH = "assets/kotlin-android-source-sets.init.gradle"

_FILE_MODE = 0o644
_DIR_MODE = 0o755


def gradle_init_script_source() -> str:
    """Content of the script resource shipped with the package."""
    return (Path(__file__).parent / _ASSET_RELPATH).read_text(encoding="utf-8")


def default_init_d_dir() -> Path:
    """``$GRADLE_USER_HOME/init.d`` (default ``~/.gradle/init.d``)."""
    gradle_home = os.environ.get("GRADLE_USER_HOME")
    base = Path(gradle_home) if gradle_home else Path.home() / ".gradle"
    return base / "init.d"


class InstallState(enum.Enum):
    """What sits at the target path.

    Distinguishing ``CURRENT`` from ``OURS_OUTDATED`` is mandatory:
    otherwise, after a new script version ships, the CLI would report
    "already installed", hiding that the user has an old (or edited) copy.
    """

    #: no file
    ABSENT = "absent"
    #: our file, content matches the shipped one
    CURRENT = "current"
    #: our marker, but the content differs (user edit or an old version)
    OURS_OUTDATED = "ours_outdated"
    #: a file with that name exists but is not ours — never touched
    FOREIGN = "foreign"


def gradle_init_state(init_d_dir: Path | None = None) -> InstallState:
    """Installation state in ``init_d_dir``."""
    target = (init_d_dir or default_init_d_dir()) / GRADLE_INIT_SCRIPT_NAME
    try:
        current = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return InstallState.ABSENT
    if not _is_ours(current):
        return InstallState.FOREIGN
    if current == gradle_init_script_source():
        return InstallState.CURRENT
    return InstallState.OURS_OUTDATED


def _is_ours(text: str) -> bool:
    first_line = text.split("\n", 1)[0].strip()
    return first_line.startswith(_VERSION_MARKER_PREFIX)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    # mkdir(mode=...) does not apply to an already-existing directory and
    # is cut by umask — set it explicitly (incident with daemon umask 0 → 777).
    path.parent.chmod(_DIR_MODE)

    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp.chmod(_FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_gradle_init_installed(init_d_dir: Path | None = None) -> bool:
    """Whether OUR script (by version marker) is installed in ``init_d_dir``.

    Does not distinguish an up-to-date from an outdated version — for
    that, ``gradle_init_state``.
    """
    return gradle_init_state(init_d_dir) in (
        InstallState.CURRENT,
        InstallState.OURS_OUTDATED,
    )


def install_gradle_init_script(
    init_d_dir: Path | None = None,
    *,
    force: bool = False,
) -> bool:
    """Put the init script into ``init_d_dir``.

    Returns:
        ``True`` if the file was written, ``False`` if no write was needed
        (already current) or it was rejected (foreign/edited file without
        ``force``).
    """
    target = (init_d_dir or default_init_d_dir()) / GRADLE_INIT_SCRIPT_NAME
    state = gradle_init_state(init_d_dir)

    if state is not InstallState.ABSENT and not force:
        # CURRENT — idempotent no-op; OURS_OUTDATED/FOREIGN must not be
        # silently clobbered, the decision is the user's (--force).
        return False

    _atomic_write(target, gradle_init_script_source())
    return True


def uninstall_gradle_init_script(init_d_dir: Path | None = None) -> bool:
    """Remove OUR init script.

    A foreign file with the same name (no version marker) is not touched.

    Returns:
        ``True`` if the file was removed, otherwise ``False``.
    """
    target = (init_d_dir or default_init_d_dir()) / GRADLE_INIT_SCRIPT_NAME
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not _is_ours(text):
        return False
    try:
        target.unlink()
    except OSError:
        return False
    return True
