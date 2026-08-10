"""Project languages in ``<project>/.serena/project.yml``.

See ``docs/DESIGN_LSP_LANGUAGES.md``.

The problem this module solves: on first activation Serena enables
exactly one language — the most frequent one (``serena_config.py:376-382``);
the rest it only offers in interactive mode, which the MCP server does not
have. On a polyglot project this silently drops half the codebase, and the
agent only sees ``Cannot extract symbols. Active languages: [...]`` without
being able to understand the cause or fix it.

Two principles:

* **Detection is foreign.** Serena itself computes the composition
  (``determine_programming_language_composition``): it walks the files via
  ``find_all_non_ignored_files`` (so ``.gitignore``, including nested ones,
  is respected) and knows its own language list. Our own walk would
  inevitably diverge from both.
* **Serena lives in its own venv.** It is installed via ``uv tool`` and is
  NOT importable from our interpreter (``ModuleNotFoundError``). So its code
  is invoked from its own environment: the composition via a short script
  run by the interpreter next to the ``serena`` binary, config creation via
  the official ``serena project create`` command. No ``pip install`` and no
  attempts to reach into a foreign ``site-packages`` directly.
* **The editor is strict.** ``project.yml`` carries a large block of
  documentation comments, so it is edited surgically rather than
  overwritten with a dump. Any ambiguity is a refusal, not a guess: a
  corrupted config is worse than an unexecuted command.
"""
from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Final

import yaml

#: Share of RECOGNIZED source files from which a language is enabled
#: automatically. The denominator is files recognized by at least one
#: language (that is how Serena counts), so images and lock files do not
#: dilute the share.
_LANGUAGE_THRESHOLD_PCT: Final = 10.0

#: Serena's project config file name.
SERENA_PROJECT_FILE: Final = "project.yml"

_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600

#: Top-level key of the languages list. Anchored to the line start — so
#: nested ``languages:`` (indented) do not match.
_TOP_LEVEL_LANGUAGES_RE: Final = re.compile(r"^languages\s*:(.*)$")

#: Block-list item line: ``- kotlin`` / ``  - kotlin``.
_LIST_ITEM_RE: Final = re.compile(r"^\s*-\s*(.+?)\s*$")

#: Start of any other top-level key — a block boundary.
_TOP_LEVEL_KEY_RE: Final = re.compile(r"^[A-Za-z_][\w-]*\s*:")


class LanguageConfigError(RuntimeError):
    """The config cannot be safely changed — the operation is aborted.

    Raised BEFORE any write, so the file on disk always stays untouched.
    """


# ── detection ─────────────────────────────────────────────────────────


#: Script executed by the Serena interpreter. Prints JSON
#: ``{"composition": {...}, "known": [...], "priority": {...}}`` —
#: everything we need from its internals in a single run.
_PROBE_SRC: Final = """
import json, sys
from serena.util.inspection import determine_programming_language_composition
from solidlsp.ls_config import Language
comp = determine_programming_language_composition(sys.argv[1])
known = [str(l.value) for l in Language.iter_all()]
prio = {}
for l in Language.iter_all(include_experimental=False):
    try:
        prio[str(l.value)] = int(l.get_priority())
    except Exception:
        prio[str(l.value)] = 0
print(json.dumps({
    "composition": {str(k.value): v for k, v in comp.items()},
    "known": known,
    "priority": prio,
}))
"""


def serena_python(serena_command: str | None = None) -> Path | None:
    """Interpreter from Serena's environment, if found.

    Serena is installed via ``uv tool`` into its own venv, so its modules
    are not available to our process. The ``serena`` binary is a symlink in
    ``.../serena-agent/bin/``, next to which the needed python lives.
    """
    import shutil  # noqa: PLC0415

    exe = shutil.which(serena_command or "serena")
    if not exe:
        return None
    real = Path(exe).resolve()
    candidate = real.parent / "python3"
    if candidate.exists():
        return candidate
    candidate = real.parent / "python"
    return candidate if candidate.exists() else None


def _probe(project_root: Path | str) -> dict[str, Any]:
    """One run in Serena's environment — composition + reference tables."""
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    python = serena_python()
    if python is None:
        raise LanguageConfigError(
            "serena is not installed (or not on PATH), cannot detect project "
            "languages; install it with 'uv tool install -p 3.13 serena-agent'"
        )
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(python), "-c", _PROBE_SRC, str(Path(project_root).resolve())],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise LanguageConfigError(
            f"failed to run the serena language probe: {e}"
        ) from e
    if proc.returncode != 0:
        raise LanguageConfigError(
            f"serena language probe failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise LanguageConfigError(
            f"serena language probe returned non-JSON output: {e}"
        ) from e


def language_composition(project_root: Path | str) -> dict[str, float]:
    """Language shares in the project per Serena's own computation.

    Returns:
        ``{language: percent}`` of recognized sources.

    Raises:
        LanguageConfigError: Serena is unavailable or the scan failed.
    """
    return dict(_probe(project_root).get("composition", {}))


def language_facts(project_root: Path | str) -> dict[str, Any]:
    """Composition + reference tables in ONE child-process run.

    A separate function so the CLI does not pay three runs where one
    suffices.
    """
    return _probe(project_root)


def _language_value(lang: Any) -> str:
    return str(getattr(lang, "value", lang))


def detected_languages(
    composition: dict[str, float],
    priority: dict[str, int] | None = None,
) -> list[str]:
    """Languages passing the threshold, in Serena order (A6).

    The sort mirrors ``serena_config.py:376-378`` — percentage
    descending, ties broken by priority descending. Order matters: for
    Serena the first language in the list is the default language, and
    it must not be left to dict-iteration order.
    """
    prio = priority or {}
    above = [
        (name, pct)
        for name, pct in composition.items()
        if pct >= _LANGUAGE_THRESHOLD_PCT
    ]
    above.sort(key=lambda item: (item[1], prio.get(item[0], 0)), reverse=True)
    return [name for name, _ in above]


def merge_languages(existing: list[str], detected: list[str]) -> list[str]:
    """Merge lists without removing or reordering anything.

    Existing entries come first and in their original order (the first is
    Serena's default language). Detected ones are only appended: a language
    may have been added by the user manually for a specific task, and
    silently dropping it because of an unmet threshold would lose their
    decision.
    """
    result = list(existing)
    seen = set(existing)
    for name in detected:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def validate_language_name(name: str, known: list[str]) -> str:
    """Validate a language name against Serena's list.

    Without this check a typo would silently land in the config, and
    Serena would refuse to start on the very next launch — far from the
    point of the error.
    """
    if name in known:
        return name
    close = [k for k in sorted(known) if name and k.startswith(name[:2])]
    hint = f"; did you mean: {', '.join(close)}" if close else ""
    raise LanguageConfigError(
        f"unknown language {name!r}{hint}. Known: {', '.join(sorted(known))}"
    )


# ── read / write project.yml ─────────────────────────────────────────


def project_config_path(project_root: Path | str) -> Path:
    return Path(project_root).resolve() / ".serena" / SERENA_PROJECT_FILE


def read_languages(config_path: Path) -> list[str]:
    """Current language list; ``[]`` if the file or key is absent."""
    try:
        raw = config_path.read_text()
    except FileNotFoundError:
        return []
    data = _safe_load(raw, config_path)
    value = data.get("languages") if isinstance(data, dict) else None
    if value is None:
        return []
    if not isinstance(value, list):
        raise LanguageConfigError(
            f"{config_path}: 'languages' must be a list, got {type(value).__name__}"
        )
    return [str(v) for v in value]


def _safe_load(raw: str, config_path: Path) -> Any:
    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise LanguageConfigError(f"{config_path}: malformed YAML ({e})") from e


def write_languages(config_path: Path, languages: list[str]) -> None:
    """Replace the language list, preserving the rest of the file byte-for-byte.

    Implemented as surgical text editing, not ``yaml.safe_dump``: Serena's
    template carries extensive documentation in comments, which a dump
    would destroy.

    Raises:
        LanguageConfigError: the file is ambiguous (duplicate key, only a
            nested match, broken YAML, missing key). The file is not
            changed.
    """
    raw = config_path.read_text()
    data = _safe_load(raw, config_path)
    if not isinstance(data, dict) or "languages" not in data:
        raise LanguageConfigError(
            f"{config_path}: no top-level 'languages' key; refusing to guess "
            f"where to write it"
        )

    lines = raw.splitlines(keepends=True)
    matches = [
        (i, m)
        for i, line in enumerate(lines)
        if (m := _TOP_LEVEL_LANGUAGES_RE.match(line)) is not None
    ]
    if len(matches) != 1:
        raise LanguageConfigError(
            f"{config_path}: expected exactly one top-level 'languages:' line, "
            f"found {len(matches)}; refusing to modify an ambiguous file"
        )

    start, match = matches[0]
    inline = match.group(1).strip()
    end = start + 1
    if not inline:
        # Block list. The node extends until the next TOP-LEVEL key (or
        # the end of the file) — neither a comment nor a blank line ends
        # the sequence. Stopping at them is not safe: a trailing
        # `- python` would survive the "removal", and PyYAML would keep
        # seeing it. Blank lines carry no semantics — we absorb them;
        # anything else inside the node (comments, folded continuations)
        # cannot be moved unambiguously → refuse without writing (A2).
        last_item = start
        while end < len(lines):
            line = lines[end]
            if _TOP_LEVEL_KEY_RE.match(line):
                break
            stripped = line.strip()
            if _LIST_ITEM_RE.match(line):
                end += 1
                last_item = end
                continue
            if not stripped or stripped.startswith("#"):
                # Blank lines and comments are decided post hoc: what
                # trails as a tail until the next key belongs to it (the
                # header), and what is squeezed between items belongs to
                # the list.
                end += 1
                continue
            raise LanguageConfigError(
                f"{config_path}: unsupported line inside the 'languages' block "
                f"at line {end + 1} ({stripped!r}); refusing to modify — "
                f"rewriting it could silently change or drop content"
            )
        # The tail after the last item is the next key's header, not
        # part of the list: return it untouched.
        end = last_item
        # A comment BETWEEN items (it stayed inside the slice) cannot be
        # moved unambiguously → refuse without writing (A2).
        for offset, line in enumerate(lines[start + 1 : end]):
            stripped = line.strip()
            if stripped.startswith("#"):
                raise LanguageConfigError(
                    f"{config_path}: comment inside the 'languages' list at "
                    f"line {start + 2 + offset} ({stripped!r}); refusing to "
                    f"modify — its intended position cannot be preserved"
                )
    elif not (inline.startswith("[") and inline.endswith("]")):
        raise LanguageConfigError(
            f"{config_path}: unsupported inline value for 'languages' "
            f"({inline!r}); refusing to modify"
        )

    newline = "\r\n" if raw.find("\r\n") != -1 else "\n"
    rendered = f"languages:{newline}" + "".join(
        f"- {name}{newline}" for name in languages
    )
    new_text = "".join(lines[:start]) + rendered + "".join(lines[end:])
    # Do not draw a final newline if there was none.
    if not raw.endswith(("\n", "\r")) and new_text.endswith(newline):
        new_text = new_text[: -len(newline)]

    _atomic_write(config_path, new_text)


def _atomic_write(target: Path, text: str) -> None:
    """Temp in the same directory → fsync → replace (A3).

    A partial write is impossible: on any error the source stays
    byte-for-byte the same, and the temp file is removed.
    """
    try:
        current_mode: int | None = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        current_mode = None

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # Keep the original mode; for a new file — private 0600.
        os.chmod(tmp_path, current_mode if current_mode is not None else _FILE_MODE)
        os.replace(str(tmp_path), str(target))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def create_config(
    project_root: Path | str,
    languages: list[str],
    *,
    serena_command: str | None = None,
) -> Path:
    """Create a valid ``project.yml`` with Serena's command (A1).

    We do not write the file by hand. A minimal config with only
    ``languages`` is invalid: ``ProjectConfig.FIELDS_WITHOUT_DEFAULTS =
    {"project_name", "languages"}`` (``serena_config.py:341``) — these
    fields get no default values, and Serena would not load such a file.
    The official ``serena project create`` command fills the whole
    template and stays true to its own format across updates.
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    root = Path(project_root).resolve()
    exe = shutil.which(serena_command or "serena")
    if not exe:
        raise LanguageConfigError(
            "serena is not installed (or not on PATH), cannot create a "
            "project config; install it with 'uv tool install -p 3.13 "
            "serena-agent'"
        )
    argv = [exe, "project", "create", str(root)]
    for name in languages:
        argv += ["--language", name]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            argv, capture_output=True, text=True, timeout=180
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise LanguageConfigError(f"'serena project create' failed: {e}") from e
    if proc.returncode != 0:
        raise LanguageConfigError(
            f"'serena project create' failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )

    target = project_config_path(root)
    if not target.exists():
        raise LanguageConfigError(
            f"'serena project create' reported success but {target} is missing"
        )
    # Serena creates the file with its own permissions; make them private.
    target.parent.chmod(_DIR_MODE)
    target.chmod(_FILE_MODE)
    # We set the language order: for Serena the first in the list is the
    # default language, and her CLI does not guarantee argument order.
    if read_languages(target) != languages:
        write_languages(target, languages)
    return target


def apply_language_changes(config_path: Path, *, desired: list[str]) -> bool:
    """Bring the language list to ``desired``.

    Returns:
        ``True`` if the file was changed, ``False`` if it was already in
        the desired state (then the file is not touched at all — mtime
        and content are preserved).

    Raises:
        LanguageConfigError: ``desired`` is empty (symbol tools would
            stop working) or the file is ambiguous.
    """
    if not desired:
        raise LanguageConfigError(
            "refusing to leave the project with no languages: symbol tools "
            "would stop working. Use 'deep-worker lsp disable' instead."
        )
    if read_languages(config_path) == desired:
        return False
    write_languages(config_path, desired)
    return True


__all__ = [
    "LanguageConfigError",
    "SERENA_PROJECT_FILE",
    "apply_language_changes",
    "create_config",
    "detected_languages",
    "language_composition",
    "merge_languages",
    "project_config_path",
    "read_languages",
    "validate_language_name",
    "write_languages",
]
