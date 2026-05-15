"""Pin-drift guard.

Background
----------
The repo declares dependencies in two ``pyproject.toml`` files
(``deepagents/pyproject.toml`` and ``deep_worker_cli/pyproject.toml``)
with EXACT pins (``==X.Y.Z``). The same packages are also installed
system-wide on the host where pytest runs.

If those two diverge — e.g. somebody bumps the pin in pyproject.toml
without re-installing on the host, or somebody updates the host without
bumping the pin — the type-check (which uses uv.lock) will see a
different langchain than pytest, and bugs that depend on API surface
(like the subagent ``recursion_limit`` regression on 2026-04-25) will
appear or disappear depending on which environment the test runner
happened to be in.

This test is the alarm: it asserts every pinned langchain*/langgraph*
in either pyproject.toml matches the version Python actually imports.
If they drift, **both** files (pyproject.toml AND the system install)
need to be brought back in sync — the test names which one is wrong.
"""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Iterator

import pytest


# Files whose ``[project] dependencies`` lists we scan.
_PYPROJECTS = (
    Path("deep_worker/deep_worker_cli/pyproject.toml"),
    Path("deep_worker/deepagents/pyproject.toml"),
)


# We only audit the package families whose drift produced the original
# incident. Other deps in pyproject.toml (rpyc-from-source, textual,
# rich, ...) have either single-version pins of their own or are local
# ``file://`` paths and are not part of the langchain version game.
_AUDITED_PREFIXES = (
    "langchain",
    "langgraph",
)


_PIN_RE = re.compile(
    r"""^\s*       # possible indent
        ['"]
        (?P<name>[A-Za-z0-9._-]+)
        ==
        (?P<version>[A-Za-z0-9._+-]+)
        ['"]
        \s*,?\s*$  # trailing comma optional
    """,
    re.VERBOSE,
)


def _iter_pinned(pyproject: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(package_name, version)`` for every ``"name==version"``
    line found in ``pyproject``. We don't try to parse TOML properly:
    this format is line-oriented and pip-style, ``"X==Y"`` shows up
    only in dependency lists.
    """
    for line in pyproject.read_text().splitlines():
        match = _PIN_RE.match(line)
        if not match:
            continue
        name = match.group("name").lower().replace("_", "-")
        if not name.startswith(_AUDITED_PREFIXES):
            continue
        yield name, match.group("version")


def _all_pinned() -> dict[Path, dict[str, str]]:
    """Return a map ``{pyproject_path: {pkg: version}}`` for every
    pyproject.toml that exists. Missing files are skipped (sibling repo
    layouts may not include both)."""
    result: dict[Path, dict[str, str]] = {}
    for path in _PYPROJECTS:
        if not path.exists():
            continue
        pinned = dict(_iter_pinned(path))
        if pinned:
            result[path] = pinned
    return result


@pytest.mark.parametrize(
    ("pyproject", "package", "pinned_version"),
    [
        (path, pkg, version)
        for path, pinned in _all_pinned().items()
        for pkg, version in pinned.items()
    ],
    ids=lambda v: str(v) if not isinstance(v, Path) else v.parent.name,
)
def test_pinned_version_matches_installed(
    pyproject: Path, package: str, pinned_version: str
) -> None:
    """Every ``langchain*``/``langgraph*`` pin in pyproject.toml must
    match the version of that package importable from the host's
    Python. Drift means the type-check environment (uv.lock-driven)
    and pytest environment (system-installed) will see different APIs
    and produce inconsistent test results.

    Resolution if this fails:

    * If the pin is correct (we want everyone on this version): bring
      the system install in sync — typically ``pip install
      <package>==<pinned_version>``.
    * If the system install is correct (we want to bump): update the
      pin in ``<pyproject>`` AND regenerate ``uv.lock`` in the same
      directory (``uv lock``).
    """
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            f"{package} is not installed on this host; "
            "the optional-dep / vertexai groups can leave it out — "
            "drift check skipped."
        )

    assert installed == pinned_version, (
        f"\n  pyproject  : {pyproject}"
        f"\n  package    : {package}"
        f"\n  pinned     : {pinned_version}"
        f"\n  installed  : {installed}"
        f"\n  -> the type-check env (uv.lock) and pytest env (system pip) "
        f"are out of sync. Re-pin the file or re-install the package, "
        f"whichever direction is intended."
    )


def test_at_least_one_pin_was_audited() -> None:
    """Guard against the parametrized test silently degenerating to
    zero cases (e.g. if both pyprojects are missing or the regex
    breaks). If we audit nothing, we are not actually guarding
    anything — fail loudly."""
    pinned = _all_pinned()
    total = sum(len(v) for v in pinned.values())
    assert total >= 4, (
        f"version-drift guard found only {total} pinned langchain/"
        f"langgraph entries across {list(pinned.keys())} — expected at "
        f"least 4 (langchain-core, langchain, langchain-anthropic, "
        f"langchain-openai). The regex or the pyproject layout changed."
    )
