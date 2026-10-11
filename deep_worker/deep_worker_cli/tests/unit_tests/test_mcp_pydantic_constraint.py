"""Guard: the ``mcp`` pin must stay paired with a pydantic upper bound.

Background
----------
``mcp==1.17.0`` imports a symbol out of pydantic's *private* internals::

    mcp/server/fastmcp/utilities/func_metadata.py:17
    from pydantic._internal._typing_extra import eval_type_backport

pydantic removed that symbol in 2.14.0. pydantic is not pinned in
``[project] dependencies``; it arrives transitively via langchain
(``>=2.7.4``). So a fresh resolve on a clean machine took 2.14.0 and
``import mcp.types`` raised ``ImportError`` during pytest **collection**,
which exits 2 and fails the whole job.

That is why CI went red on commits whose own diff was unrelated (and in
one case docs-only): nothing in the repo changed, the dependency floated
underneath it. Developer machines kept passing because they held an
older pydantic from an earlier install. Bisected boundary: 2.13.0 imports
fine, 2.14.0 does not.

This test fails if the upper bound is ever dropped while the ``mcp`` pin
still needs it, so the next person to touch either line is told which
one they broke instead of discovering it from a red CI run on an
innocent commit.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

# tests/unit_tests/ -> tests/ -> deep_worker_cli/
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: First ``mcp`` release that stopped importing the private pydantic
#: symbol. Once the pin moves here or beyond, the bound may be lifted.
_MCP_FIXED_IN = (1, 30, 0)

#: First pydantic release without ``_typing_extra.eval_type_backport``.
_PYDANTIC_BREAKS_AT = "2.14"


def _dependencies() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return list(data["project"]["dependencies"])


def _requirement(name: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(name)}\s*(?P<spec>[<>=!~].*)?$")
    for dep in _dependencies():
        m = pattern.match(dep.strip())
        if m:
            return dep.strip()
    return None


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def test_mcp_is_still_pinned_below_the_fix() -> None:
    """Sanity: this guard only applies while ``mcp`` predates the fix.

    Without this the test would silently become vacuous after an ``mcp``
    upgrade — passing because it no longer checks anything.
    """
    mcp_req = _requirement("mcp")
    assert mcp_req is not None, "mcp disappeared from [project] dependencies"
    pinned = _version_tuple(mcp_req)
    if pinned >= _MCP_FIXED_IN:
        pytest.fail(
            f"mcp is now pinned at {pinned}, which is >= {_MCP_FIXED_IN} and no "
            f"longer imports pydantic's private eval_type_backport. Drop the "
            f"'pydantic<{_PYDANTIC_BREAKS_AT}' bound and delete this guard."
        )


def test_pydantic_upper_bound_is_declared() -> None:
    """The bound the ``mcp`` pin depends on must be present."""
    req = _requirement("pydantic")
    assert req is not None, (
        "pydantic has no entry in [project] dependencies, so it floats in "
        "transitively via langchain and a fresh resolve will take "
        f"{_PYDANTIC_BREAKS_AT}+. That breaks 'import mcp.types' at pytest "
        "collection time and fails CI on unrelated commits. Restore "
        f"'pydantic<{_PYDANTIC_BREAKS_AT}'."
    )
    assert f"<{_PYDANTIC_BREAKS_AT}" in req, (
        f"pydantic is constrained as {req!r}, which does not exclude "
        f"{_PYDANTIC_BREAKS_AT}. mcp=={_version_tuple(_requirement('mcp') or '')} "
        f"imports pydantic._internal._typing_extra.eval_type_backport, removed "
        f"in {_PYDANTIC_BREAKS_AT}.0."
    )


def test_the_installed_pydantic_can_satisfy_the_private_import() -> None:
    """The environment running this suite must itself be consistent.

    Pins describe intent; this asserts the fact. If the symbol is gone,
    ``import mcp.types`` would die at collection — the failure mode this
    whole module exists to prevent — so catching it here names the cause
    instead of leaving an exit-2 traceback.
    """
    pydantic = pytest.importorskip("pydantic")
    from pydantic._internal import _typing_extra

    assert hasattr(_typing_extra, "eval_type_backport"), (
        f"installed pydantic {pydantic.VERSION} dropped "
        f"_typing_extra.eval_type_backport, which the pinned mcp imports at "
        f"module scope. Reinstall with 'pydantic<{_PYDANTIC_BREAKS_AT}'."
    )
