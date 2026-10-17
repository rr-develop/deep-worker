"""PyPI-packaging config guards (static, fast).

These tests assert the ``deep_worker_cli/pyproject.toml`` shape that makes
the package PyPI-installable:

- no ``file://`` / direct-reference dependencies (PyPI rejects those);
- no external ``deepagents`` runtime dependency — the in-tree fork is
  vendored into the wheel/sdist instead (owner decision 2026-08-11);
- the vendored tree and LICENSE are force-included in both artifacts;
- the local-dev seam keeps the fork importable (test dependency-group +
  ``[tool.uv.sources]``) without polluting METADATA;
- the copied ``deep_worker_cli/LICENSE`` is byte-for-byte the root LICENSE
  (drift guard).

Paths are relative to this file so the suite runs identically in a
development tree and the main tree.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# deep_worker/deep_worker_cli/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
# <repo>/LICENSE (two levels above deep_worker_cli: deep_worker/..)
REPO_ROOT = PROJECT_ROOT.parent.parent
ROOT_LICENSE = REPO_ROOT / "LICENSE"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _wheel_config(p: dict) -> dict:
    return p.get("tool", {}).get("hatch", {}).get("build", {}).get(
        "targets", {}
    ).get("wheel", {})


def _sdist_config(p: dict) -> dict:
    return p.get("tool", {}).get("hatch", {}).get("build", {}).get(
        "targets", {}
    ).get("sdist", {})


def test_no_direct_reference_dependencies() -> None:
    """No dependency may use file:// or any @ direct reference."""
    deps = _load_pyproject()["project"]["dependencies"]
    offenders = [d for d in deps if "@" in d or d.startswith("file:")]
    assert not offenders, f"direct-reference deps present: {offenders}"


def test_rpyc_async_declared_as_a_publishable_pin() -> None:
    """The rpyc-async DECLARATION must stay publishable: a plain version pin.

    This is a packaging constraint on this file, nothing more. A ``file://``
    or ``@`` direct reference cannot be uploaded to PyPI, so the declaration
    has to be a version pin — and that pin is the FALLBACK.

    It is NOT a prohibition on running from sources: a local source checkout
    of rpyc_async takes priority over any installed distribution at
    runtime (owner decision 2026-10-11, implemented in
    ``deep_worker_cli/rpyc_async_source.py`` — see its module docstring,
    which also corrects the retracted "PyPI only (D5)" claim).
    """
    deps = _load_pyproject()["project"]["dependencies"]
    rpyc = [d for d in deps if d.lower().startswith("rpyc-async")]
    assert len(rpyc) == 1, f"expected exactly one rpyc-async dep, got {rpyc}"
    assert "file://" not in rpyc[0]
    assert "/" not in rpyc[0]
    assert ">=" in rpyc[0] or "==" in rpyc[0]


def test_no_external_deepagents_dependency() -> None:
    """deepagents must NOT be a runtime dependency (it is vendored)."""
    deps = _load_pyproject()["project"]["dependencies"]
    da = [d for d in deps if d.lower().startswith("deepagents")]
    assert not da, f"deepagents must not be in dependencies: {da}"


def test_deepagents_in_test_group_with_uv_source() -> None:
    """Local-dev seam: fork stays installable via test group + uv.sources."""
    p = _load_pyproject()
    groups = p.get("dependency-groups", {})
    test_group = groups.get("test", [])
    assert any(
        d.lower().startswith("deepagents") for d in test_group
    ), "deepagents must be in [dependency-groups] test"

    sources = p.get("tool", {}).get("uv", {}).get("sources", {})
    da_source = sources.get("deepagents")
    assert da_source is not None, "missing [tool.uv.sources] deepagents"
    assert da_source.get("path", "").endswith("deepagents")
    assert da_source.get("editable") is True


def test_fork_deps_moved_into_cli() -> None:
    """Deps actually imported by the vendored fork must be runtime deps."""
    deps = _load_pyproject()["project"]["dependencies"]
    joined = "\n".join(deps)
    assert "langchain-anthropic" in joined, "fork graph.py imports langchain_anthropic"
    assert "wcmatch" in joined, "fork backends/filesystem.py imports wcmatch"
    assert "langchain-core" in joined, "fork graph.py uses langchain_core"


def test_langgraph_pinned_to_host_compatible_pair() -> None:
    """langgraph must be pinned to the pair compatible with langchain 1.2.9.

    langgraph-prebuilt>=1.0.9 imports ExecutionInfo from langgraph.runtime,
    which does not exist in any langgraph 1.0.x release — an unpinned
    ``pip install deep-worker`` would resolve prebuilt 1.0.13 + langgraph
    1.0.10 and crash at import time (verified 2026-08-11). The host pair
    langgraph 1.0.8 + langgraph-prebuilt 1.0.7 is pinned explicitly.
    """
    deps = _load_pyproject()["project"]["dependencies"]
    joined = "\n".join(deps)
    assert "langgraph==" in joined, "langgraph must be pinned with =="
    assert "langgraph-prebuilt==" in joined, "langgraph-prebuilt must be pinned with =="
    lg = next(d for d in deps if d.startswith("langgraph=="))
    pb = next(d for d in deps if d.startswith("langgraph-prebuilt=="))
    assert lg == "langgraph==1.0.8", f"unexpected langgraph pin: {lg}"
    assert pb == "langgraph-prebuilt==1.0.7", f"unexpected langgraph-prebuilt pin: {pb}"


def test_force_include_vendored_deepagents_and_license() -> None:
    """Wheel AND sdist must force-include the vendored tree and LICENSE."""
    p = _load_pyproject()
    for name, cfg in (("wheel", _wheel_config(p)), ("sdist", _sdist_config(p))):
        fi = cfg.get("force-include", {})
        assert "deepagents" in fi, f"{name}: missing force-include deepagents"
        assert fi["deepagents"] == "deepagents"
        assert "LICENSE" in fi, f"{name}: missing force-include LICENSE"
        assert fi["LICENSE"] == "LICENSE"


def test_deepagents_symlink_exists() -> None:
    """In-tree symlink deep_worker_cli/deepagents -> ../deepagents/deepagents.

    This is what makes the single ``deepagents`` force-include path valid
    both in the monorepo checkout and inside the extracted sdist.
    """
    link = PROJECT_ROOT / "deepagents"
    assert link.is_symlink(), f"expected symlink at {link}"
    target = (link.parent / link.readlink()).resolve()
    expected = (PROJECT_ROOT.parent / "deepagents" / "deepagents").resolve()
    assert target == expected, f"symlink points to {target}, expected {expected}"
    assert (target / "__init__.py").exists()


def test_no_allow_direct_references() -> None:
    """[tool.hatch.metadata] allow-direct-references must be gone."""
    p = _load_pyproject()
    meta = p.get("tool", {}).get("hatch", {}).get("metadata", {})
    assert "allow-direct-references" not in meta


def test_no_repos_rpyc_async_in_extra_paths() -> None:
    """``ty`` extra-paths must not hard-code a local rpyc_async path.

    A checked-in absolute path would make type checking depend on one
    machine's layout, which breaks CI (it runs under ``$GITHUB_WORKSPACE``).
    ``ty`` therefore resolves rpyc_async from the installed distribution.

    Again a type-checking arrangement only, not a statement about runtime:
    the checkout wins at runtime via ``rpyc_async_source``, so ``ty`` may be
    reading an older API surface than the code runs against.
    """
    p = _load_pyproject()
    extra = p.get("tool", {}).get("ty", {}).get("environment", {}).get(
        "extra-paths", []
    )
    assert not any("rpyc_async" in path for path in extra)


# ===========================================================================
# Web UI requirements are CORE
# ===========================================================================
# The web UI ships inside the wheel (see hooks/stage_providers.py), and
# its third-party requirements are ordinary core dependencies: `pip install
# deep-worker` must yield a working `deep-worker web` without the user
# knowing an extra exists (owner requirement 2026-10-13).
#
# The set is derived from what the code actually IMPORTS — not from
# the web UI's requirements.txt, which lists packages the imports never
# ask for.

# Measured by walking every import statement in the web UI's app/** and
# run.py (23 files, 35 distinct top-level names), then subtracting what core
# already declared: uvicorn, starlette, pydantic, mcp, rpyc-async.
_WEB_REQUIREMENTS_IN_CORE = {"fastapi", "pydantic-settings", "anyio"}

# Packages that LOOK like web requirements but are not imported anywhere:
#   jinja2          — Jinja2Templates came from fastapi.templating, and the
#                     dead app/templates/ branch that used it is gone
#   python-multipart, websockets
#                   — no import site; WS goes through Starlette/FastAPI
#
# "Not DECLARED" is not the same as "not installed": resolving the core set
# pulls python-multipart in transitively via fastapi/mcp. The rule here is
# that we do not declare what we do not import — a direct declaration is a
# promise to keep using it, and these are not ours to promise.
_NOT_WEB_REQUIREMENTS = {"jinja2", "python-multipart", "websockets"}


def _requirement_name(spec: str) -> str:
    """Bare distribution name of a requirement string."""
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "[", ";"):
        spec = spec.split(sep)[0]
    return spec.strip().lower()


def test_web_requirements_are_core_dependencies() -> None:
    """Every third-party package the web code imports must be in core.

    A missing entry means ``pip install deep-worker`` then ``deep-worker
    web`` dies on an ImportError — which is exactly the failure this work
    exists to remove.
    """
    core = {
        _requirement_name(d)
        for d in _load_pyproject()["project"]["dependencies"]
    }
    missing = _WEB_REQUIREMENTS_IN_CORE - core
    assert not missing, (
        f"the web UI imports {sorted(missing)}, which core does not "
        f"declare. `pip install deep-worker` would install a web UI that "
        f"cannot start. Core deps present: {len(core)}."
    )


def test_no_web_extra_is_declared() -> None:
    """The requirements must NOT be hidden behind an extra.

    Owner requirement 2026-10-13: a user does not work "without the web",
    so ``pip install deep-worker`` has to be enough. An extra would make
    the working configuration the one nobody types.
    """
    extras = _load_pyproject()["project"].get("optional-dependencies", {})
    assert "web" not in extras, (
        f"a [web] extra is declared ({extras.get('web')}). The web "
        f"requirements belong in core so plain `pip install deep-worker` "
        f"gives a working `deep-worker web`."
    )


def test_web_requirements_are_declared_once_each() -> None:
    """Positive control for the two tests above.

    Moving the three names into core by ADDING them while an extra still
    listed them would satisfy "present in core" and "no extra" separately
    yet leave two pins for one package — and a second, looser pin can widen
    a constraint core narrowed on purpose (this repo pins ``pydantic<2.14``
    for exactly that reason).
    """
    p = _load_pyproject()
    core_specs = p["project"]["dependencies"]
    names = [_requirement_name(d) for d in core_specs]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, (
        f"core declares {sorted(duplicates)} more than once; two pins for "
        f"one package let the looser one win."
    )

    all_extra_names = {
        _requirement_name(d)
        for specs in p["project"].get("optional-dependencies", {}).values()
        for d in specs
    }
    restated = _WEB_REQUIREMENTS_IN_CORE & all_extra_names
    assert not restated, (
        f"{sorted(restated)} appear in BOTH core and an extra. Core is the "
        f"single place the web requirements are declared."
    )


def test_core_omits_packages_the_web_code_never_imports() -> None:
    """Guard the specific false positives, so a later "looks related, add
    it" edit has to argue with a named measurement.

    This matters more now than it did behind an extra: these would be
    downloaded by every single user of the package.
    """
    core = {
        _requirement_name(d)
        for d in _load_pyproject()["project"]["dependencies"]
    }
    offenders = core & _NOT_WEB_REQUIREMENTS
    assert not offenders, (
        f"core declares {sorted(offenders)}, which the web UI never "
        f"imports. jinja2 in particular is reached only through "
        f"fastapi.templating, and the app/templates/ branch that touched it "
        f"has been removed."
    )


# ===========================================================================
# rpyc-async floor
# ===========================================================================


def test_rpyc_async_floor_requires_async_call() -> None:
    """The declared floor must be a version that HAS ``async_call``.

    Measured against PyPI: 1.0.0 exposes neither
    ``rpyc_async.core.netref.async_call`` nor
    ``rpyc_async.utils.helpers.fire_and_forget_call``; 1.1.0 (uploaded
    2026-10-11) exposes both. the web UI's ``mcp_service``
    calls ``async_call``, so a host that resolves the published 1.0.0
    installs a web UI that cannot send inter-agent messages — and nothing
    in the dependency metadata says so.

    The pin is still only the FALLBACK (a local rpyc_async source checkout wins at
    runtime), which is exactly why it must not understate the requirement:
    the fallback is what a pip user gets.
    """
    deps = _load_pyproject()["project"]["dependencies"]
    rpyc = [d for d in deps if d.lower().startswith("rpyc-async")]
    assert len(rpyc) == 1, f"expected exactly one rpyc-async dep, got {rpyc}"

    spec = rpyc[0]
    assert ">=1.0.0" not in spec, (
        f"rpyc-async floor is still {spec!r}. 1.0.0 lacks "
        f"netref.async_call, which the web UI calls, so this floor permits "
        f"an installation where inter-agent messaging is silently broken."
    )
    assert ">=1.1.0" in spec, (
        f"rpyc-async floor is {spec!r}; expected >=1.1.0, the first "
        f"published version with netref.async_call."
    )


def test_rpyc_async_comment_does_not_still_await_the_release() -> None:
    """The comment above the pin set its own condition — "raise this floor
    once 1.1.0 is on PyPI". 1.1.0 is on PyPI, so leaving the text in place
    would make the file instruct a reader to do what has been done."""
    text = PYPROJECT.read_text()
    assert "once 1.1.0 is on PyPI" not in text, (
        "pyproject.toml still says to raise the floor 'once 1.1.0 is on "
        "PyPI'. It was published 2026-10-11 and the floor is now >=1.1.0; "
        "the instruction is stale."
    )


def test_license_copy_matches_root() -> None:
    """deep_worker_cli/LICENSE must be byte-for-byte the root LICENSE."""
    cli_license = PROJECT_ROOT / "LICENSE"
    assert cli_license.exists(), "missing deep_worker_cli/LICENSE"
    assert ROOT_LICENSE.exists(), "missing root LICENSE"
    assert cli_license.read_bytes() == ROOT_LICENSE.read_bytes(), (
        "deep_worker_cli/LICENSE drifted from root LICENSE"
    )
