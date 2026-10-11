"""Provider-from-wheel artifact positive-path tests (REAL build + clean venv).

Exercises the artifact acceptance gates for BOTH artifacts (monorepo wheel AND sdist-derived wheel):

- archive-assert — NO top-level provider dirs
  (``codex_provider/``, ``grok_provider/``,
  ``semantic_search/``, ``deepworker_a2a/``, ``acp/``,
  ``claude_code_provider/``) in wheel OR sdist; the wheel rebuilt from
  the sdist passes the same assert again;
- sdist carries the verified ``vendor_staging/`` payload + ``staging.manifest.json``
  so an extracted-sdist wheel rebuild is self-sufficient (no ``../../``);
- from the isolated venv, ``deep_worker.providers.<pkg>`` imports
  resolve under site-packages; top-level alias ``find_spec`` → None;
- core package imports without any provider extra installed
  (lazy roots must not eager-import heavy submodules).

Like ``test_pypi_build.py`` these require network (build isolation downloads
hatchling; pip resolves dependencies) so they are deselected by default
(``-m 'not integration'``) and marked ``integration``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

import pytest

# deep_worker/deep_worker_cli/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

PROVIDER_PACKAGES = [
    "codex_provider",
    "grok_provider",
    "semantic_search",
    "deepworker_a2a",
    "acp",
    "claude_code_provider",
]
TOP_LEVEL_PROVIDER_DIRS = [pkg + "/" for pkg in PROVIDER_PACKAGES]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(1800),
]


# Source selectors that must never leak from the host into artifact
# subprocesses — the source tree would otherwise shadow the installed wheel.
_SOURCE_SELECTORS = (
    "PYTHONPATH",
    "DEEP_WORKER_SOURCE_ROOT",
    "DEEP_WORKER_PROVIDER_SOURCE",
    "CLAUDE_AGENT_SDK_SRC",
)


def _artifact_env(
    extra: dict[str, str | None] | None = None,
    *,
    keep_source_selectors: bool = False,
) -> dict[str, str]:
    """Host env minus every source selector (DESIGN §3.1 / §5.8).

    ``extra`` values of ``None`` REMOVE the variable from the subprocess
    environment; string values SET it. ``keep_source_selectors=True`` keeps
    the source selectors (never pointing at the host source tree).
    """
    env = dict(os.environ)
    if not keep_source_selectors:
        for key in _SOURCE_SELECTORS:
            env.pop(key, None)
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env


def _run(cmd: list[str], cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_artifact_env(),
        timeout=timeout,
    )


def _build_dist(outdir: Path, cwd: Path) -> tuple[Path, Path]:
    res = _run([sys.executable, "-m", "build", "--outdir", str(outdir)], cwd=cwd)
    assert res.returncode == 0, f"build failed:\n{res.stdout}\n{res.stderr}"
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected 1 wheel, got {wheels}"
    assert len(sdists) == 1, f"expected 1 sdist, got {sdists}"
    return wheels[0], sdists[0]


def _wheel_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def _sdist_entries(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as tf:
        names = tf.getnames()
    # Strip the top-level "<name>-<version>/" directory prefix so entries are
    # comparable with wheel paths.
    stripped: list[str] = []
    for n in names:
        parts = n.split("/", 1)
        stripped.append(parts[1] if len(parts) == 2 else parts[0])
    return stripped


def _assert_no_top_level_provider_dirs(entries: list[str], label: str) -> None:
    """Archive-assert: providers only under their canonical location.

    For the wheel that is ``deep_worker/providers/<pkg>/``; for the sdist the
    payload is shipped as ``vendor_staging/`` (mapped into the wheel at build
    time), so the sdist assert checks the vendored prefix instead.
    """
    for prefix in TOP_LEVEL_PROVIDER_DIRS:
        offenders = [e for e in entries if e.startswith(prefix)]
        assert not offenders, f"{label}: top-level provider dir entries: {offenders}"
    # Canonical prefixed entries must exist. In the sdist the payload prefix is
    # `vendor_staging/` (plus the staging integrity marker next to it).
    in_sdist = any(e.startswith("vendor_staging/") for e in entries)
    canonical_prefix = "vendor_staging/" if in_sdist else "deep_worker/providers/"
    for pkg in PROVIDER_PACKAGES:
        assert any(
            e.startswith(f"{canonical_prefix}{pkg}/") for e in entries
        ), f"{label}: missing {canonical_prefix}{pkg}/"
    if canonical_prefix == "deep_worker/providers/":
        assert "deep_worker/__init__.py" in entries, (
            f"{label}: missing deep_worker/__init__.py"
        )
        assert "deep_worker/providers/__init__.py" in entries, (
            f"{label}: missing deep_worker/providers/__init__.py"
        )


def _extract_sdist(sdist: Path, dest: Path) -> Path:
    dest.mkdir()
    res = _run(["tar", "-xzf", str(sdist), "-C", str(dest)], cwd=dest.parent)
    assert res.returncode == 0, f"tar extract failed: {res.stderr}"
    return dest / next(d for d in dest.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# No-network socket-allowlist hook (DESIGN §5.8, rev 4 precision-fix)
# ---------------------------------------------------------------------------
# Every in-venv assertion script runs under an ACTIVE outbound-socket guard:
# only loopback (127.x / ::1 / localhost) connections are permitted — RPyC and
# A2A live on localhost — anything else raises AssertionError. This turns the
# "we promise not to call the LLM" rule into an enforced invariant.
# The guard source lives in tests/socket_guard.py (unit-testable canaries +
# shared by the artifact scripts).

from tests.socket_guard import GUARD_SOURCE as _SOCKET_GUARD


def _py_script(body: str) -> str:
    """Wrap an assertion body in the no-network socket-allowlist hook."""
    return _SOCKET_GUARD + body


_ORIGIN_ASSERT = """\
# source isolation: no host source selectors; every module under the venv
import os as _os, sys as _sys
from pathlib import Path as _Path
for _k in ('PYTHONPATH', 'DEEP_WORKER_SOURCE_ROOT',
           'DEEP_WORKER_PROVIDER_SOURCE', 'CLAUDE_AGENT_SDK_SRC'):
    assert _k not in _os.environ, (_k, _os.environ.get(_k))
assert not any('/repos' in p for p in _sys.path), _sys.path
import deep_worker as _dw_mod, deep_worker_cli as _dwc_mod
_venv_prefix = _Path(_sys.prefix)
for _m in (_dw_mod, _dwc_mod):
    _mf = _Path(_m.__file__).resolve()
    assert _mf.is_relative_to(_venv_prefix), (_m.__name__, _mf)
    assert '/repos' not in str(_mf), (_m.__name__, _mf)
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_archive_assert_both_artifacts_and_sdist_rebuild(tmp_path: Path) -> None:
    """Wheel AND sdist alias-free; sdist-derived wheel passes again."""
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel, sdist = _build_dist(dist, cwd=PROJECT_ROOT)

    _assert_no_top_level_provider_dirs(_wheel_entries(wheel), "monorepo wheel")
    sdist_entries = _sdist_entries(sdist)
    _assert_no_top_level_provider_dirs(sdist_entries, "sdist")

    # sdist must carry the self-sufficient staging payload + integrity marker.
    sdist_root = _extract_sdist(sdist, tmp_path / "extracted")
    assert (sdist_root / "vendor_staging").is_dir(), (
        "sdist must contain vendor_staging/ payload"
    )
    assert (sdist_root / "staging.manifest.json").is_file(), (
        "sdist must contain staging.manifest.json integrity marker"
    )

    # Rebuild wheel from the extracted sdist → same archive assert passes.
    dist2 = tmp_path / "dist2"
    dist2.mkdir()
    wheel2, _ = _build_dist(dist2, cwd=sdist_root)
    _assert_no_top_level_provider_dirs(_wheel_entries(wheel2), "sdist-derived wheel")


def test_clean_venv_prefix_imports_and_no_top_level_aliases(tmp_path: Path) -> None:
    """Installed wheel — prefixed imports work, aliases absent."""
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel, _ = _build_dist(dist, cwd=PROJECT_ROOT)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"

    res = _run([str(pip), "install", str(wheel)], cwd=REPO_ROOT, timeout=900)
    assert res.returncode == 0, f"pip install failed:\n{res.stdout}\n{res.stderr}"

    # Each provider package resolves under deep_worker/providers/<pkg> and no
    # top-level alias exists in the installed wheel.
    code = _py_script(
        "\n".join(
            [
                "import importlib, importlib.util, sys",
                "from pathlib import Path",
                f"pkgs = {PROVIDER_PACKAGES!r}",
                "for p in pkgs:",
                "    m = importlib.import_module(f'deep_worker.providers.{p}')",
                "    assert m.__file__ is not None",
                "    assert 'site-packages' in str(Path(m.__file__).resolve()), m.__file__",
                "    assert importlib.util.find_spec(p) is None, f'top-level alias {p}'",
                "print('PREFIX-OK')",
            ]
        )
    )
    res = _run([str(py), "-c", code], cwd=venv_dir, timeout=300)
    assert res.returncode == 0, f"prefix import check failed:\n{res.stdout}\n{res.stderr}"
    assert "PREFIX-OK" in res.stdout


def test_clean_venv_core_import_without_provider_extras(tmp_path: Path) -> None:
    """Core install (no [a2a]/[semantic-search] extras) imports cleanly.

    Lazy roots (§3.2/§3.6) must not eager-import heavy submodules, and no
    provider extra dependency is installed in this venv.
    """
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel, _ = _build_dist(dist, cwd=PROJECT_ROOT)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"

    res = _run([str(pip), "install", str(wheel)], cwd=REPO_ROOT, timeout=900)
    assert res.returncode == 0, f"pip install failed:\n{res.stdout}\n{res.stderr}"

    code = _py_script(
        "import deep_worker_cli\n"
        "import sys\n"
        "assert 'claude_agent_sdk' not in sys.modules, 'eager claude_agent_sdk'\n"
        "assert 'llama_index' not in sys.modules, 'eager llama_index'\n"
        "import deep_worker\n"
        "import deep_worker.providers\n"
        "print('CORE-OK')\n"
    )
    res = _run([str(py), "-c", code], cwd=venv_dir, timeout=300)
    assert res.returncode == 0, f"core import check failed:\n{res.stdout}\n{res.stderr}"
    assert "CORE-OK" in res.stdout


def test_parallel_wheel_builds_no_conflict_and_git_status_clean(
    tmp_path: Path,
) -> None:
    """Two concurrent wheel builds from the monorepo.

    - both succeed and produce exactly one wheel each;
    - NO source-tree staging appears (``vendor_staging/`` / ``staging.manifest.json``
      must not be created in the tree — staging lives in per-process temp dirs);
    - ``git status --porcelain`` of the tree is byte-identical before and after
      (builds leave the working tree clean).
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # never leak the source tree into subprocesses

    def git_status() -> str:
        res = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
        )
        assert res.returncode == 0, f"git status failed: {res.stderr}"
        return res.stdout

    before = git_status()

    dist1 = tmp_path / "dist1"
    dist2 = tmp_path / "dist2"
    dist1.mkdir()
    dist2.mkdir()
    common = [sys.executable, "-m", "build", "--wheel"]
    p1 = subprocess.Popen(
        common + ["--outdir", str(dist1)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    p2 = subprocess.Popen(
        common + ["--outdir", str(dist2)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out1, err1 = p1.communicate(timeout=900)
    out2, err2 = p2.communicate(timeout=900)
    assert p1.returncode == 0, f"parallel build 1 failed:\n{out1}\n{err1}"
    assert p2.returncode == 0, f"parallel build 2 failed:\n{out2}\n{err2}"
    assert len(list(dist1.glob("*.whl"))) == 1, list(dist1.glob("*"))
    assert len(list(dist2.glob("*.whl"))) == 1, list(dist2.glob("*"))

    assert not (PROJECT_ROOT / "vendor_staging").exists(), (
        "parallel builds must not leave source-tree staging"
    )
    assert not (PROJECT_ROOT / "staging.manifest.json").exists(), (
        "parallel builds must not leave the staging integrity marker"
    )
    assert git_status() == before, "parallel builds changed git status of the tree"


# ---------------------------------------------------------------------------
# Real-Hatch staging lifecycle: finalize on success, atexit on
# failure-after-initialize (DESIGN §3.4 lifetime contract)
# ---------------------------------------------------------------------------


def _make_hatch_tree(tmp_path: Path) -> Path:
    """Build a self-contained hatch project exercising the REAL staging hook.

    Layout mirrors the monorepo (``<tree>/__init__.py`` + ``<tree>/providers/``
    + ``<tree>/proj/`` with the hook + manifest).  ``stage_providers.py`` is a
    byte-identical copy of the production hook; a test-only, env-gated
    injection is appended so a real ``python -m build`` can be made to fail in
    the ARTIFACT phase — after ``initialize()`` has returned — without touching
    production code.
    """
    hook_src = PROJECT_ROOT / "hooks" / "stage_providers.py"
    assert hook_src.is_file(), f"missing production hook: {hook_src}"

    tree = tmp_path / "tree"
    tree.mkdir(parents=True)
    (tree / "__init__.py").write_text("", encoding="utf-8")  # deep_worker/
    providers = tree / "providers"
    providers.mkdir()
    (providers / "__init__.py").write_text("", encoding="utf-8")
    for pkg in PROVIDER_PACKAGES:
        pkg_dir = providers / pkg
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(f"VALUE_{pkg}\n", encoding="utf-8")
        (pkg_dir / "py.typed").write_text("", encoding="utf-8")
        (pkg_dir / "alpha.py").write_text(
            "def alpha() -> int:\n    return 1\n", encoding="utf-8"
        )

    proj = tree / "proj"
    hooks = proj / "hooks"
    hooks.mkdir(parents=True)
    hook_copy = hooks / "stage_providers.py"
    shutil.copy2(hook_src, hook_copy)
    # Test-only, env-gated injections (never active in production: the env
    # vars are only set by the failure/canary tests below).
    hook_copy.write_text(
        hook_copy.read_text(encoding="utf-8")
        + (
            "\n"
            "# --- test-only injections (never active in production) ---\n"
            "import os as _os\n"
            "if _os.environ.get('DW_TEST_INJECT_FAILURE'):\n"
            "    _orig_initialize = StageProvidersHook.initialize\n"
            "    def _injecting_initialize(self, version, build_data):\n"
            "        _orig_initialize(self, version, build_data)\n"
            "        build_data['force_include']['/nonexistent-force-include-target'] = 'x'\n"
            "    StageProvidersHook.initialize = _injecting_initialize\n"
            "if _os.environ.get('DW_TEST_FINALIZE_CANARY'):\n"
            "    _orig_finalize = StageProvidersHook.finalize\n"
            "    def _canary_finalize(self, version, build_data, artifact):\n"
            "        _orig_finalize(self, version, build_data, artifact)\n"
            "        from pathlib import Path as _P\n"
            "        _P(_os.environ['DW_TEST_FINALIZE_CANARY']).write_text(\n"
            "            'finalize-called\\n', encoding='utf-8')\n"
            "    StageProvidersHook.finalize = _canary_finalize\n"
        ),
        encoding="utf-8",
    )

    # Generate the two-way manifest via the real hook CLI.
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage_providers", hook_src)
    assert spec and spec.loader
    hook_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook_mod)
    rc = hook_mod._cli(["--update", "--project-root", str(proj)])
    assert rc == 0, "manifest --update failed in hatch tree"

    (proj / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['hatchling']\n"
        "build-backend = 'hatchling.build'\n"
        "\n"
        "[project]\n"
        "name = 'deep-worker-hatch-tree'\n"
        "version = '0.0.1'\n"
        "\n"
        "[tool.hatch.build.hooks.custom]\n"
        "path = 'hooks/stage_providers.py'\n",
        encoding="utf-8",
    )
    return proj


def _temp_leftovers() -> list[Path]:
    temp_root = Path(tempfile.gettempdir())
    return sorted(temp_root.glob("dw-provider-stage.*"))


def test_real_hatch_success_invokes_finalize(tmp_path: Path) -> None:
    """A SUCCESSFUL real hatch build must call finalize() — proven by
    BOTH direct evidence (a canary written from an overridden ``finalize``
    after the real method ran) AND no ``dw-provider-stage.*`` temp surviving
    the build process."""
    proj = _make_hatch_tree(tmp_path)
    before = _temp_leftovers()
    dist = tmp_path / "dist"
    dist.mkdir()
    canary = tmp_path / "finalize.canary"
    env = _artifact_env({"DW_TEST_FINALIZE_CANARY": str(canary)})

    res = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=proj,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert res.returncode == 0, f"hatch build failed:\n{res.stdout}\n{res.stderr}"
    assert len(list(dist.glob("*.whl"))) == 1
    # Direct evidence: hatchling really invoked finalize() on our hook.
    assert canary.exists() and canary.read_text(encoding="utf-8").strip() == (
        "finalize-called"
    ), f"finalize canary missing: {canary}"
    assert _temp_leftovers() == before, (
        f"successful build leaked temps: {_temp_leftovers()}"
    )


def test_real_hatch_failure_after_initialize_leaves_no_temp(tmp_path: Path) -> None:
    """A real hatch build failing in the ARTIFACT phase (after
    ``initialize()`` succeeded) must leave NO ``dw-provider-stage.*`` temp by
    process exit.

    hatchling calls ``finalize()`` only on success, so the only cleanup for
    this lifecycle is the emergency ``atexit`` handler — this test proves that
    handler runs and removes the temp before the build process exits.
    """
    proj = _make_hatch_tree(tmp_path)
    before = _temp_leftovers()
    dist = tmp_path / "dist"
    dist.mkdir()

    env = _artifact_env({"DW_TEST_INJECT_FAILURE": "1"})
    res = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=proj,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert res.returncode != 0, (
        "injected artifact failure must fail the build:\n"
        f"{res.stdout}\n{res.stderr}"
    )
    assert "Forced include not found" in res.stderr, res.stderr
    assert _temp_leftovers() == before, (
        f"failed build leaked temps: {_temp_leftovers()}"
    )


# ---------------------------------------------------------------------------
# Coexistence with legacy top-level name owners
# ---------------------------------------------------------------------------


def _make_fixture_wheel(
    dist_name: str, pkg_name: str, marker: str, outdir: Path
) -> Path:
    """Build a deterministic local wheel owning a legacy top-level name.

    The fixture is built by hand (zipfile + RECORD) so the coexistence gate
    needs NO network and NO build backend — pip accepts the artifact as-is.
    """
    norm = dist_name.replace("-", "_")
    dist_info = f"{norm}-0.0.1.dist-info"
    wheel_path = outdir / f"{norm}-0.0.1-py3-none-any.whl"

    init_py = f"FIXTURE_OWNER = {marker!r}\n"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {dist_name}\n"
        "Version: 0.0.1\n"
        f"Summary: deterministic fixture owning legacy top-level name {pkg_name}\n"
    )
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: deep-worker-coexistence-fixture\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )
    files: list[tuple[str, bytes]] = [
        (f"{pkg_name}/__init__.py", init_py.encode()),
        (f"{dist_info}/METADATA", metadata.encode()),
        (f"{dist_info}/WHEEL", wheel_meta.encode()),
    ]

    def _record_line(rel: str, data: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        return f"{rel},sha256={digest.rstrip(b'=').decode()},{len(data)},"

    record_lines = [_record_line(rel, data) for rel, data in files]
    record_lines.append(f"{dist_info}/RECORD,,")
    files.append((f"{dist_info}/RECORD", ("\n".join(record_lines) + "\n").encode()))

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, data in files:
            zf.writestr(rel, data)
    return wheel_path


def test_coexistence_with_top_level_name_fixtures(tmp_path: Path) -> None:
    """Prefixed packages coexist with legacy top-level name owners.

    Deterministic LOCAL fixture wheels own the old top-level names
    ``codex_provider`` / ``semantic_search`` (mirroring the real
    PyPI collision). In the same venv:

    - the fixture's top-level import works and resolves under site-packages;
    - ``deep_worker.providers.<pkg>`` works from OUR wheel with a DISTINCT
      origin under ``deep_worker/providers/``;
    - RECORD paths of the two distributions do not overlap;
    - ``pip uninstall`` of the fixtures does NOT damage
      ``deep_worker.providers.*`` (re-import succeeds).
    """
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel, _ = _build_dist(dist, cwd=PROJECT_ROOT)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"

    res = _run([str(pip), "install", str(wheel)], cwd=REPO_ROOT, timeout=900)
    assert res.returncode == 0, f"pip install failed:\n{res.stdout}\n{res.stderr}"

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    fx_ccp = _make_fixture_wheel(
        "codex-provider", "codex_provider", "fixture-ccp", fixtures
    )
    fx_ss = _make_fixture_wheel(
        "semantic-search", "semantic_search", "fixture-ss", fixtures
    )
    for fx in (fx_ccp, fx_ss):
        res = _run([str(pip), "install", str(fx)], cwd=REPO_ROOT, timeout=300)
        assert res.returncode == 0, f"fixture install failed:\n{res.stdout}\n{res.stderr}"

    check = _py_script(
        "import importlib, importlib.metadata as md\n"
        "from pathlib import Path\n"
        "ccp = importlib.import_module('codex_provider')\n"
        "ss = importlib.import_module('semantic_search')\n"
        "assert ccp.FIXTURE_OWNER == 'fixture-ccp', ccp.__file__\n"
        "assert ss.FIXTURE_OWNER == 'fixture-ss', ss.__file__\n"
        "p_ccp = importlib.import_module('deep_worker.providers.codex_provider')\n"
        "p_ss = importlib.import_module('deep_worker.providers.semantic_search')\n"
        "fx_ccp_f = str(Path(ccp.__file__).resolve())\n"
        "fx_ss_f = str(Path(ss.__file__).resolve())\n"
        "our_ccp_f = str(Path(p_ccp.__file__).resolve())\n"
        "our_ss_f = str(Path(p_ss.__file__).resolve())\n"
        "# fixture origins live under site-packages, NOT under deep_worker/\n"
        "assert '/site-packages/' in fx_ccp_f and 'deep_worker' not in fx_ccp_f, fx_ccp_f\n"
        "assert '/site-packages/' in fx_ss_f and 'deep_worker' not in fx_ss_f, fx_ss_f\n"
        "# our origins live under site-packages/deep_worker/providers/\n"
        "assert '/site-packages/deep_worker/providers/' in our_ccp_f, our_ccp_f\n"
        "assert '/site-packages/deep_worker/providers/' in our_ss_f, our_ss_f\n"
        "assert our_ccp_f != fx_ccp_f and our_ss_f != fx_ss_f, 'origins must differ'\n"
        "# RECORD paths must not overlap\n"
        "def _paths(dist_name):\n"
        "    rec = md.distribution(dist_name).read_text('RECORD') or ''\n"
        "    return {ln.split(',', 1)[0] for ln in rec.splitlines() if ln}\n"
        "dw_paths = _paths('deep-worker')\n"
        "assert not (_paths('codex-provider') & dw_paths), 'ccp RECORD overlap'\n"
        "assert not (_paths('semantic-search') & dw_paths), 'ss RECORD overlap'\n"
        "# our wheel serves providers from deep_worker/providers only\n"
        "assert any(p.startswith('deep_worker/providers/codex_provider/')\n"
        "           for p in dw_paths), 'missing prefixed ccp in deep-worker RECORD'\n"
        "print('COEX-OK')\n"
    )
    res = _run([str(py), "-c", check], cwd=venv_dir, timeout=300)
    assert res.returncode == 0, f"coexistence check failed:\n{res.stdout}\n{res.stderr}"
    assert "COEX-OK" in res.stdout

    # Uninstall the fixtures → prefixed imports must remain intact.
    res = _run(
        [str(pip), "uninstall", "-y", "codex-provider", "semantic-search"],
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert res.returncode == 0, f"uninstall failed:\n{res.stdout}\n{res.stderr}"
    after = _py_script(
        "import importlib\n"
        "import deep_worker.providers.codex_provider\n"
        "import deep_worker.providers.semantic_search\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec('codex_provider') is None\n"
        "assert importlib.util.find_spec('semantic_search') is None\n"
        "print('POST-UNINSTALL-OK')\n"
    )
    res = _run([str(py), "-c", after], cwd=venv_dir, timeout=300)
    assert res.returncode == 0, (
        f"post-uninstall re-import failed:\n{res.stdout}\n{res.stderr}"
    )
    assert "POST-UNINSTALL-OK" in res.stdout


# ---------------------------------------------------------------------------
# P0-1 (rev 8 CODE CHANGES REQUESTED): REAL positive-path coverage of
# the artifact acceptance gates on BOTH artifacts, no mocks.
#
# The module-scoped fixtures build the artifacts ONCE and install each wheel
# into its own isolated venv (the two wheels are byte-identical, but the gate
# demands the evidence per artifact — so both are exercised explicitly).
# Every in-venv assertion script runs under the socket allowlist guard
# (loopback-only), so the "we never call the LLM" rule stays an enforced
# invariant (DESIGN §5.8).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> dict[str, Path]:
    """Build the monorepo wheel + sdist and the sdist-derived wheel ONCE."""
    root = tmp_path_factory.mktemp("artifacts")
    dist = root / "dist"
    dist.mkdir()
    wheel, sdist = _build_dist(dist, cwd=PROJECT_ROOT)
    extracted = root / "extracted"
    sdist_root = _extract_sdist(sdist, extracted)
    dist2 = root / "dist2"
    dist2.mkdir()
    wheel2, _ = _build_dist(dist2, cwd=sdist_root)
    return {"wheel": wheel, "wheel_sdist": wheel2, "sdist": sdist}


def _install_venv(
    tmp_path_factory: pytest.TempPathFactory, label: str, install_spec: str
) -> tuple[Path, Path]:
    """Create a clean venv, install *install_spec*, verify ``pip check``."""
    root = tmp_path_factory.mktemp(label)
    venv_dir = root / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    res = _run(
        [str(pip), "install", "-q", "--disable-pip-version-check", install_spec],
        cwd=REPO_ROOT,
        timeout=1500,
    )
    assert res.returncode == 0, f"pip install failed:\n{res.stdout}\n{res.stderr}"
    chk = _run([str(pip), "check"], cwd=REPO_ROOT, timeout=300)
    assert chk.returncode == 0, f"pip check failed:\n{chk.stdout}\n{chk.stderr}"
    return venv_dir, py


@pytest.fixture(scope="module")
def venv_monorepo(tmp_path_factory: pytest.TempPathFactory, artifacts: dict[str, Path]):
    """Clean venv with the monorepo wheel installed (core deps only)."""
    return _install_venv(tmp_path_factory, "venv_monorepo", str(artifacts["wheel"]))


@pytest.fixture(scope="module")
def venv_sdist(tmp_path_factory: pytest.TempPathFactory, artifacts: dict[str, Path]):
    """Clean venv with the sdist-derived wheel installed (core deps only)."""
    return _install_venv(
        tmp_path_factory, "venv_sdist", str(artifacts["wheel_sdist"])
    )


@pytest.fixture(scope="module")
def venv_extras(tmp_path_factory: pytest.TempPathFactory, artifacts: dict[str, Path]):
    """Clean venv with the wheel + [vertexai,a2a,semantic-search] extras."""
    return _install_venv(
        tmp_path_factory,
        "venv_extras",
        f"{artifacts['wheel']}[vertexai,a2a,semantic-search]",
    )


def _run_venv(
    py: Path,
    body: str,
    cwd: Path,
    env_extra: dict[str, str | None] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run an assertion body in the venv under the socket guard.

    ``env_extra`` values of ``None`` REMOVE the variable from the subprocess
    environment (e.g. to unset the host's real ``GOOGLE_API_KEY`` so the
    vertexai ADC path becomes selectable). Source selectors are stripped
    first; ``env_extra`` may deliberately re-add them.
    """
    return subprocess.run(
        [str(py), "-c", _py_script(body)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_artifact_env(env_extra),
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# §5.2 model construction + tool assembly for every API provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("venv_name", ["venv_monorepo", "venv_sdist"])
def test_model_construction_core_providers_both_wheels(
    venv_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """§5.2: deepseek/openai/anthropic/google reach REAL model construction.

    Fake API keys are allowed — construction must NOT call the network (the
    socket guard enforces that). ``bind_tools`` is a real LangChain tool
    assembly on each constructed model (no mocks). Runs on BOTH wheels.
    """
    _venv_dir, py = request.getfixturevalue(venv_name)
    body = (
        _ORIGIN_ASSERT
        + "from deep_worker_cli.config import create_model\n"
        "m1 = create_model(provider='deepseek', model_name='deepseek-chat')\n"
        "assert type(m1).__name__ == 'ReasoningChatDeepSeek', type(m1)\n"
        "m2 = create_model(provider='openai', model_name='gpt-4o')\n"
        "assert type(m2).__name__ == 'ChatOpenAI', type(m2)\n"
        "m3 = create_model(provider='anthropic', model_name='claude-sonnet-4-5')\n"
        "assert type(m3).__name__ == 'ChatAnthropic', type(m3)\n"
        "m4 = create_model(provider='google', model_name='gemini-2.0-flash')\n"
        "assert type(m4).__name__ == 'ChatGoogleGenerativeAI', type(m4)\n"
        "# real tool assembly on every provider\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
        "for m in (m1, m2, m3, m4):\n"
        "    assert m.bind_tools([ping]) is not None\n"
        "print('MODEL-MATRIX-OK')\n"
    )
    res = _run_venv(
        py,
        body,
        cwd=tmp_path,
        env_extra={
            "DEEPSEEK_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
        },
    )
    assert res.returncode == 0, f"model construction failed:\n{res.stdout}\n{res.stderr}"
    assert "MODEL-MATRIX-OK" in res.stdout


def test_model_construction_vertexai_with_extra(
    venv_extras: tuple[Path, Path], tmp_path: Path
) -> None:
    """§5.2: vertexai-with-extra — both the Gemini (ADC) and the
    ``ChatAnthropicVertex`` (model-garden) construction paths.

    The [vertexai] extra must be installed (fixture); ``GOOGLE_API_KEY`` is
    explicitly REMOVED so ``has_vertex_ai`` selects ADC, and the model-garden
    path requires ``GOOGLE_CLOUD_LOCATION``.
    """
    _venv_dir, py = venv_extras
    body = (
        _ORIGIN_ASSERT
        + "from deep_worker_cli.config import create_model\n"
        "m1 = create_model(provider='vertexai', model_name='gemini-2.0-flash')\n"
        "assert type(m1).__name__ == 'ChatGoogleGenerativeAI', type(m1)\n"
        "assert getattr(m1, 'vertexai', False) is True, 'not vertexai ADC'\n"
        "m2 = create_model(provider='vertexai', model_name='claude-sonnet-4-5')\n"
        "assert type(m2).__name__ == 'ChatAnthropicVertex', type(m2)\n"
        "# §5.2 tool assembly on BOTH vertex models (real local tool)\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
        "assert m1.bind_tools([ping]) is not None, 'vertex gemini bind_tools'\n"
        "assert m2.bind_tools([ping]) is not None, 'vertex claude bind_tools'\n"
        "print('VERTEXAI-WITH-EXTRA-OK')\n"
    )
    res = _run_venv(
        py,
        body,
        cwd=tmp_path,
        env_extra={
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_API_KEY": None,  # must be unset for the ADC path
        },
    )
    assert res.returncode == 0, f"vertexai construction failed:\n{res.stdout}\n{res.stderr}"
    assert "VERTEXAI-WITH-EXTRA-OK" in res.stdout


# ---------------------------------------------------------------------------
# §5.3 full RPyC lifecycle: DeepSeek + a second API provider, both wheels
# ---------------------------------------------------------------------------

_AGENT_SPECS = [
    ("dw_wheel_ds", "deepseek", "deepseek-chat"),
    ("dw_wheel_oa", "openai", "gpt-4o"),
]

_LIFE_CHILD = """
import asyncio, json, os, sys
from pathlib import Path
from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry

{origin_assert}
DATA = Path({data_dir!r})
REG = ProcessRegistry(registry_file=Path({registry!r}), lock_file=Path({lock!r}))
CONFIG = AgentConfig(working_dir=DATA, provider={provider!r}, model={model!r})

def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\\n")
    sys.stdout.flush()

async def main():
    proc = AgentProcess(name={name!r}, config=CONFIG, data_dir=DATA, rpyc_port=0,
                        registry=REG)
    # Observe the PRODUCTION registration so the barrier fires strictly after
    # the real RPyC server is up and the real registry write happened.
    real_register = proc._register
    async def observed_register():
        await real_register()
        emit(event="ready", name=proc.name, port=proc.rpyc_port, pid=os.getpid())
    proc._register = observed_register
    await proc.start()  # blocks until SIGTERM/SIGINT -> production shutdown
    emit(event="stopped", name=proc.name)
    return 0

sys.exit(asyncio.run(main()))
"""

_LIFE_CLIENT = """
import asyncio, json, sys
from pathlib import Path
from deep_worker_cli.basesession.protocol_types import SessionStatus
from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry

{origin_assert}
REG = ProcessRegistry(registry_file=Path({registry!r}), lock_file=Path({lock!r}))
NAMES = {names!r}

async def main():
    for name in NAMES:
        client = AgentClient(name, registry=REG)  # same path dwcli --status uses
        await client.connect()
        st = await client.get_status()
        # §5.3: the agent must be in the exact production READY state, not
        # merely "some non-null status".
        assert st is not None, (name, st)
        assert st.status == SessionStatus.READY, (name, getattr(st, 'status', None))
        await client.disconnect()
        print(f'CLIENT-{{name}}-OK')
    return 0

sys.exit(asyncio.run(main()))
"""


async def _await_event(
    proc: asyncio.subprocess.Process, event: str, name: str, timeout: float = 120.0
) -> dict:
    """Read one JSON event line from the child (an OBSERVED barrier)."""
    assert proc.stdout is not None
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"child {name} emitted no {event!r} within {timeout}s"
            ) from None
        if not line:
            raise AssertionError(f"child {name} closed stdout before {event!r}")
        try:
            msg = json.loads(line.decode(errors="replace").strip())
        except ValueError:
            continue  # unrelated startup chatter
        if msg.get("event") == event and msg.get("name") == name:
            return msg


@pytest.mark.parametrize("venv_name", ["venv_monorepo", "venv_sdist"])
async def test_rpyc_lifecycle_two_providers_both_wheels(
    venv_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """§5.3: DeepSeek + OpenAI full lifecycle on BOTH wheels.

    Two REAL agent processes (production ``AgentProcess.start()``: real model
    construction, real tool assembly, real RPyC AsyncioServer on localhost,
    real ProcessRegistry) → dwcli-connect path (``AgentClient``, the exact
    client ``deep-worker-cli --status`` uses) → SessionStatus.READY → clean
    SIGTERM stop → both unregistered. Only the LLM is fake (a fake API key —
    the socket guard forbids any outbound call). No mocks anywhere.
    """
    _venv_dir, py = request.getfixturevalue(venv_name)
    home = tmp_path / "home"
    home.mkdir()
    registry = tmp_path / "registry.json"
    lock = tmp_path / "registry.lock"
    env = _artifact_env(
        {
            "HOME": str(home),
            "DEEPSEEK_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
        }
    )

    procs: list[asyncio.subprocess.Process] = []
    try:
        for name, provider, model in _AGENT_SPECS:
            data_dir = tmp_path / name
            data_dir.mkdir()
            child = _LIFE_CHILD.format(
                name=name,
                provider=provider,
                model=model,
                data_dir=str(data_dir),
                registry=str(registry),
                lock=str(lock),
                origin_assert=_ORIGIN_ASSERT,
            )
            proc = await asyncio.create_subprocess_exec(
                str(py),
                "-c",
                _py_script(child),
                cwd=tmp_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            procs.append(proc)

        ready: dict[str, int] = {}
        for proc, (name, _p, _m) in zip(procs, _AGENT_SPECS):
            sig = await _await_event(proc, "ready", name)
            ready[name] = sig["port"]
        assert len(ready) == 2, ready

        # dwcli-connect path: real RPyC connect + get_status over localhost.
        client_body = _LIFE_CLIENT.format(
            registry=str(registry),
            lock=str(lock),
            names=sorted(ready),
            origin_assert=_ORIGIN_ASSERT,
        )
        cenv = dict(env)
        client = await asyncio.create_subprocess_exec(
            str(py),
            "-c",
            _py_script(client_body),
            cwd=tmp_path,
            env=cenv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        cout, cerr = await asyncio.wait_for(client.communicate(), 90)
        assert client.returncode == 0, f"client failed:\n{cout}\n{cerr}"
        assert b"CLIENT-dw_wheel_ds-OK" in cout and b"CLIENT-dw_wheel_oa-OK" in cout, cout

        # Clean stop: SIGTERM -> production shutdown path -> unregister.
        for proc in procs:
            proc.terminate()
        for proc in procs:
            rc = await asyncio.wait_for(proc.wait(), 60)
            assert rc == 0, f"child exit code {rc} (expected clean 0)"
        assert json.loads(registry.read_text()) == {}, registry.read_text()
    finally:
        for proc in procs:
            if proc.returncode is None:
                proc.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), 10)


# ---------------------------------------------------------------------------
# §5.5 positive custom providers (codex/grok fake CLIs, native-tools assembly,
# missing-CLI documented error)
# ---------------------------------------------------------------------------


def test_custom_providers_construction_and_missing_cli(
    venv_monorepo: tuple[Path, Path], tmp_path: Path
) -> None:
    """§5.5: codex/grok with deterministic fake executables — REAL
    construction + native-tools assembly; missing codex CLI → the documented
    FileNotFoundError."""
    _venv_dir, py = venv_monorepo

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for cli in ("codex", "grok"):
        exe = fake_bin / cli
        exe.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
        exe.chmod(0o755)

    body = (
        "from deep_worker_cli.config import create_model\n"
        "c = create_model(provider='codex', model_name=None)\n"
        "assert type(c).__name__ == 'CodexChatModel', type(c)\n"
        "g = create_model(provider='grok', model_name=None)\n"
        "assert type(g).__name__ == 'GrokChatModel', type(g)\n"
        "from deep_worker.providers.codex_provider import native_tools as nt\n"
        "assert c.bind_tools([nt.command_execution, nt.apply_patch]) is not None\n"
        "print('CUSTOM-OK')\n"
    )
    res = _run_venv(
        py,
        body,
        cwd=tmp_path,
        env_extra={"PATH": f"{fake_bin}:/usr/bin:/bin"},
    )
    assert res.returncode == 0, f"custom provider construction failed:\n{res.stdout}\n{res.stderr}"
    assert "CUSTOM-OK" in res.stdout

    # Missing CLI -> the documented error (codex resolves its CLI at
    # construction; the message is the operator-facing contract).
    res2 = _run_venv(
        py,
        "from deep_worker_cli.config import create_model\n"
        "try:\n"
        "    create_model(provider='codex', model_name=None)\n"
        "    raise SystemExit('UNEXPECTED-SUCCESS')\n"
        "except FileNotFoundError as e:\n"
        "    assert 'codex CLI not found on PATH' in str(e), e\n"
        "print('MISSING-CLI-OK')\n",
        cwd=tmp_path,
        env_extra={"PATH": "/usr/bin:/bin"},
    )
    assert res2.returncode == 0, f"missing-CLI check failed:\n{res2.stdout}\n{res2.stderr}"
    assert "MISSING-CLI-OK" in res2.stdout


# ---------------------------------------------------------------------------
# §5.5 semantic-search extra: install + pip check + real import + min-op
# ---------------------------------------------------------------------------


def test_semantic_search_extra_min_operation(
    venv_extras: tuple[Path, Path], tmp_path: Path
) -> None:
    """§5.5 semantic-search: extra installed (fixture does ``pip check``), the
    lazy root does NOT pull llama-index, and a REAL min-operation (scan + hash
    + language detection on a tiny repo) runs network-free under the guard."""
    _venv_dir, py = venv_extras
    body = (
        "import pathlib, sys, tempfile\n"
        "import deep_worker.providers.semantic_search as ss\n"
        "assert 'llama_index' not in sys.modules, 'eager llama_index'\n"
        "from deep_worker.providers.semantic_search.scanner import (\n"
        "    compute_file_hash, get_language, is_code_file, scan_files,\n"
        ")\n"
        "d = pathlib.Path(tempfile.mkdtemp())\n"
        "(d / 'a.py').write_text('x = 1')\n"
        "(d / 'note.md').write_text('# hi')\n"
        "files = sorted(scan_files(d))\n"
        "assert any(f.name == 'a.py' for f in files), files\n"
        "assert get_language(pathlib.Path('a.py')) == 'python'\n"
        "assert is_code_file(pathlib.Path('a.py'))\n"
        "assert not is_code_file(pathlib.Path('note.md'))\n"
        "assert len(compute_file_hash(d / 'a.py')) == 32\n"
        "# heavy submodule REAL import — proves the extra's deps resolve\n"
        "from deep_worker.providers.semantic_search import index_repository\n"
        "assert 'llama_index' in sys.modules\n"
        "print('SEMANTIC-MINOP-OK')\n"
    )
    res = _run_venv(py, body, cwd=tmp_path)
    assert res.returncode == 0, f"semantic min-op failed:\n{res.stdout}\n{res.stderr}"
    assert "SEMANTIC-MINOP-OK" in res.stdout


# ---------------------------------------------------------------------------
# §5.5 A2A extra: core-disabled start + [a2a] server/client smoke (localhost)
# ---------------------------------------------------------------------------


def test_a2a_extra_server_client_smoke(
    venv_extras: tuple[Path, Path], tmp_path: Path
) -> None:
    """§5.5 A2A: with the [a2a] extra the REAL uvicorn server starts on
    localhost, the REAL a2a-sdk client resolves the AgentCard over HTTP and
    sends a message; only the agent turn engine is a stand-in (LLM fake
    allowed). The socket guard permits only this loopback traffic."""
    _venv_dir, py = venv_extras
    body = (
        "import asyncio\n"
        "async def main():\n"
        "    import httpx\n"
        "    from a2a.client import A2ACardResolver\n"
        "    from deep_worker.providers.deepworker_a2a.client import A2AOutboundClient\n"
        "    from deep_worker.providers.deepworker_a2a.config import A2AConfig\n"
        "    from deep_worker.providers.deepworker_a2a.server import start_a2a_server\n"
        "    class FakeAgent:\n"
        "        name = 'wheela2a'\n"
        "        async def _handle_invoke(self, command, connection=None,\n"
        "                                 notify_callback=None, approval_callback=None,\n"
        "                                 thread_id_override=None):\n"
        "            return {'success': True,\n"
        "                    'response': {'role': 'assistant',\n"
        "                                 'content': 'echo: ' + command.get('message', '')}}\n"
        "        def cancel_current_worker_task(self):\n"
        "            return False\n"
        "    cfg = A2AConfig(enabled=True, host='127.0.0.1', port=0)\n"
        "    handle = await start_a2a_server(FakeAgent(), cfg,\n"
        "                                     agent_name='wheela2a', version='0.0.19')\n"
        "    try:\n"
        "        for _ in range(250):\n"
        "            if getattr(handle.server, 'started', False):\n"
        "                break\n"
        "            await asyncio.sleep(0.02)\n"
        "        async with httpx.AsyncClient(timeout=10) as hc:\n"
        "            card = await A2ACardResolver(hc, handle.url).get_agent_card()\n"
        "        assert card.name == 'wheela2a', card\n"
        "        resp = await A2AOutboundClient().send(handle.url, 'hello a2a')\n"
        "        assert resp.get('success') is True, resp\n"
        "        assert 'echo: hello a2a' in resp.get('answer', ''), resp\n"
        "        print('A2A-SMOKE-OK')\n"
        "    finally:\n"
        "        await handle.shutdown()\n"
        "asyncio.run(main())\n"
    )
    res = _run_venv(py, body, cwd=tmp_path, timeout=120)
    assert res.returncode == 0, f"A2A smoke failed:\n{res.stdout}\n{res.stderr}"
    assert "A2A-SMOKE-OK" in res.stdout


# ---------------------------------------------------------------------------
# §5.4/§5.5 tri-state web-search + TaskManager / non-interactive wiring
# ---------------------------------------------------------------------------


def test_web_search_tri_state_and_task_manager_wiring(
    venv_monorepo: tuple[Path, Path], tmp_path: Path
) -> None:
    """§5.4 tri-state web-search in the wheel venv:

    - ``auto`` → tool skipped with a one-time warning, agent keeps working
      (``claude_agent_sdk`` never loaded);
    - ``enabled`` → ``CapabilityError(capability='web_search')`` with remedy;
    - ``disabled`` → NO probe at all (no import attempt, no warning).

    §5.5: the REAL production assembly entry points — TaskManager
    (``assemble_manager_web_tools``, the exact function ``_initialize_manager``
    calls at ``task_manager.py``) and non-interactive mode
    (``assemble_non_interactive_web_tools``, the exact function
    ``run_non_interactive`` calls at ``non_interactive.py``) — are executed
    with the tool unavailable (skipped).
    """
    _venv_dir, py = venv_monorepo
    body = (
        "import sys\n"
        "from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig\n"
        "from deep_worker_cli.config import WebSearchPolicy\n"
        "from deep_worker_cli.exceptions import CapabilityError\n"
        "from deep_worker_cli.tools import resolve_web_search_tool\n"
        "# auto: SDK absent -> tool skipped, agent keeps working, no SDK import\n"
        "r = resolve_web_search_tool('auto')\n"
        "assert r.tool is None and r.reason == 'sdk_missing', r\n"
        "assert 'claude_agent_sdk' not in sys.modules\n"
        "# enabled: CapabilityError with capability web_search\n"
        "try:\n"
        "    resolve_web_search_tool('enabled')\n"
        "    raise SystemExit('UNEXPECTED')\n"
        "except CapabilityError as e:\n"
        "    assert e.capability == 'web_search', e.capability\n"
        "# disabled: NO availability probe at all\n"
        "r3 = resolve_web_search_tool('disabled')\n"
        "assert r3.tool is None and r3.reason == 'disabled', r3\n"
        "assert 'claude_agent_sdk' not in sys.modules\n"
        "# §5.5 UNAVAILABLE state — real TaskManager + non-interactive assembly\n"
        "# entry points, real AgentConfig (not a dummy):\n"
        "from deep_worker_cli.non_interactive import assemble_non_interactive_web_tools\n"
        "from deep_worker_cli.task_manager.task_manager import assemble_manager_web_tools\n"
        "cfg = AgentConfig(working_dir='.', provider='deepseek',\n"
        "                  model='deepseek-chat', api_key='test-key')\n"
        "tm_tools = assemble_manager_web_tools(cfg)\n"
        "tm_names = [getattr(t, 'name', getattr(t, '__name__', type(t).__name__)) for t in tm_tools]\n"
        "assert tm_names == ['fetch_url'], tm_names\n"
        "ni_tools = assemble_non_interactive_web_tools()\n"
        "ni_names = [getattr(t, 'name', getattr(t, '__name__', type(t).__name__)) for t in ni_tools]\n"
        "assert ni_names == ['http_request', 'fetch_url'], ni_names\n"
        "print('WEBSRCH-TRI-OK')\n"
    )
    res = _run_venv(py, body, cwd=tmp_path)
    assert res.returncode == 0, f"web-search tri-state failed:\n{res.stdout}\n{res.stderr}"
    assert "WEBSRCH-TRI-OK" in res.stdout
