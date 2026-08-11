"""PyPI packaging integration tests — REAL build + clean-env install.

These tests exercise the acceptance criteria in §4 of the
PyPI packaging design document:

- build wheel + sdist from the monorepo checkout (twine check PASSED);
- METADATA of BOTH wheels: no file://, no direct references, no
  ``Requires-Dist: deepagents``, expected rpyc-async constraint;
- R1: sdist is self-contained — ``python -m build --wheel`` from the
  extracted sdist produces a wheel with top-level ``deepagents/`` and
  ``LICENSE``;
- clean-venv runtime smoke for BOTH wheels (rpyc_async API, vendored
  fork leaf, entry points).

They require network (build isolation downloads hatchling; pip resolves
dependencies), so they live in ``tests/integration_tests/`` and are
deselected by default (``-m 'not integration'``).

NOTE: these tests are NOT hermetic w.r.t. the local checkout — they
build from THIS tree on purpose.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

# deep_worker/deep_worker_cli/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REPO_ROOT = PROJECT_ROOT.parent.parent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(1200),
]


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Ensure the in-tree fork/rpyc paths do NOT leak into subprocesses that
    # must resolve from installed wheels. Keep the rest of the env intact.
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout
    )


def _build_dist(outdir: Path, cwd: Path) -> None:
    """python -m build → wheel + sdist into outdir."""
    res = _run([sys.executable, "-m", "build", "--outdir", str(outdir)], cwd=cwd)
    assert res.returncode == 0, f"build failed:\n{res.stdout}\n{res.stderr}"
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected 1 wheel, got {wheels}"
    assert len(sdists) == 1, f"expected 1 sdist, got {sdists}"


def _read_metadata_from_wheel(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        meta = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(meta) == 1, f"expected 1 METADATA, got {meta}"
        return zf.read(meta[0]).decode("utf-8")


def _wheel_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def _assert_metadata_clean(metadata: str) -> None:
    """Global METADATA criterion (§4.2) for any wheel."""
    assert "file://" not in metadata, "file:// found in METADATA"
    for line in metadata.splitlines():
        if line.startswith("Requires-Dist:"):
            assert "@" not in line, f"direct reference in METADATA: {line}"
            assert "deepagents" not in line.lower(), (
                f"Requires-Dist: deepagents found: {line}"
            )
    assert "rpyc-async" in metadata, "missing rpyc-async constraint in METADATA"


def _clean_venv_smoke(wheel: Path, venv_dir: Path) -> None:
    """Install wheel into a fresh venv and exercise the runtime (§4.5)."""
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"

    res = _run(
        [str(pip), "install", str(wheel)], cwd=REPO_ROOT, timeout=900
    )
    assert res.returncode == 0, f"pip install failed:\n{res.stdout}\n{res.stderr}"

    smoke = (
        "from rpyc_async.core.protocol import DEFAULT_CONFIG\n"
        "from rpyc_async.utils.async_server import AsyncioServer\n"
        "import deepagents\n"
        "from deepagents.graph import create_deep_agent\n"
        "from deepagents.middleware.message_timestamps import (\n"
        "    MessageTimestampMiddleware,\n"
        ")\n"
        "import deep_worker_cli\n"
        "print('SMOKE-OK')\n"
    )
    res = _run([str(py), "-c", smoke], cwd=venv_dir, timeout=300)
    assert res.returncode == 0, f"smoke import failed:\n{res.stdout}\n{res.stderr}"
    assert "SMOKE-OK" in res.stdout

    # Entry points.
    for entry in ("deep-worker", "dwcli", "deep-worker-cli"):
        bin_path = venv_dir / "bin" / entry
        assert bin_path.exists(), f"entry point {entry} missing"
        res = _run([str(bin_path), "--help"], cwd=venv_dir, timeout=300)
        assert res.returncode == 0, (
            f"{entry} --help failed:\n{res.stdout}\n{res.stderr}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_and_twine_check_from_monorepo(tmp_path: Path) -> None:
    """§4.1: build wheel+sdist from monorepo; twine check PASSED for both."""
    dist = tmp_path / "dist"
    dist.mkdir()
    _build_dist(dist, cwd=PROJECT_ROOT)

    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))

    twine = shutil.which("twine")
    assert twine, "twine not on PATH (install test group)"
    res = _run([twine, "check", str(wheel), str(sdist)], cwd=PROJECT_ROOT)
    assert res.returncode == 0, f"twine check failed:\n{res.stdout}\n{res.stderr}"


def test_metadata_clean_for_both_wheels(tmp_path: Path) -> None:
    """§4.2: global METADATA criterion for monorepo AND sdist-derived wheels."""
    dist = tmp_path / "dist"
    dist.mkdir()
    _build_dist(dist, cwd=PROJECT_ROOT)

    # Wheel from monorepo.
    wheel = next(dist.glob("*.whl"))
    _assert_metadata_clean(_read_metadata_from_wheel(wheel))

    # Wheel rebuilt from the extracted sdist (R1).
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    sdist = next(dist.glob("*.tar.gz"))
    res = _run(
        ["tar", "-xzf", str(sdist), "-C", str(extracted)], cwd=tmp_path
    )
    assert res.returncode == 0, f"tar extract failed: {res.stderr}"
    sdist_root = extracted / next(
        d for d in extracted.iterdir() if d.is_dir()
    )
    assert (sdist_root / "deepagents").is_dir(), (
        "R1: sdist must contain vendored deepagents/"
    )
    assert (sdist_root / "LICENSE").exists(), "R1: sdist must contain LICENSE"

    dist2 = tmp_path / "dist2"
    dist2.mkdir()
    _build_dist(dist2, cwd=sdist_root)
    wheel2 = next(dist2.glob("*.whl"))
    _assert_metadata_clean(_read_metadata_from_wheel(wheel2))

    # Both wheels must contain vendored code and LICENSE (§4.6).
    for w, label in ((wheel, "monorepo wheel"), (wheel2, "sdist-derived wheel")):
        entries = _wheel_entries(w)
        assert "deepagents/__init__.py" in entries, (
            f"{label}: missing vendored deepagents/"
        )
        assert "LICENSE" in entries, f"{label}: missing LICENSE"


def test_clean_venv_install_smoke(tmp_path: Path) -> None:
    """§4.5: clean venv (no source tree) install + runtime smoke for BOTH wheels."""
    dist = tmp_path / "dist"
    dist.mkdir()
    _build_dist(dist, cwd=PROJECT_ROOT)
    wheel = next(dist.glob("*.whl"))
    _clean_venv_smoke(wheel, tmp_path / "venv1")

    # sdist-derived wheel.
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    sdist = next(dist.glob("*.tar.gz"))
    res = _run(["tar", "-xzf", str(sdist), "-C", str(extracted)], cwd=tmp_path)
    assert res.returncode == 0, f"tar extract failed: {res.stderr}"
    sdist_root = extracted / next(
        d for d in extracted.iterdir() if d.is_dir()
    )
    dist2 = tmp_path / "dist2"
    dist2.mkdir()
    _build_dist(dist2, cwd=sdist_root)
    wheel2 = next(dist2.glob("*.whl"))
    _clean_venv_smoke(wheel2, tmp_path / "venv2")
