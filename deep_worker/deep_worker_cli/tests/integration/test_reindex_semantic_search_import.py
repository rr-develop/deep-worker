"""Regression guard for the offline ``--reindex-history`` import path.

Bug (regression of 72c55bb): ``deep-worker --reindex-history <stopped-agent>``
crashed with ``No module named 'semantic_search.history_indexer'`` because the
offline CLI call-site imported the nested ``semantic_search`` package WITHOUT
first injecting the ``<tree>/semantic_search`` wrapper dir onto ``sys.path``
(runtime middleware modules do that injection, but they are not loaded on the
offline path).

Why a SUBPROCESS: the original in-process test missed the bug because pytest's
``conftest`` and already-imported middleware had leaked the wrapper dir onto
``sys.path`` / the module cache. Only a fresh interpreter with a launcher-like
``PYTHONPATH`` (that does NOT contain ``semantic_search``) and a neutral cwd
reproduces the clean offline path. Do NOT rewrite this as an in-process import.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# <tree>/deep_worker/deep_worker_cli/tests/integration/<this file>
_TREE = Path(__file__).resolve().parents[4]
_SD = _TREE / "deep_worker" / "deep_worker_cli"
_PD = _TREE / "deep_worker"
# Launcher PYTHONPATH (prod, no --dev): SCRIPT_DIR : PARENT/deepagents : PARENT.
# Deliberately EXCLUDES <tree>/semantic_search — reproducing the offline gap.
_LAUNCHER_PYTHONPATH = f"{_SD}:{_PD / 'deepagents'}:{_PD}"


def _run_offline_import_probe(
    target: str, neutral_cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Import ``_history_indexer_funcs`` from *target* module and call it in a
    fresh interpreter with a launcher-like environment and neutral cwd.

    ``neutral_cwd`` must be an EMPTY directory (a pytest ``tmp_path``): a shared
    dir like ``/tmp`` can contain stray ``*.py`` files that shadow real
    packages and make the probe fail for unrelated reasons.
    """
    code = textwrap.dedent(
        f"""
        import importlib
        mod = importlib.import_module({target!r})
        clear, index = mod._history_indexer_funcs()
        assert callable(clear) and callable(index)
        print("OK")
        """
    )
    env = {
        # Inherit the real interpreter's env (site-packages for heavy deps like
        # mcp / claude_agent_sdk) but pin PYTHONPATH to the launcher's value so
        # semantic_search is NOT trivially importable unless the code injects it.
        **_inherited_env(),
        "PYTHONPATH": _LAUNCHER_PYTHONPATH,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TQDM_DISABLE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(neutral_cwd),  # empty cwd: no local *.py to shadow packages
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _inherited_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.integration
def test_offline_cli_call_site_resolves_semantic_search(tmp_path) -> None:
    """agent_command._history_indexer_funcs() must resolve the nested package
    from a clean offline environment."""
    proc = _run_offline_import_probe("deep_worker_cli.cli.agent_command", tmp_path)
    assert proc.returncode == 0, (
        "offline reindex import probe failed:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout


@pytest.mark.integration
def test_owned_process_call_site_resolves_semantic_search(tmp_path) -> None:
    """process.agent_process._history_indexer_funcs() must also resolve it
    without relying on middleware import order."""
    proc = _run_offline_import_probe(
        "deep_worker_cli.process.agent_process", tmp_path
    )
    assert proc.returncode == 0, (
        "owned-process reindex import probe failed:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
