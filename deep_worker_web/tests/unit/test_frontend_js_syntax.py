"""Parse every static JS file with ``node --check`` to catch SyntaxErrors
before they reach a browser.

Rationale: the whole Python test suite (unit + integration) dials the
backend directly via httpx / websockets and never loads ``main.js`` in a
real JS engine. That is how a duplicate ``const pendingRequests``
declaration sat in ``main.js`` for several commits without a single red
test — yet the moment a user opened the page, the entire UI went dark
(``SyntaxError: Identifier 'pendingRequests' has already been
declared`` fails the whole script).

This test runs ``node --check`` on each ``.js`` file under
``static/js/``. It is a cheap static-parse check — no DOM, no network,
no server — and it catches exactly the class of bug that hit us.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC_JS_DIR = Path(__file__).resolve().parents[2] / "static" / "js"


def _static_js_files() -> list[Path]:
    if not STATIC_JS_DIR.is_dir():
        return []
    return sorted(STATIC_JS_DIR.rglob("*.js"))


@pytest.mark.parametrize("js_file", _static_js_files(), ids=lambda p: p.name)
def test_static_js_file_parses_cleanly(js_file: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available on this host — skipping JS parse check")

    result = subprocess.run(
        [node, "--check", str(js_file)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"{js_file.name} failed node --check:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
