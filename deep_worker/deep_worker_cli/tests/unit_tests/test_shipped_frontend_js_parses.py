"""Every shipped frontend JS file must be syntactically valid.

Reported 2026-10-19: the web UI listed no agents on a `uv tool install
deep-worker==0.0.22` while the server log showed "Discovered 2 agents from
registry" and "Successfully connected to agent test1". The backend was
healthy; the browser never ran a line of the frontend.

Cause: in the PUBLISHED wheel, `static/js/main.js` did not parse. The
publish-time sanitizer that strips internal incident/agent names from
comments had eaten a newline, leaving

    // Copy last response to clipboardasync function copyLassResponse(event) {

so `async function copyLastResponse` sat inside a `//` comment and its
`await` ended up outside any async function. A browser aborts the whole
file on SyntaxError, so `loadAgents()`, `renderAgentList()` and
`connectWebSocket()` never ran — an empty UI with a working backend and
nothing in the server log to explain it.

The damage was invisible to every existing check: the file was present,
the right size, and listed in the wheel. Only parsing it catches this.

These tests guard the files in THIS tree. They cannot inspect an already
published artifact, but they do fail on any transform that lands broken
JS in the repo, and they document the one command that checks a wheel:

    node --check <site-packages>/deep_worker_cli/web/static/js/main.js
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# deep_worker/deep_worker_cli/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# <repo>/ — holds deep_worker/ and deep_worker_web/
REPO_ROOT = PROJECT_ROOT.parent.parent
STATIC_JS = REPO_ROOT / "deep_worker_web" / "static" / "js"


def _shipped_js() -> list[Path]:
    if not STATIC_JS.is_dir():
        return []
    return sorted(STATIC_JS.glob("*.js"))


def test_frontend_js_directory_is_where_we_think() -> None:
    """Positive control for the path.

    Without this, a wrong path would make the parse tests below collect
    zero files and pass while checking nothing.
    """
    assert STATIC_JS.is_dir(), f"no shipped JS directory at {STATIC_JS}"
    names = {p.name for p in _shipped_js()}
    assert "main.js" in names, f"main.js missing from {STATIC_JS}: {sorted(names)}"


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed on this host"
)
@pytest.mark.parametrize("js_path", _shipped_js(), ids=lambda p: p.name)
def test_shipped_js_parses(js_path: Path) -> None:
    """`node --check` must accept every JS file we ship.

    A parse failure here is not a style issue: the browser executes
    nothing from a file that does not parse, so the UI silently dies
    while the backend looks healthy.
    """
    result = subprocess.run(
        ["node", "--check", str(js_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{js_path.name} does not parse — a browser would abort the whole "
        f"file and the UI would render nothing:\n{result.stderr.strip()}"
    )


def test_no_statement_is_glued_onto_a_line_comment() -> None:
    """Catch the exact corruption shape even without node.

    A `//` comment that swallowed a newline shows up as a declaration
    keyword fused to the end of comment prose. This runs everywhere,
    including hosts and CI images with no node, so the regression cannot
    slip through on a node-less runner.
    """
    offenders: list[str] = []
    # Keywords that must never appear mid-comment glued to a word. The
    # space-free junction is the signature: "clipboardasync function".
    keywords = ("async function", "function ", "const ", "let ", "class ")
    for path in _shipped_js():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.lstrip()
            if not stripped.startswith("//"):
                continue
            for kw in keywords:
                idx = stripped.find(kw)
                if idx <= 0:
                    continue
                # Preceded by a word character means no whitespace split
                # the comment prose from the keyword — i.e. a lost newline.
                if stripped[idx - 1].isalnum():
                    offenders.append(
                        f"{path.name}:{lineno}: {kw!r} glued to comment text: "
                        f"{stripped[:100]!r}"
                    )
                    break
    assert not offenders, (
        "a line comment appears to have swallowed a newline, hiding a "
        "declaration inside it:\n" + "\n".join(offenders)
    )


# --------------------------------------------------------------------------
# Wheel-level guards. The tree being correct is not enough: the published
# 0.0.22 shipped a main.js that no file in the tree matched. Anything that
# transforms assets between the tree and the artifact has to be caught on
# the artifact itself.
# --------------------------------------------------------------------------

#: Asset subtree inside the wheel (see hooks/stage_providers.py WEB_DEST).
_WHEEL_WEB = "deep_worker_cli/web"


def _wheel_members(wheel: Path) -> dict[str, bytes]:
    import zipfile

    with zipfile.ZipFile(wheel) as z:
        return {
            n: z.read(n)
            for n in z.namelist()
            if n.startswith(f"{_WHEEL_WEB}/static/") and not n.endswith("/")
        }


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel once and hand its path to the wheel-level tests."""
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [
            "python3", "-m", "build", "--wheel",
            "--outdir", str(out), str(PROJECT_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build unavailable here: {result.stderr[-400:]!r}")
    wheels = sorted(out.glob("*.whl"))
    assert wheels, f"build reported success but produced no wheel in {out}"
    return wheels[0]


def test_wheel_static_matches_the_tree_byte_for_byte(built_wheel: Path) -> None:
    """Staging must not alter a single byte of the static assets.

    This is the check that would have caught the published 0.0.22: its
    main.js matched no file in any source tree. ``shutil.copy2`` is
    byte-exact today, so any drift means a new transform step crept in.
    """
    members = _wheel_members(built_wheel)
    assert members, f"wheel ships no static assets under {_WHEEL_WEB}/static/"

    tree_static = REPO_ROOT / "deep_worker_web" / "static"
    mismatches: list[str] = []
    for name, blob in sorted(members.items()):
        rel = name[len(f"{_WHEEL_WEB}/static/"):]
        src = tree_static / rel
        if not src.is_file():
            mismatches.append(f"{rel}: in wheel but not in tree")
        elif src.read_bytes() != blob:
            mismatches.append(
                f"{rel}: wheel bytes differ from tree "
                f"({len(blob)} vs {len(src.read_bytes())} bytes)"
            )
    assert not mismatches, "static assets were altered on the way into the wheel:\n" + "\n".join(mismatches)


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed on this host"
)
def test_wheel_js_parses(built_wheel: Path, tmp_path: Path) -> None:
    """Every JS file INSIDE the built wheel must parse.

    Fails on the published 0.0.22 artifact, which is the point: a browser
    runs nothing from a file that does not parse, so the UI renders no
    agents while the backend log looks perfectly healthy.
    """
    members = {
        n: b for n, b in _wheel_members(built_wheel).items() if n.endswith(".js")
    }
    assert members, "wheel ships no JS — the UI cannot work"

    failures: list[str] = []
    for name, blob in sorted(members.items()):
        target = tmp_path / Path(name).name
        target.write_bytes(blob)
        result = subprocess.run(
            ["node", "--check", str(target)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            failures.append(f"{name}:\n{result.stderr.strip()}")
    assert not failures, "the wheel ships JS that does not parse:\n" + "\n\n".join(failures)


def test_wheel_contains_every_asset_the_html_references(built_wheel: Path) -> None:
    """No HTML may reference a static asset the wheel does not ship.

    A missing stylesheet or script is invisible server-side — it surfaces
    only as a 404 in the browser and a page that half-renders.
    """
    import re

    members = _wheel_members(built_wheel)
    available = {n[len(f"{_WHEEL_WEB}/static/"):] for n in members}
    pattern = re.compile(r'(?:src|href)\s*=\s*"([^"]+)"')

    missing: list[str] = []
    for name, blob in sorted(members.items()):
        if not name.endswith(".html"):
            continue
        for ref in pattern.findall(blob.decode("utf-8")):
            # Skip absolute URLs, protocol-relative URLs and fragments.
            if ref.startswith(("http://", "https://", "//", "#", "data:", "mailto:")):
                continue
            # Page-relative API/route references are not static assets.
            if not ref.startswith("/static/"):
                continue
            rel = ref[len("/static/"):].split("?", 1)[0].split("#", 1)[0]
            if rel not in available:
                missing.append(f"{Path(name).name} references /static/{rel}, absent from the wheel")
    assert not missing, "the wheel is missing assets its own pages load:\n" + "\n".join(missing)
