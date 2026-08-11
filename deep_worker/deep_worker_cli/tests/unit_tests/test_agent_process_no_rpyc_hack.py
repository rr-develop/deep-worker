"""Guards for the rpyc_async sys.path hack removal (agent_process.py,
agent_client.py, cli/interactive.py).

After switching ``rpyc-async`` to the PyPI package (>=1.0.0), the
the local-checkout sys.path injection in
``deep_worker_cli/process/agent_process.py`` (and the twin copies in
``deep_worker_cli/client/agent_client.py`` and
``deep_worker_cli/cli/interactive.py``, DESIGN_PROVIDERS_FROM_WHEEL §1
B6) must be gone: on a developer machine the path exists, and the hack
would shadow the PyPI package with sources — making a "clean env" wheel
test dishonest.
"""

from __future__ import annotations

import re
from pathlib import Path

AGENT_PROCESS = (
    Path(__file__).resolve().parent.parent.parent
    / "deep_worker_cli"
    / "process"
    / "agent_process.py"
)

AGENT_CLIENT = (
    Path(__file__).resolve().parent.parent.parent
    / "deep_worker_cli"
    / "client"
    / "agent_client.py"
)

INTERACTIVE = (
    Path(__file__).resolve().parent.parent.parent
    / "deep_worker_cli"
    / "cli"
    / "interactive.py"
)

_HACK_MARKERS = (
    "_rpyc_async_path",
    'sys.path.insert(0, str(_rpyc_async_path)',
    # The removed hack hard-coded an absolute checkout path in a Path(...)
    # literal. Built at runtime so this guard does not itself re-introduce
    # the very string it forbids.
    'Path("/' + 'repos/rpyc_async")',
)


def _assert_no_hack(src: str, label: str) -> None:
    for marker in _HACK_MARKERS:
        assert marker not in src, (
            f"{label} must not contain rpyc_async sys.path hack marker {marker!r}"
        )


def test_no_rpyc_async_syspath_hack_agent_process() -> None:
    src = AGENT_PROCESS.read_text(encoding="utf-8")
    # The hack injected a local rpyc_async checkout into sys.path at import.
    _assert_no_hack(src, "agent_process.py")
    # Plain imports from the installed package remain (that is the intent).
    assert re.search(
        r"from rpyc_async\.core\.protocol import DEFAULT_CONFIG", src
    ), "missing rpyc_async.core.protocol import"
    assert re.search(
        r"from rpyc_async\.utils\.async_server import AsyncioServer", src
    ), "missing rpyc_async.utils.async_server import"


def test_no_rpyc_async_syspath_hack_agent_client() -> None:
    """DESIGN_PROVIDERS_FROM_WHEEL §1 B6 — agent_client.py:23-25."""
    src = AGENT_CLIENT.read_text(encoding="utf-8")
    _assert_no_hack(src, "agent_client.py")
    assert re.search(
        r"import rpyc_async as rpyc", src
    ), "missing rpyc_async import in agent_client.py"


def test_no_rpyc_async_syspath_hack_interactive() -> None:
    """DESIGN_PROVIDERS_FROM_WHEEL §1 B6 — cli/interactive.py:26-28."""
    src = INTERACTIVE.read_text(encoding="utf-8")
    _assert_no_hack(src, "interactive.py")
    assert re.search(
        r"import rpyc_async as rpyc", src
    ), "missing rpyc_async import in interactive.py"
