"""Guards for the rpyc_async sys.path hack removal (agent_process.py).

After switching ``rpyc-async`` to the PyPI package (>=1.0.0), the
local source-checkout sys.path injection in
``deep_worker_cli/process/agent_process.py`` must be gone: on a
developer machine the path exists, and the hack would shadow the PyPI
package with sources — making a "clean env" wheel test dishonest.
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


def test_no_rpyc_async_syspath_hack() -> None:
    src = AGENT_PROCESS.read_text(encoding="utf-8")
    # The hack injected a local source checkout into sys.path at module import.
    assert "_rpyc_async_path" not in src, "rpyc_async sys.path hack must be removed"
    assert 'sys.path.insert(0, str(_rpyc_async_path)' not in src
    assert "Path(\"/path/to/rpyc_async\")" not in src, (
        "agent_process.py must not build a local rpyc_async source path"
    )
    # Plain imports from the installed package remain (that is the intent).
    assert re.search(
        r"from rpyc_async\.core\.protocol import DEFAULT_CONFIG", src
    ), "missing rpyc_async.core.protocol import"
    assert re.search(
        r"from rpyc_async\.utils\.async_server import AsyncioServer", src
    ), "missing rpyc_async.utils.async_server import"
