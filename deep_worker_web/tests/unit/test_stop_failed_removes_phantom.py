"""TDD for the phantom-agent Stop bug (incident observed 2026-05-24).

Symptom: an agent that already died (gone from the registry) still shows a card
in the web UI; clicking Stop "does nothing". Root cause: ``request_stop`` for an
unknown agent broadcasts ``agent_stop_failed`` with ``error_type:"LookupError"``
("Agent '<name>' not found"), but the frontend's ``agent_stop_failed`` handler
only shows an error notification and never removes the now-confirmed-dead card —
so Stop appears to do nothing and the phantom lingers.

Fix: expose a pure decision function ``stopFailureMeansAgentGone(data)`` in
main.js that returns true when a stop/restart failure means the agent is not
running / not found, so the handler can remove the card (like
``agent_terminated``). We test the pure function by importing main.js in node
(it adds a ``module.exports`` guard that is inert in the browser).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

MAIN_JS = Path(__file__).resolve().parents[2] / "static" / "js" / "main.js"


def _run_decision(data: dict[str, object]) -> bool:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available on this host")
    # main.js is browser code; stub the DOM globals it touches at import time
    # so node can load it and we can call the pure decision function. We do
    # NOT execute any handler — only ``stopFailureMeansAgentGone``.
    script = textwrap.dedent(f"""
        global.document = {{ addEventListener() {{}}, getElementById() {{ return null; }} }};
        global.window = {{}};
        const m = require({json.dumps(str(MAIN_JS))});
        const fn = m.stopFailureMeansAgentGone;
        if (typeof fn !== 'function') {{
            console.error('NO_FUNCTION');
            process.exit(2);
        }}
        const out = fn({json.dumps(data)});
        process.stdout.write(JSON.stringify(out));
    """)
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, (
        f"node failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return json.loads(result.stdout)


def test_lookup_error_means_agent_gone() -> None:
    """A not-found stop failure → the agent is gone, remove its card."""
    assert _run_decision({
        "type": "agent_stop_failed",
        "agent_name": "agent_1",
        "error": "Agent 'agent_1' not found",
        "error_type": "LookupError",
    }) is True


def test_not_found_text_means_agent_gone() -> None:
    """Match on the message text too (robust to error_type variations)."""
    assert _run_decision({
        "type": "agent_stop_failed",
        "agent_name": "agent_1",
        "error": "Agent 'agent_1' not found",
        "error_type": "ValueError",
    }) is True


def test_generic_stop_failure_keeps_card() -> None:
    """A real failure against a LIVE agent (e.g. SIGKILL race, timeout) must
    NOT remove the card — the agent may still be running."""
    assert _run_decision({
        "type": "agent_stop_failed",
        "agent_name": "agent_2",
        "error": "Connection timeout after 10s",
        "error_type": "IPCConnectionError",
    }) is False


def test_restart_failed_not_found_means_gone() -> None:
    """Restart-failed for a not-found agent also means it's gone."""
    assert _run_decision({
        "type": "agent_restart_failed",
        "agent_name": "ghost",
        "error": "Agent 'ghost' not found",
        "error_type": "LookupError",
    }) is True


def test_missing_fields_safe_default_keeps_card() -> None:
    """Defensive: a malformed event must not nuke a card."""
    assert _run_decision({"type": "agent_stop_failed"}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
