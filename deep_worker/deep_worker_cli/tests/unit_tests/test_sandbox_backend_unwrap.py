"""Regression: ``_get_sandbox_backend`` must unwrap nested backend wrappers.

Root cause of the 2026-05-28 execute-unavailable incident (a related
internal incident analysis, not included here):

Commit 9b8b570 wrapped the agent's final backend in an OUTER
``PrivateSummariesBackend`` (agent.py:910) but did not update
``_get_sandbox_backend`` (backends.py), which only unwrapped ``CompositeBackend``.
``PrivateSummariesBackend`` subclasses ``BackendProtocol`` (not
``CompositeBackend``), so ``isinstance(..., CompositeBackend)`` is False and the
unwrap fell through to ``return None`` — making ``execute`` /
``wait_for_job`` / ``kill_job`` answer "this agent's backend does not support
command execution" for EVERY local-shell agent restarted after the wrap landed.

The agent's real backend chain is:
    PrivateSummariesBackend(_backend = CompositeBackend(default = CLIShellBackend))

``_get_sandbox_backend`` must reach the ``CLIShellBackend`` through BOTH layers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from deepagents.backends.composite import CompositeBackend

from deep_worker_cli.backends import CLIShellBackend, _get_sandbox_backend
from deep_worker_cli.summaries import PrivateSummariesBackend


def _make_shell() -> CLIShellBackend:
    return CLIShellBackend(root_dir=tempfile.mkdtemp(), inherit_env=True, env={})


class TestGetSandboxBackendUnwrap:
    def test_bare_sandbox(self) -> None:
        shell = _make_shell()
        assert _get_sandbox_backend(shell) is shell

    def test_composite_default_sandbox(self) -> None:
        shell = _make_shell()
        composite = CompositeBackend(default=shell, routes={})
        assert _get_sandbox_backend(composite) is shell

    def test_private_summaries_wraps_composite_sandbox(self) -> None:
        """THE regression: outer PrivateSummariesBackend over a CompositeBackend
        whose default is the CLIShellBackend. Must still resolve the sandbox."""
        shell = _make_shell()
        composite = CompositeBackend(default=shell, routes={})
        wrapped = PrivateSummariesBackend(composite, Path(tempfile.mkdtemp()))

        resolved = _get_sandbox_backend(wrapped)
        assert resolved is shell, (
            "execute/wait_for_job/kill_job resolve the sandbox through this; "
            "if it returns None the agent reports 'backend does not support "
            "command execution' (the 2026-05-28 incident)."
        )

    def test_private_summaries_directly_wrapping_sandbox(self) -> None:
        """Defensive: wrapper directly over the sandbox (no composite layer)."""
        shell = _make_shell()
        wrapped = PrivateSummariesBackend(shell, Path(tempfile.mkdtemp()))
        assert _get_sandbox_backend(wrapped) is shell

    def test_non_sandbox_still_none(self) -> None:
        """A wrapper/composite whose innermost backend is NOT a sandbox must
        still return None (no false positive)."""
        from deepagents.backends.filesystem import FilesystemBackend

        fs = FilesystemBackend(root_dir=tempfile.mkdtemp())
        composite = CompositeBackend(default=fs, routes={})
        wrapped = PrivateSummariesBackend(composite, Path(tempfile.mkdtemp()))
        assert _get_sandbox_backend(wrapped) is None
