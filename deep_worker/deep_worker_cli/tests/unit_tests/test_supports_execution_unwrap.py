"""Regression: ``_supports_execution`` must see through proxy backend wrappers.

Part of a related internal incident analysis (not included here).

A recent commit wrapped the agent's final backend in an OUTER proxy
(``PrivateSummariesBackend`` in deep_worker_cli — a ``BackendProtocol`` that
delegates to an inner ``_backend``). ``_supports_execution`` only special-cased
``CompositeBackend``, so for the real chain

    Proxy(_backend = CompositeBackend(default = <sandbox>))

both ``isinstance`` checks were False and it returned ``False``. The filesystem
middleware then FILTERED the ``execute`` tool out of ``request.tools`` before
``bind_tools`` (filesystem.py ``awrap_model_call``) — so the model never saw
``execute`` at all (while ``wait_for_job``/``kill_job`` survived). That is the
"execute is missing from my toolset" symptom.

``_supports_execution`` must peel transparent ``_backend`` proxy wrappers first.
We use the REAL ``CLIShellBackend`` as the sandbox (it satisfies
``SandboxBackendProtocol``) and a duck-typed ``_Proxy`` mirroring
``PrivateSummariesBackend``'s ``_backend`` convention.
"""

from __future__ import annotations

import tempfile

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import _supports_execution

from deep_worker_cli.backends import CLIShellBackend


class _Proxy:
    """Transparent proxy mirroring PrivateSummariesBackend's ``_backend`` chain."""

    def __init__(self, backend: object) -> None:
        self._backend = backend


def _sandbox() -> CLIShellBackend:
    return CLIShellBackend(root_dir=tempfile.mkdtemp(), inherit_env=True, env={})


class TestSupportsExecutionUnwrap:
    def test_bare_sandbox(self) -> None:
        assert _supports_execution(_sandbox()) is True

    def test_composite_default_sandbox(self) -> None:
        assert _supports_execution(CompositeBackend(default=_sandbox(), routes={})) is True

    def test_proxy_over_composite_sandbox(self) -> None:
        """THE regression: proxy(composite(sandbox)) must report executable."""
        proxy = _Proxy(CompositeBackend(default=_sandbox(), routes={}))
        assert _supports_execution(proxy) is True, (
            "execute is filtered out of the tool list when this returns False — "
            "the model then has no execute tool (observed live, 2026-05-28)."
        )

    def test_proxy_directly_over_sandbox(self) -> None:
        assert _supports_execution(_Proxy(_sandbox())) is True

    def test_proxy_over_composite_non_sandbox_is_false(self) -> None:
        """No false positive: innermost is not a sandbox -> not executable."""
        proxy = _Proxy(CompositeBackend(default=FilesystemBackend(root_dir=tempfile.mkdtemp()), routes={}))
        assert _supports_execution(proxy) is False
