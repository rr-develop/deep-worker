"""Tests for agent_fixture cleanup of ``<agent-data-dir>/<agent_name>/``.

Background
----------
Each running agent persists ``session_metadata.json`` and other
files under ``<agent-data-dir>/<agent_name>/`` (the
``SessionManager.data_dir`` default — see
``AgentProcess.__init__`` line 204:
``self.data_dir = data_dir or (deep_worker_home() / name)``).

The ``agent_fixture`` context manager cleans up the agent **process**,
the registry record, the ``/tmp/deep_worker_<name>.pid`` file, and the
temp ``working_dir``. It used to leave
``<agent-data-dir>/<test_*>/`` on disk forever — hundreds of stale
directories accumulated over time (466 observed at one point).

The user's mandate:

    Tests MUST clean up everything they created on passed runs.
    Failed runs may preserve artefacts for debugging.

This test file pins the new contract.

Why unit tests?
---------------
Spawning a real agent in a unit test is slow and fragile. We instead
test the small helper ``_cleanup_agent_files`` directly: it now also
removes ``<agent-data-dir>/<agent_name>/`` (only when the name starts
with ``test_`` — a safety guard so a buggy fixture can never wipe a
production agent's data directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from deep_worker_cli.storage_paths import deep_worker_home

# ``agent_fixture`` lives in ``tests/integration_ai/`` and uses
# relative imports (``from .models import ...``). To resolve those
# from this unit test we add the deep_worker_cli root to sys.path
# and import via the full ``tests.integration_ai`` package path.
_DW_CLI_ROOT = Path(__file__).resolve().parents[2]
if str(_DW_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DW_CLI_ROOT))

from tests.integration_ai.agent_fixture import (  # type: ignore[import-not-found]  # noqa: E402
    _cleanup_agent_files,
)
from tests.integration_ai.models import (  # type: ignore[import-not-found]  # noqa: E402
    AgentFixtureInfo,
    AgentStatus,
)


def _make_agent_info(name: str) -> AgentFixtureInfo:
    return AgentFixtureInfo(
        agent_name=name,
        working_dir=Path("/tmp"),
        rpyc_port=12345,
        pid=99999,
        status=AgentStatus.RUNNING,
        created_by_fixture=True,
    )


class TestCleanupRemovesDataDir:
    """``_cleanup_agent_files`` must remove
    ``<agent-data-dir>/<agent_name>/`` for test agents."""

    def test_removes_existing_test_agent_data_dir(
        self, tmp_path: Path
    ) -> None:
        # Redirect Path.home() to tmp_path so the test never touches
        # the real <agent-data-dir>.
        from unittest.mock import patch

        fake_home = tmp_path
        data_dir = fake_home / Path("~/.deep_worker").name / "test_cleanup_smoke"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "session_metadata.json").write_text("{}")
        assert data_dir.exists()

        with patch("pathlib.Path.home", return_value=fake_home):
            _cleanup_agent_files(_make_agent_info("test_cleanup_smoke"))

        assert not data_dir.exists(), (
            "After cleanup the agent's <agent-data-dir>/<name>/ "
            "directory must be gone."
        )

    def test_pydantic_blocks_non_test_names_at_construction(self) -> None:
        """First line of defence: ``AgentFixtureInfo.agent_name`` is
        validated against ``^test_[a-z0-9_]+$``. ``_cleanup_agent_files``
        therefore literally cannot be called with a production agent
        name — pydantic raises before we ever reach the helper.

        This guard is what lets ``_cleanup_agent_files`` safely
        ``shutil.rmtree`` the data dir without a separate prefix
        check: by the time the helper runs, the name is guaranteed
        to be a test name.
        """
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            _make_agent_info("agent_a")

    def test_no_op_when_data_dir_missing(self, tmp_path: Path) -> None:
        """If the directory does not exist (e.g. agent never reached
        the SessionManager init phase), cleanup must silently no-op
        rather than raise."""
        from unittest.mock import patch

        fake_home = tmp_path
        # No mkdir — directory does not exist
        with patch("pathlib.Path.home", return_value=fake_home):
            # MUST NOT raise
            _cleanup_agent_files(_make_agent_info("test_never_started"))

    def test_also_removes_pid_file(self, tmp_path: Path) -> None:
        """The legacy behaviour (removing /tmp/deep_worker_<name>.pid)
        must remain intact alongside the new data-dir removal."""
        from unittest.mock import patch

        agent_name = "test_pid_cleanup_xyz"
        pid_file = Path(f"/tmp/deep_worker_{agent_name}.pid")
        pid_file.write_text("99999")
        try:
            fake_home = tmp_path
            (fake_home / Path("~/.deep_worker").name / agent_name).mkdir(
                parents=True, exist_ok=True
            )
            with patch("pathlib.Path.home", return_value=fake_home):
                _cleanup_agent_files(_make_agent_info(agent_name))
            assert not pid_file.exists(), (
                "/tmp/deep_worker_<name>.pid must still be removed"
            )
        finally:
            pid_file.unlink(missing_ok=True)
