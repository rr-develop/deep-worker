"""TDD for grok provider on-disk session persistence.

Mirrors ``codex_provider/session_state.py`` but persists
``(grok_session_id, sent_count)`` to
``~/.deep_worker/<agent>/grok_session_state.json`` — deliberately a
distinct filename from codex/claude-code so providers never collide in a
shared agent data dir.

POLICIES.md §4 / DESIGN §6.1, a related issue: the data dir is created 0o700 and
the state file 0o600. NO 777/666. The write is atomic (temp + chmod +
os.replace). All tests are hermetic: ``Path.home()`` is monkeypatched to
a tmp dir; no real grok binary.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider.session_state import (  # noqa: E402
    load_session_state,
    save_session_state,
    session_state_path,
)


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestPath:
    def test_path_under_agent_data_dir(self, fake_home) -> None:
        p = session_state_path("agent-1")
        assert p == deep_worker_home(fake_home) / "agent-1" / "grok_session_state.json"

    def test_tm_suffix_shares_agent_dir(self, fake_home) -> None:
        base = session_state_path("agent-1")
        tm = session_state_path("agent-1:tm")
        assert base.parent == tm.parent  # same agent dir

    def test_distinct_filename_from_codex(self, fake_home) -> None:
        assert session_state_path("a").name == "grok_session_state.json"


class TestRoundtrip:
    def test_save_then_load(self, fake_home) -> None:
        save_session_state("agentX", sent_count=7, session_id="sess-abc")
        loaded = load_session_state("agentX")
        assert loaded == {"sent_count": 7, "session_id": "sess-abc"}

    def test_load_missing_returns_none(self, fake_home) -> None:
        assert load_session_state("never_saved") is None

    def test_none_session_id_roundtrips(self, fake_home) -> None:
        save_session_state("agentY", sent_count=0, session_id=None)
        assert load_session_state("agentY") == {"sent_count": 0, "session_id": None}

    def test_multiple_threads_share_file_without_clobber(self, fake_home) -> None:
        save_session_state("agentZ", sent_count=1, session_id="s1")
        save_session_state("agentZ:tm", sent_count=2, session_id="s2")
        assert load_session_state("agentZ") == {"sent_count": 1, "session_id": "s1"}
        assert load_session_state("agentZ:tm") == {"sent_count": 2, "session_id": "s2"}


class TestCorruptionTolerance:
    def test_corrupt_json_degrades_to_none(self, fake_home) -> None:
        path = session_state_path("brokenAgent")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json")
        assert load_session_state("brokenAgent") is None

    def test_save_never_raises_on_unwritable(self, fake_home, monkeypatch) -> None:
        # Simulate write failure — must be swallowed (best-effort).
        import deep_worker.providers.grok_provider.session_state as ss

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        # Should not raise.
        save_session_state("agentErr", sent_count=3, session_id="s")


class TestPermissions:
    def test_dir_is_0700_and_file_is_0600(self, fake_home) -> None:
        save_session_state("permAgent", sent_count=1, session_id="s")
        path = session_state_path("permAgent")
        file_mode = stat.S_IMODE(path.stat().st_mode)
        dir_mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert file_mode == 0o600, f"file mode {oct(file_mode)} != 0o600"
        assert dir_mode == 0o700, f"dir mode {oct(dir_mode)} != 0o700"

    def test_no_world_or_group_bits(self, fake_home) -> None:
        save_session_state("permAgent2", sent_count=1, session_id="s")
        path = session_state_path("permAgent2")
        file_mode = stat.S_IMODE(path.stat().st_mode)
        # No group/other read/write/execute bits whatsoever.
        assert file_mode & 0o077 == 0
