"""TDD for grok provider on-disk session persistence.

Since 3fe834d1 the grok provider inherits the shared ACP layer, so the state
lives in ``deep_worker/providers/acp/session_state.py`` and persists
``(session_id, sent_count)`` to
``~/.deep_worker/<agent>/acp_session_state.json``. The filename is still
deliberately distinct from codex/claude-code-sdk so providers never collide in
a shared agent data dir; entries inside it are additionally scoped by
``provider`` + session ``fingerprint``.

POLICIES.md §4 / DESIGN §6.1 issue #5: the data dir is created 0o700 and
the state file 0o600. NO 777/666. The write is atomic (temp + chmod +
os.replace). All tests are hermetic: ``Path.home()`` is monkeypatched to
a tmp dir; no real grok binary.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import agent_root, deep_worker_home
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.acp.session_state import (  # noqa: E402
    load_session_state,
    save_session_state,
    session_state_path,
)
from deep_worker.providers.grok_provider.profile import (  # noqa: E402
    build_profile,
)

# The provider id + session fingerprint the grok provider really scopes its
# persisted state with. Taken from the live profile rather than hardcoded so
# an adapter/capability bump cannot leave these tests asserting a stale shape.
PROVIDER = build_profile().provider
FINGERPRINT = build_profile().fingerprint()


def _save(thread_id: str, *, sent_count: int, session_id: str | None) -> None:
    save_session_state(
        thread_id,
        provider=PROVIDER,
        session_id=session_id,
        sent_count=sent_count,
        fingerprint=FINGERPRINT,
    )


def _load(thread_id: str) -> dict | None:
    return load_session_state(
        thread_id, provider=PROVIDER, fingerprint=FINGERPRINT
    )


def _core(loaded: dict | None) -> dict | None:
    """The (sent_count, session_id) pair these tests are about.

    ``load_session_state`` also reports the context gauge
    (``last_response_at`` / ``last_context_used`` / ``model_id``); those are
    covered elsewhere and are not part of this module's oracle.
    """
    if loaded is None:
        return None
    return {"sent_count": loaded["sent_count"], "session_id": loaded["session_id"]}


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestPath:
    def test_path_under_agent_data_dir(self, fake_home) -> None:
        p = session_state_path("agent_1")
        # The agent data dir is ``~/.deep_worker/agents/<name>/`` since
        # DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §3; assert via the canonical
        # accessor so a future relayout cannot drift this test silently.
        expected_dir = agent_root("agent_1", home=fake_home, create=False)
        assert p == expected_dir / "acp_session_state.json"
        # Still inside the deep-worker home for this (fake) user.
        assert deep_worker_home(fake_home) in p.parents

    def test_tm_suffix_shares_agent_dir(self, fake_home) -> None:
        base = session_state_path("agent_1")
        tm = session_state_path("agent_1:tm")
        assert base.parent == tm.parent  # same agent dir

    def test_distinct_filename_from_codex(self, fake_home) -> None:
        assert session_state_path("a").name == "acp_session_state.json"


class TestRoundtrip:
    def test_save_then_load(self, fake_home) -> None:
        _save("agentX", sent_count=7, session_id="sess-abc")
        assert _core(_load("agentX")) == {"sent_count": 7, "session_id": "sess-abc"}

    def test_load_missing_returns_none(self, fake_home) -> None:
        assert _load("never_saved") is None

    def test_none_session_id_roundtrips(self, fake_home) -> None:
        _save("agentY", sent_count=0, session_id=None)
        assert _core(_load("agentY")) == {"sent_count": 0, "session_id": None}

    def test_multiple_threads_share_file_without_clobber(self, fake_home) -> None:
        _save("agentZ", sent_count=1, session_id="s1")
        _save("agentZ:tm", sent_count=2, session_id="s2")
        assert _core(_load("agentZ")) == {"sent_count": 1, "session_id": "s1"}
        assert _core(_load("agentZ:tm")) == {"sent_count": 2, "session_id": "s2"}

    def test_other_provider_entry_is_not_read(self, fake_home) -> None:
        """Provider scoping (DESIGN §3.5): the shared file is keyed by
        thread_id, so an entry written by ANOTHER provider must not be
        resumed as if it were grok's."""
        save_session_state(
            "agentShared",
            provider="codex",
            session_id="codex-sid",
            sent_count=5,
            fingerprint=FINGERPRINT,
        )
        assert _load("agentShared") is None

    def test_fingerprint_mismatch_reads_as_fresh(self, fake_home) -> None:
        """An adapter upgrade must never resume a session it cannot
        faithfully continue (DESIGN §5.2)."""
        _save("agentFp", sent_count=3, session_id="sid-old-adapter")
        stale = dict(FINGERPRINT or {})
        stale["session-capability"] = "load-v999"
        assert (
            load_session_state("agentFp", provider=PROVIDER, fingerprint=stale)
            is None
        )


class TestCorruptionTolerance:
    def test_corrupt_json_degrades_to_none(self, fake_home) -> None:
        path = session_state_path("brokenAgent")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json")
        assert _load("brokenAgent") is None

    def test_save_never_raises_on_unwritable(self, fake_home, monkeypatch) -> None:
        # Simulate write failure — must be swallowed (best-effort).
        import deep_worker.providers.acp.session_state as ss  # noqa: F401

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        # Should not raise.
        _save("agentErr", sent_count=3, session_id="s")


class TestPermissions:
    def test_dir_is_0700_and_file_is_0600(self, fake_home) -> None:
        _save("permAgent", sent_count=1, session_id="s")
        path = session_state_path("permAgent")
        file_mode = stat.S_IMODE(path.stat().st_mode)
        dir_mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert file_mode == 0o600, f"file mode {oct(file_mode)} != 0o600"
        assert dir_mode == 0o700, f"dir mode {oct(dir_mode)} != 0o700"

    def test_no_world_or_group_bits(self, fake_home) -> None:
        _save("permAgent2", sent_count=1, session_id="s")
        path = session_state_path("permAgent2")
        file_mode = stat.S_IMODE(path.stat().st_mode)
        # No group/other read/write/execute bits whatsoever.
        assert file_mode & 0o077 == 0
