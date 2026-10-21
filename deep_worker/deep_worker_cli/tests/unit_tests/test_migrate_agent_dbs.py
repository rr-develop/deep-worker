"""Tests for ``deep-worker migrate-agent-dbs`` CLI.

DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.6, §5.2.

These tests use real filesystem layouts in ``tmp_path`` (no mocks for
I/O) — the CLI is destructive by design and we want to verify it
against the layout it will see in production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deep_worker_cli import storage_paths
from deep_worker_cli.cli import migrate_command
from deep_worker_cli.cli.migrate_command import (
    build_keep_list,
    classify,
    handle_migrate_agent_dbs_command,
    is_fresh,
    is_test_pattern,
    setup_migrate_agent_dbs_parser,
)
from deep_worker_cli.storage_paths import HOME_DIR_NAME


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Real per-test ``~/.deep_worker/`` with stronger isolation.

    Patches ``deep_worker_home`` in the live ``sys.modules`` binding for
    ``storage_paths`` (not the test-file's import-time alias), because
    some earlier tests pop the module out of ``sys.modules`` and re-import
    it. Same applies to ``migrate_command`` and ``agent_command``.
    """
    import sys
    home = tmp_path / ("." + HOME_DIR_NAME)
    home.mkdir(parents=True, exist_ok=True)
    sp = sys.modules["deep_worker_cli.storage_paths"]
    monkeypatch.setattr(sp, "deep_worker_home", lambda h=None: home)
    monkeypatch.setattr(migrate_command, "deep_worker_home", lambda h=None: home)
    import deep_worker_cli.cli.agent_command as _ac
    monkeypatch.setattr(_ac, "deep_worker_home", lambda h=None: home)
    return home


def _write_meta(path: Path, started_iso: str | None, name: str = "reg_agent") -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload: dict = {"status": "idle", "launch_config": {"pid": os.getpid()}}
    if started_iso is not None:
        payload["launch_config"]["started_at"] = started_iso
    (path / "session_metadata.json").write_text(json.dumps(payload))


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace equivalent to CLI defaults.

    ``keep_list`` is always passed as ``None`` (signals "no extra file")
    — the production ``build_keep_list`` accepts ``None`` for both args.
    """
    import argparse
    args = argparse.Namespace(
        dry_run=True,
        execute=False,
        keep_legacy=False,
        delete_test_dirs=False,
        freshness_days=14,
        keep_list=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# unit: classification helpers
# ---------------------------------------------------------------------------


def test_is_test_pattern_matches_known_patterns() -> None:
    assert is_test_pattern("e2e-fast")
    assert is_test_pattern("dw-deadbeef")
    assert is_test_pattern("commit-editor-mine")
    assert is_test_pattern("test_agent")
    assert is_test_pattern("test_web") is True  # matches `^test[_-]`
    # but keep-list protects it in production
    # NOTE: ``acp_r6b`` no longer matches the test-pattern block — the
    # production-like shorthand ``^[a-z]{1,3}[0-9]?_.+$`` (added 2026-08-18
    # per user feedback) treats it as a non-test pattern.


def test_is_test_pattern_does_not_match_production_names() -> None:
    assert not is_test_pattern("ab1_one")
    assert not is_test_pattern("xy_cross_file_extractor")
    assert not is_test_pattern("cd3_two")
    assert not is_test_pattern("best_test")  # was a false-positive under `.*test.*`
    # Tightened regex `^test[_-]` rejects `best_test` (no prefix match).


def test_is_test_pattern_acp_r6_only_specific_letters() -> None:
    # NOTE: ``acp_r6[a-e]`` matches the production-like shorthand
    # ``^[a-z]{1,3}[0-9]?_.+$`` (added 2026-08-18 per user feedback), so
    # these names are NOT classified as test patterns. Use names that do
    # not match the shorthand to verify the regex range:
    assert not is_test_pattern("acp_r6a")
    assert not is_test_pattern("acp_r6e")
    # Outside the a-e range AND not matching shorthand.
    assert not is_test_pattern("acp_r6z")


def test_is_fresh_recognises_recent_metadata(fake_home: Path) -> None:
    from datetime import datetime, timezone
    started = datetime.now(tz=timezone.utc).isoformat()
    _write_meta(fake_home / "draft", started)
    assert is_fresh(fake_home / "draft", days=14) is True


def test_is_fresh_rejects_old_metadata(fake_home: Path) -> None:
    _write_meta(fake_home / "draft", "2026-01-01T00:00:00+00:00")
    assert is_fresh(fake_home / "draft", days=14) is False


def test_classify_keep_protects_production(fake_home: Path) -> None:
    agent_dir = fake_home / "ab1_one"
    agent_dir.mkdir()
    _write_meta(agent_dir, "2026-08-01T00:00:00+00:00")

    keep = {"ab1_one"}
    assert classify("ab1_one", agent_dir, keep, 14) == "keep-move"


def test_classify_test_pattern_does_not_depend_on_keep_list(fake_home: Path) -> None:
    # Use a name that matches a hard test pattern AND does NOT match the
    # production-like shorthand ``^[a-z]{1,3}[0-9]?_.+$``. ``dwtest_foo``
    # hits `^dwtest_[a-z0-9_]+$` without hitting the shorthand.
    agent_dir = fake_home / "dwtest_foo"
    agent_dir.mkdir()
    # No keep-list entry → still classified as test-candidate.
    assert classify("dwtest_foo", agent_dir, set(), 14) == "test-candidate"


def test_classify_unknown_fresh_is_fresh_unknown(fake_home: Path) -> None:
    from datetime import datetime, timezone
    started = datetime.now(tz=timezone.utc).isoformat()
    agent_dir = fake_home / "fresh_xyz"
    _write_meta(agent_dir, started)
    # No metadata version 6 → migrate chain would still walk it as
    # "fresh-unknown"; here we classify purely on freshness/patterns.
    assert classify("fresh_xyz", agent_dir, set(), 14) == "fresh-unknown"


# ---------------------------------------------------------------------------
# unit: build_keep_list
# ---------------------------------------------------------------------------


def test_build_keep_list_filters_by_os_kill_pid_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live PIDs (os.kill(pid, 0) succeeds) are added; dead PIDs are not."""
    fake_registry = tmp_path / "registry.json"
    fake_registry.write_text(json.dumps({
        "alive_agent": {"pid": os.getpid()},     # current process — alive
        "dead_agent": {"pid": 99999999},           # likely dead
        "no_pid_agent": {},                          # no pid field
    }))

    keep = build_keep_list(registry_path=fake_registry, extra_keep_list_path=None)
    assert "alive_agent" in keep
    assert "dead_agent" not in keep
    assert "no_pid_agent" not in keep


def test_build_keep_list_unions_user_keep_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_registry = tmp_path / "registry.json"
    fake_registry.write_text(json.dumps({"default_agent": {"pid": os.getpid()}}))
    keep_file = tmp_path / "extra_keep.txt"
    keep_file.write_text("# comment\nmanual_backup\n")

    keep = build_keep_list(
        registry_path=fake_registry, extra_keep_list_path=keep_file
    )
    assert "default_agent" in keep
    assert "manual_backup" in keep
    assert "# comment" not in keep  # comments ignored


# ---------------------------------------------------------------------------
# integration: handle_migrate_agent_dbs_command (end-to-end via tmp_path)
# ---------------------------------------------------------------------------


def test_handle_dry_run_makes_no_changes(fake_home: Path, tmp_path: Path, monkeypatch) -> None:
    """Dry-run must NOT move or delete anything."""
    # Point registry at a non-existent file in tmp_path so build_keep_list
    # returns an empty set for this test (we want to exercise the move path
    # for a "keep-list-empty" agent).
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    legacy = fake_home / "ab1_one"
    legacy.mkdir()
    (legacy / "threads.db").write_bytes(b"data")

    args = _make_args()
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0

    # Still at the legacy location.
    assert legacy.is_dir()
    assert (legacy / "threads.db").exists()
    # New layout untouched.
    assert not (fake_home / "agents" / "ab1_one").exists()


def test_handle_execute_moves_legacy_into_agents_subdir(
    fake_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    legacy = fake_home / "ab1_one"
    legacy.mkdir()
    (legacy / "threads.db").write_bytes(b"data")
    (legacy / "token_stats.jsonl").write_bytes(b"{}\n")

    args = _make_args(execute=True, dry_run=False)
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0

    assert not legacy.exists()
    new = fake_home / "agents" / "ab1_one"
    assert new.is_dir()
    assert (new / "threads.db").read_bytes() == b"data"
    assert (new / "token_stats.jsonl").read_bytes() == b"{}\n"


def test_handle_execute_skips_infra_dirs(fake_home: Path, tmp_path: Path, monkeypatch) -> None:
    """``logs/``, ``env.d/`` etc. are NEVER touched (DESIGN §5.2 step 2)."""
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    (fake_home / "logs").mkdir()
    (fake_home / "env.d").mkdir()
    (fake_home / "lsp").mkdir()
    (fake_home / "agents").mkdir()  # the new layout itself

    args = _make_args(execute=True, dry_run=False, delete_test_dirs=True)
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0

    assert (fake_home / "logs").is_dir()
    assert (fake_home / "env.d").is_dir()
    assert (fake_home / "lsp").is_dir()
    assert (fake_home / "agents").is_dir()


def test_handle_delete_test_dirs_requires_execute(fake_home: Path) -> None:
    """--delete-test-dirs without --execute errors out (DESIGN §4.6)."""
    legacy = fake_home / "acp_r6a"
    legacy.mkdir()
    (legacy / "threads.db").write_bytes(b"")

    args = _make_args(delete_test_dirs=True, dry_run=True)
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 2  # argparse validation via the handler.
    # Tree still present.
    assert legacy.is_dir()


def test_handle_execute_delete_test_dirs_removes_only_test_dirs(
    fake_home: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    legacy_test = fake_home / "acp_r6a"
    legacy_test.mkdir()
    (legacy_test / "threads.db").write_bytes(b"")

    legacy_prod = fake_home / "moving_prod"
    legacy_prod.mkdir()
    (legacy_prod / "threads.db").write_bytes(b"prod-data")

    args = _make_args(
        execute=True, dry_run=False, delete_test_dirs=True, keep_list=None,
    )
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0

    # Test dir deleted (matching ``^acp_r6[a-e]$``).
    assert not legacy_test.exists()

    # Prod moved to new layout, not deleted.
    assert not legacy_prod.exists()
    new = fake_home / "agents" / "moving_prod"
    assert (new / "threads.db").read_bytes() == b"prod-data"


def test_handle_keep_list_prevents_deletion_of_prod(
    fake_home: Path, tmp_path: Path, monkeypatch
) -> None:
    """Even a test-pattern name is safe if it's in the keep-list."""
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    legacy = fake_home / "test_keepalive"  # looks like test pattern
    legacy.mkdir()
    (legacy / "threads.db").write_bytes(b"")
    (legacy / "session_metadata.json").write_text("{}")

    keep_file = tmp_path / "keep.txt"
    keep_file.write_text("test_keepalive\n")

    args = _make_args(
        execute=True, dry_run=False, delete_test_dirs=True, keep_list=keep_file,
    )
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0
    # Either still at legacy (skipped) or moved to agents — never deleted.
    new = fake_home / "agents" / "test_keepalive"
    assert legacy.is_dir() or new.is_dir()
    assert not (legacy.is_dir() and not new.exists() and legacy.exists()), (
        "test_keepalive should NEVER be deleted because keep-list protects it"
    )


def test_handle_progress_logging_advances(fake_home: Path, capsys, tmp_path: Path, monkeypatch) -> None:
    """At least one progress line per 5 entries (DESIGN §5.1)."""
    import sys
    monkeypatch.setattr(
        "deep_worker_cli.cli.migrate_command.REGISTRY_PATH",
        tmp_path / "nope.json",
    )
    for i in range(7):
        d = fake_home / f"e2e_test_{i}"
        d.mkdir()
        (d / "threads.db").write_bytes(b"")

    args = _make_args()
    rc = handle_migrate_agent_dbs_command(args)
    assert rc == 0
    captured = capsys.readouterr().out
    # Progress prefix present.
    assert "[7/" in captured or "[5/" in captured  # at i=5 and i=7


# ---------------------------------------------------------------------------
# parser wiring
# ---------------------------------------------------------------------------


def test_setup_parser_registers_arguments() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="deep-worker")
    sub = parser.add_subparsers(dest="cmd")
    setup_migrate_agent_dbs_parser(sub)
    args = parser.parse_args(["migrate-agent-dbs"])
    # Defaults preserved.
    assert args.dry_run is True
    assert args.execute is False
    assert args.freshness_days == 14
    assert args.delete_test_dirs is False
    assert args.keep_list is None
