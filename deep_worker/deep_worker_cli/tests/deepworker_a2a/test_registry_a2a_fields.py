"""Unit tests for optional A2A fields in the process registry (RC-6).

``AgentProcessInfo`` and ``ProcessRegistry`` gain optional ``a2a_port`` /
``a2a_url``. These MUST be backward-compatible: existing entries without the
fields (and callers that don't pass them) keep working exactly as before.
"""

from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


@pytest.fixture
def registry(tmp_path):
    return ProcessRegistry(
        registry_file=tmp_path / "registry.json",
        lock_file=tmp_path / "registry.lock",
    )


def _info(name="agentA", a2a_port=None, a2a_url=None):
    return AgentProcessInfo(
        name=name,
        pid=__import__("os").getpid(),  # alive process so lookup() keeps it
        pid_file=Path("/tmp/agentA.pid"),
        working_dir="/tmp/x",
        started_at=datetime.now(),
        rpyc_port=51000,
        a2a_port=a2a_port,
        a2a_url=a2a_url,
    )


def test_agentprocessinfo_a2a_fields_default_none():
    info = AgentProcessInfo(
        name="a",
        pid=1,
        pid_file=Path("/tmp/a.pid"),
        working_dir="/x",
        started_at=datetime.now(),
        rpyc_port=1,
    )
    assert info.a2a_port is None
    assert info.a2a_url is None


def test_register_and_lookup_roundtrips_a2a_fields(registry):
    registry.register(_info(a2a_port=9123, a2a_url="http://127.0.0.1:9123/"))
    got = registry.lookup("agentA")
    assert got is not None
    assert got.a2a_port == 9123
    assert got.a2a_url == "http://127.0.0.1:9123/"
    assert got.rpyc_port == 51000  # unchanged


def test_register_without_a2a_is_backward_compatible(registry):
    registry.register(_info())  # no a2a_*
    got = registry.lookup("agentA")
    assert got is not None
    assert got.a2a_port is None
    assert got.a2a_url is None


def test_legacy_entry_missing_a2a_keys_lookup_ok(registry):
    # Simulate a registry written by older code (no a2a_* keys at all).
    import json

    registry.registry_file.write_text(
        json.dumps(
            {
                "old": {
                    "pid": __import__("os").getpid(),
                    "pid_file": "/tmp/old.pid",
                    "working_dir": "/x",
                    "started_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "rpyc_port": 50001,
                }
            }
        )
    )
    got = registry.lookup("old")
    assert got is not None
    assert got.a2a_port is None
    assert got.a2a_url is None


def test_list_all_includes_a2a_fields(registry):
    registry.register(_info(a2a_port=9200, a2a_url="http://127.0.0.1:9200/"))
    entries = {e.name: e for e in registry.list_all()}
    assert entries["agentA"].a2a_port == 9200
    assert entries["agentA"].a2a_url == "http://127.0.0.1:9200/"
