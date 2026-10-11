"""Tests for the host MCP server catalog (``external_mcp.catalog``).

The catalog lives in the product runtime home (NOT in the repository) and
is filled by an on-demand scan plus manual entries. The contract pinned
here comes from deep_worker_web/docs/DESIGN_MCP_TAB.md §4:

* identity is the pair ``(origin, id)`` — ``id`` alone is NOT unique;
* a rescan never touches ``pinned`` (user-added) entries;
* entries missing from a scan are marked ``stale``, never deleted;
* an absent/malformed catalog reads as empty, never raises;
* permissions are 0700 / 0600;
* nothing here ever executes a candidate server.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.catalog import (
    CatalogEntry,
    catalog_path,
    classify_weight,
    display_label,
    load_catalog,
    merge_scan,
    save_catalog,
    scan_config_files,
)


def _entry(**kw) -> CatalogEntry:
    base = dict(
        id="srv",
        origin="/o/mcp.json",
        spec={"command": "x"},
        label="srv",
        source="scan:mcp_config",
        weight="unknown",
        pinned=False,
        stale=False,
    )
    base.update(kw)
    return CatalogEntry(**base)


# ── location & permissions ───────────────────────────────────────────


def test_catalog_path_is_under_product_home_not_repo(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    assert path == tmp_path / ".deep_worker" / "mcp" / "catalog.json"


def test_save_creates_0700_dir_and_0600_file(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    save_catalog(path, [_entry()])

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    save_catalog(path, [_entry(id="a"), _entry(id="b", origin="/other/mcp.json")])

    got = load_catalog(path)

    assert {e.id for e in got} == {"a", "b"}


def test_save_leaves_no_tmp_file(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    save_catalog(path, [_entry()])
    assert [p.name for p in path.parent.iterdir()] == ["catalog.json"]


# ── tolerant reads ───────────────────────────────────────────────────


def test_absent_catalog_is_empty_not_error(tmp_path: Path) -> None:
    assert load_catalog(catalog_path(home=tmp_path)) == []


def test_malformed_catalog_is_empty_not_raise(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert load_catalog(path) == []


def test_catalog_with_wrong_shape_is_empty_not_raise(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"servers": "not-a-list"}))
    assert load_catalog(path) == []


def test_entries_missing_required_fields_are_skipped(tmp_path: Path) -> None:
    path = catalog_path(home=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"version": 1, "servers": [{"id": "ok", "origin": "/o", "spec": {}},
                                       {"no_id": True}]}
        )
    )
    assert [e.id for e in load_catalog(path)] == ["ok"]


# ── merge: identity is (origin, id) ──────────────────────────────────


def test_same_id_different_origin_are_two_entries() -> None:
    """The core of §4.2: one name legitimately exists in several files."""
    existing = [_entry(id="serena", origin="/proj-a/mcp.json")]
    scanned = [_entry(id="serena", origin="/proj-b/mcp.json")]

    merged = merge_scan(existing, scanned)

    assert len(merged) == 2
    assert {e.origin for e in merged} == {"/proj-a/mcp.json", "/proj-b/mcp.json"}


def test_same_pair_is_updated_in_place() -> None:
    existing = [_entry(id="s", origin="/o", spec={"command": "old"})]
    scanned = [_entry(id="s", origin="/o", spec={"command": "new"})]

    merged = merge_scan(existing, scanned)

    assert len(merged) == 1
    assert merged[0].spec["command"] == "new"


def test_unknown_pair_is_added() -> None:
    merged = merge_scan([_entry(id="a", origin="/o")], [_entry(id="b", origin="/o")])
    assert {e.id for e in merged} == {"a", "b"}


# ── merge: pinned is sacred, missing becomes stale ───────────────────


def test_missing_entry_is_marked_stale_not_deleted() -> None:
    existing = [_entry(id="gone", origin="/o")]

    merged = merge_scan(existing, [])

    assert len(merged) == 1
    assert merged[0].stale is True


def test_stale_entry_is_revived_when_seen_again() -> None:
    existing = [_entry(id="s", origin="/o", stale=True)]

    merged = merge_scan(existing, [_entry(id="s", origin="/o")])

    assert merged[0].stale is False


def test_pinned_entry_is_never_overwritten_by_scan() -> None:
    existing = [_entry(id="s", origin="/o", spec={"command": "mine"}, pinned=True)]
    scanned = [_entry(id="s", origin="/o", spec={"command": "scanned"})]

    merged = merge_scan(existing, scanned)

    assert merged[0].spec["command"] == "mine"
    assert merged[0].pinned is True


def test_pinned_entry_is_never_marked_stale() -> None:
    existing = [_entry(id="s", origin="manual", pinned=True)]

    merged = merge_scan(existing, [])

    assert merged[0].stale is False


# ── display label disambiguation ─────────────────────────────────────


def test_label_is_bare_id_when_unique() -> None:
    entries = [_entry(id="a", origin="/x"), _entry(id="b", origin="/y")]
    assert display_label(entries[0], entries) == "a"


def test_label_is_disambiguated_when_id_repeats() -> None:
    entries = [
        _entry(id="serena", origin="/proj-a/.deep_worker/mcp.json"),
        _entry(id="serena", origin="/proj-b/.deep_worker/mcp.json"),
    ]
    first = display_label(entries[0], entries)
    second = display_label(entries[1], entries)

    assert first != second
    assert first.startswith("serena")
    assert "proj-a" in first


# ── weight is advisory metadata, never a veto ────────────────────────


def test_weight_flags_known_heavy_binary() -> None:
    assert classify_weight({"command": "serena", "type": "stdio"})[0] == "heavy"


def test_weight_ignores_path_prefix_on_heavy_binary() -> None:
    assert classify_weight({"command": "/opt/bin/rust-analyzer"})[0] == "heavy"


def test_weight_http_is_service_backed() -> None:
    assert classify_weight({"type": "http", "url": "http://x/mcp"})[0] == "service-backed"


def test_weight_plain_stdio_is_unknown_not_heavy() -> None:
    """Unknown must not be reported as heavy — the label is advisory and
    must not smear ordinary servers."""
    assert classify_weight({"command": "my-little-server"})[0] == "unknown"


def test_classify_weight_returns_reason() -> None:
    weight, reason = classify_weight({"command": "gradle"})
    assert weight == "heavy"
    assert reason


# ── scan: read-only, tolerant, portable ──────────────────────────────


def test_scan_finds_entries_in_mcp_json(tmp_path: Path) -> None:
    cfg = tmp_path / "proj" / ".deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"srv": {"command": "x"}}}))

    found = scan_config_files([cfg])

    assert [e.id for e in found] == ["srv"]
    assert found[0].origin == str(cfg)


def test_scan_records_origin_per_file(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}))
    b.write_text(json.dumps({"mcpServers": {"s": {"command": "y"}}}))

    found = scan_config_files([a, b])

    assert {e.origin for e in found} == {str(a), str(b)}


def test_scan_skips_missing_and_malformed_files(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}))
    bad.write_text("{broken")

    found = scan_config_files([bad, tmp_path / "absent.json", good])

    assert [e.id for e in found] == ["s"]


def test_scan_of_nothing_is_empty_not_error() -> None:
    assert scan_config_files([]) == []


def test_scan_marks_entries_as_not_pinned(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}))
    assert scan_config_files([cfg])[0].pinned is False


def test_scan_keeps_env_placeholders_verbatim(tmp_path: Path) -> None:
    """Scanning must not resolve — and therefore never leak — secrets."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {"mcpServers": {"s": {"type": "http", "url": "http://x",
                                  "headers": {"Authorization": "${env:TOK}"}}}}
        )
    )
    entry = scan_config_files([cfg])[0]
    assert entry.spec["headers"]["Authorization"] == "${env:TOK}"


def test_scan_excludes_internal_proxy_servers(tmp_path: Path) -> None:
    """Our own in-agent proxy must never be offered for attaching."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "deepworker": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
                    "deep_worker_acp_proxy_123": {"type": "http", "url": "http://x/mcp"},
                    "real": {"command": "x"},
                }
            }
        )
    )
    assert [e.id for e in scan_config_files([cfg])] == ["real"]
