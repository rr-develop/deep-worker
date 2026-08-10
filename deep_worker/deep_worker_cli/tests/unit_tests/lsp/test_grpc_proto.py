"""Checks the .proto contract of the LSP service and its generated stubs.

Commit 3 of DESIGN_LSP_SHARED_SERVICE.md §15.

Invariants (from reviewer comments, Q1–Q3):
- stubs are committed to the repository and imported as ordinary Python;
- RPC names exactly match §4.1 of the design;
- transport-neutral scheme: client/server must not hard-code UDS/TCP.

The test does NOT touch the gRPC channel (the integration suite in commits 4+
does); here we pin the contract as "data" — schema + servicer surface.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path


def _pkg_root() -> Path:
    """Ascend from this test file to the ``deep_worker_cli`` package root.

    Kept as a helper so the suite runs from any worktree copy — the
    reviewer flagged hard-coded absolute paths as a portability smell in
    an earlier review.

    Layout: ``<repo>/deep_worker/deep_worker_cli/tests/unit_tests/lsp/<this>``
    → parents[3] is ``deep_worker_cli``.
    """
    return Path(__file__).resolve().parents[3]


def _proto_path() -> Path:
    return (
        _pkg_root()
        / "deep_worker_cli"
        / "lsp"
        / "service"
        / "proto"
        / "lsp_service.proto"
    )


def _pb2_module_name() -> str:
    return "deep_worker_cli.lsp.service.proto.lsp_service_pb2"


def _pb2_grpc_module_name() -> str:
    return "deep_worker_cli.lsp.service.proto.lsp_service_pb2_grpc"


def test_proto_file_present() -> None:
    assert _proto_path().is_file(), (
        f"missing .proto contract: {_proto_path()}"
    )


def test_proto_declares_expected_rpc_names() -> None:
    """§4.1 of the design: the RPC set is pinned and cannot silently drift."""
    text = _proto_path().read_text(encoding="utf-8")
    expected = {
        "Health",
        "AttachSession",
        "FindSymbol",
        "FindReferencingSymbols",
        "FindDeclaration",
        "FindImplementations",
        "GetSymbolsOverview",
        "GetDiagnosticsForFile",
        "RenameSymbol",
        "ReplaceSymbolBody",
        "InsertBeforeSymbol",
        "InsertAfterSymbol",
        "SafeDeleteSymbol",
        "RefreshProject",
        "ProjectStatus",
    }
    declared = set(re.findall(r"\brpc\s+(\w+)\s*\(", text))
    missing = expected - declared
    extra = declared - expected
    assert not missing, f"RPCs missing from proto: {sorted(missing)}"
    assert not extra, (
        f"RPCs added without design update: {sorted(extra)} "
        "(add to §4.1 first, then to the schema)"
    )


def test_proto_uses_proto3_and_stable_package() -> None:
    text = _proto_path().read_text(encoding="utf-8")
    assert re.search(r'^\s*syntax\s*=\s*"proto3"', text, re.MULTILINE), (
        "proto3 required (proto2 has different default-value semantics)"
    )
    assert re.search(r"^\s*package\s+deep_worker\.lsp\.v1\s*;", text, re.MULTILINE), (
        "package must be deep_worker.lsp.v1 — versioned, stable for future migrations"
    )


def test_proto_attach_session_is_server_stream() -> None:
    """AttachSession must be server-stream (see §4.1 / idle-shutdown §8)."""
    text = _proto_path().read_text(encoding="utf-8")
    m = re.search(r"\brpc\s+AttachSession\s*\(([^)]+)\)\s*returns\s*\(([^)]+)\)", text)
    assert m is not None, "AttachSession rpc line not found"
    returns_clause = m.group(2)
    assert "stream" in returns_clause, (
        "AttachSession must return `stream` (keep-alive + push events)"
    )


def test_generated_stubs_importable() -> None:
    """_pb2 and _pb2_grpc must live in the repo (Q2: commit stubs)."""
    pb2 = importlib.import_module(_pb2_module_name())
    pb2_grpc = importlib.import_module(_pb2_grpc_module_name())
    # Sanity: the service has a servicer/add and a Stub.
    assert hasattr(pb2_grpc, "LspServiceStub")
    assert hasattr(pb2_grpc, "LspServiceServicer")
    assert hasattr(pb2_grpc, "add_LspServiceServicer_to_server")
    # Check that the message classes exist.
    for name in ("HealthRequest", "HealthReply", "AttachRequest", "SessionEvent"):
        assert hasattr(pb2, name), f"pb2 missing message {name}"


def test_servicer_surface_matches_proto() -> None:
    """The generated servicer exposes exactly the same RPCs as the proto."""
    pb2_grpc = importlib.import_module(_pb2_grpc_module_name())
    servicer_cls = pb2_grpc.LspServiceServicer
    text = _proto_path().read_text(encoding="utf-8")
    declared = set(re.findall(r"\brpc\s+(\w+)\s*\(", text))
    for name in declared:
        assert hasattr(servicer_cls, name), (
            f"generated servicer missing method {name!r} — regenerate stubs "
            f"(python -m grpc_tools.protoc …)"
        )


def test_stubs_are_committed_not_gitignored() -> None:
    """Q2: stubs go into git; .gitignore must not hide them."""
    import subprocess

    for suffix in ("_pb2.py", "_pb2_grpc.py"):
        path = _proto_path().parent / f"lsp_service{suffix}"
        assert path.is_file(), f"missing generated stub: {path}"
        # git check-ignore returns exit=1 when the file is NOT ignored —
        # that is exactly what we want.
        r = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=str(_pkg_root().parents[1]),  # → repo root
            check=False,
        )
        assert r.returncode == 1, f"stub {path} is git-ignored (must be committed)"
