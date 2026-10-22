"""Guard: shared, capability-gated functionality must not live in (or be
imported from) ``claude_code_sdk_provider`` by code outside it.

``claude_code_sdk_provider`` is excluded from the public distribution (it is
the monorepo-only, privately-forked CLI integration). A module that is
registered by DUCK-TYPED CAPABILITY rather than by concrete provider class —
``InterAgentDeliveryResolveMiddleware`` is the motivating case — must live in
a package every build ships, or the registration silently no-ops the moment
``claude_code_sdk_provider`` is absent.

Regression this guards against: ``_maybe_add_delivery_resolve_middleware``
imported the middleware from ``claude_code_sdk_provider.delivery_resolve_middleware``
inside a bare ``try: ... except Exception: return``. In a tree without
``claude_code_sdk_provider`` (the published one), the import always failed,
the ``except`` swallowed it, and the middleware was never wired for ANY
model — including ACP, whose ``AcpChatModel`` does implement the duck-typed
``resolve_pending_tool`` and should have gotten it. Two tests in
``tests/acp_provider/test_turn_ending_resolve.py::TestMiddlewareRegistration``
caught this live in the public CI run, not locally, because
``claude_code_sdk_provider`` is always present in this monorepo checkout.

This guard is static (AST, no import) so it reports the same answer whether
or not ``claude_code_sdk_provider`` happens to be present in the environment
running it — mirroring ``test_optional_dependency_skips.py``'s rationale.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

#: Trees that ship in every build and must never reach into
#: ``claude_code_sdk_provider`` except through the already-guarded,
#: capability-checked sites below.
SCANNED_ROOTS = (
    REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli",
    REPO_ROOT / "deep_worker" / "providers" / "acp",
    REPO_ROOT / "deep_worker" / "providers" / "grok_provider",
    REPO_ROOT / "deep_worker" / "providers" / "codex_provider",
    REPO_ROOT / "deep_worker" / "providers" / "claude_code_provider",
)

#: file:lineno sites that legitimately import a genuinely claude-code-sdk-
#: specific symbol (``ClaudeCodeChatModel`` and its ``chat_model``/
#: ``web_search`` submodules), each gated by ``ensure_provider_importable(...)``
#: + a loud ``CapabilityError`` before the import — never a bare swallow.
#: Adding to this list is a deliberate per-site decision, not a blanket
#: exemption. The ``--import-claude`` session-import path used to be here too
#: (``agent_command.py``) until it was moved to ``deep_worker_cli.
#: claude_session_import`` — it has no claude-code-sdk-specific dependency and
#: belongs in always-shipped code, not behind a capability probe.
ALLOWED_SITES = frozenset(
    {
        ("deep_worker_cli/config.py", 2005),
        ("deep_worker_cli/process/agent_process.py", 2419),
        ("deep_worker_cli/tools.py", 154),
    }
)


def _iter_source_files() -> list[Path]:
    seen: set[Path] = set()
    for root in SCANNED_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            seen.add(p.resolve())
    return sorted(seen)


def _module_name(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if "claude_code_sdk_provider" in alias.name:
                return alias.name
        return None
    if isinstance(node, ast.ImportFrom) and node.module:
        if "claude_code_sdk_provider" in node.module:
            return node.module
    return None


def _relative_id(path: Path) -> str:
    for anchor in ("deep_worker_cli/deep_worker_cli", "providers"):
        marker = anchor.split("/")[-1]
        parts = path.parts
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            return "/".join(parts[idx:])
    return str(path)


def _find_offending_imports() -> list[tuple[str, int]]:
    offenders: list[tuple[str, int]] = []
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        rel = _relative_id(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if _module_name(node) is None:
                continue
            offenders.append((rel, node.lineno))
    return offenders


def test_no_unguarded_import_of_claude_code_sdk_provider() -> None:
    offenders = [
        (rel, lineno)
        for rel, lineno in _find_offending_imports()
        if (rel, lineno) not in ALLOWED_SITES
    ]
    assert offenders == [], (
        "code outside claude_code_sdk_provider imports from it at an "
        f"un-reviewed site: {offenders}. Either move the shared "
        "functionality out of claude_code_sdk_provider (preferred — see "
        "deep_worker.providers.acp.delivery_resolve_middleware for the "
        "precedent), or add the exact (file, lineno) to ALLOWED_SITES only "
        "if the import is genuinely claude-code-sdk-specific and is gated "
        "by a loud capability check, never a bare except-swallow."
    )


@pytest.mark.parametrize("rel,lineno", sorted(ALLOWED_SITES))
def test_allowed_sites_still_exist_at_the_recorded_line(rel: str, lineno: int) -> None:
    """An allow-listed exemption must point at a real, still-matching import —
    otherwise a refactor that moves the import could silently widen the
    exemption to a different, unreviewed site at the same line number."""
    full_path = REPO_ROOT / "deep_worker" / "deep_worker_cli" / rel
    assert full_path.is_file(), f"allow-listed file no longer exists: {full_path}"
    tree = ast.parse(full_path.read_text(encoding="utf-8"), filename=str(full_path))
    matches = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _module_name(node) is not None
        and node.lineno == lineno
    ]
    assert matches, (
        f"{rel}:{lineno} is allow-listed but no claude_code_sdk_provider "
        "import was found there — update ALLOWED_SITES to the new line"
    )
