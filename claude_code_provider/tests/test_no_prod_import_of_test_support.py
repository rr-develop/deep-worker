"""Lint-style guard: production code must NEVER import from tests/_support/.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §5.2
("Threat boundary and no-prod-hook proof").

The subprocess integration test installs a fake ``ClaudeSDKClient`` transport
via a pytest-owned ``sitecustomize`` shim under ``tests/_support/``. Only the
pytest fixture ever prepends that directory to a child's ``PYTHONPATH`` — no
production process should ever reach it. This test scans every ``.py`` file
under production trees and RED-fails on any import of
``claude_code_provider.tests._support`` or ``tests._support`` or a bare
``sitecustomize``/``_fake_sdk_boot`` import.
"""

from __future__ import annotations

import re
from pathlib import Path

# Production trees to scan. Test directories are deliberately excluded — the
# whole point is that tests may (and do) import _support, but production may
# not.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROD_ROOTS = [
    _REPO_ROOT / "claude_code_provider",
    _REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli",
    _REPO_ROOT / "deep_worker" / "deepagents",
]

# Directories inside those trees that are actually TESTS (allowed to import
# _support). Matched as any path segment.
_TEST_SEGMENTS = frozenset({"tests", "test", "_support"})

# Patterns that constitute a forbidden import of the test-support seam.
_FORBIDDEN_PATTERNS = [
    re.compile(r"^\s*from\s+claude_code_provider\.tests\._support\b", re.M),
    re.compile(r"^\s*import\s+claude_code_provider\.tests\._support\b", re.M),
    re.compile(r"^\s*from\s+tests\._support\b", re.M),
    re.compile(r"^\s*import\s+tests\._support\b", re.M),
    re.compile(r"^\s*from\s+_fake_sdk_boot\b", re.M),
    re.compile(r"^\s*import\s+_fake_sdk_boot\b", re.M),
]


def _is_test_path(p: Path) -> bool:
    return bool(_TEST_SEGMENTS.intersection(p.parts))


def test_no_production_import_of_test_support() -> None:
    offenders: list[tuple[str, str]] = []
    for root in _PROD_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if _is_test_path(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pat in _FORBIDDEN_PATTERNS:
                m = pat.search(text)
                if m:
                    offenders.append((str(path), m.group(0).strip()))
                    break
    assert not offenders, (
        "Production code must NEVER import from tests/_support/. "
        "The test-only fake-SDK seam is enforced no-prod-import per "
        "docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §5.2. "
        f"Offenders:\n" + "\n".join(f"  {p}: {line!r}" for p, line in offenders)
    )
