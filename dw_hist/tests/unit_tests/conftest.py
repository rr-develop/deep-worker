"""Ensure dw_hist (and its dependencies) are importable for the unit tests.

The package is consumed via source-imports (no pip install), so we put repo
roots on sys.path here. Mirrors the launcher's PYTHONPATH setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for sub in ("dw_hist", "deep_worker/deep_worker_cli", "semantic_search"):
    p = _REPO_ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
