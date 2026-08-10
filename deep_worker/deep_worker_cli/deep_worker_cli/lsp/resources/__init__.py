"""Shipped resource files for the LSP subsystem.

Currently only ``deep_worker_serena_mode.yml`` — the Serena mode YAML
that opts ``restart_language_server`` into the advertised tool set
(commit 11 refresh-tool fix).
"""
from __future__ import annotations

from pathlib import Path

SERENA_MODE_YML: Path = (
    Path(__file__).resolve().parent / "deep_worker_serena_mode.yml"
)

__all__ = ["SERENA_MODE_YML"]
