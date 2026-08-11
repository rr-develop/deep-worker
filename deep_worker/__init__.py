"""deep_worker — root package for the deep-worker project.

This package is the canonical import parent for the vendored providers:

    deep_worker.providers.<pkg>

It intentionally imports nothing heavy at package import time. The
providers tree is resolved lazily (see ``deep_worker_cli/provider_paths.py``).
"""
