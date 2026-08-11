"""deep_worker.providers — vendored provider packages.

Canonical import names for the five in-tree providers:

- ``deep_worker.providers.claude_code_provider``
- ``deep_worker.providers.codex_provider``
- ``deep_worker.providers.grok_provider``
- ``deep_worker.providers.semantic_search``
- ``deep_worker.providers.deepworker_a2a``

This package imports nothing at import time: each provider root is lazy
(PEP 562) so importing the tree never pulls optional/private deps.
"""
