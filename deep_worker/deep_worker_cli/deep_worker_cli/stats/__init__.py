"""Stats package for agent LLM-usage statistics (DESIGN_AGENT_TOKEN_STATS.md).

Dependency-free core: schema, normalization, recorder, reader and aggregator.
deep_worker_cli writes records; deep_worker_cli and deep_worker_web both read
and aggregate them through the functions in :mod:`token_stats`.
"""
