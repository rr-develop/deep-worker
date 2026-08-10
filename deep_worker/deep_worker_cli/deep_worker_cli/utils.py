"""Utility functions for deep-worker CLI."""

import logging
import os
import sys

from deep_worker_cli.stats.token_stats import normalize_usage


def is_running_tests() -> bool:
    """Detect if running in test environment.

    Checks multiple indicators with priority:
    1. DEEP_WORKER_FORCE_PRODUCTION=1 - override, force production mode (return False)
    2. DEEP_WORKER_TEST_MODE=1 - explicit test mode (return True)
    3. PYTEST_CURRENT_TEST - pytest sets this automatically
    4. pytest in sys.modules - pytest is loaded

    Returns:
        True if running in test environment, False otherwise
    """
    # Allow explicit override to force production mode in tests
    if os.environ.get("DEEP_WORKER_FORCE_PRODUCTION") == "1":
        return False

    # Explicit test mode
    if os.environ.get("DEEP_WORKER_TEST_MODE") == "1":
        return True

    # Pytest environment variable
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    # Pytest module loaded
    if "pytest" in sys.modules:
        return True

    return False


def log_token_usage(
    logger: logging.Logger,
    agent_name: str,
    response_metadata: dict | None,
    label: str = "",
) -> None:
    """Log token usage and DeepSeek KV-cache hit information.

    Extracts ``prompt_tokens``, ``completion_tokens``, cache hit/miss from
    LangChain's ``AIMessage.response_metadata["token_usage"]`` and logs
    them at INFO level.

    The extraction goes through :func:`deep_worker_cli.stats.token_stats.normalize_usage`
    (shared with the token-stats recorder, DESIGN_AGENT_TOKEN_STATS.md §5.3);
    the log-line format and behaviour are preserved byte-for-byte.

    Args:
        logger: Logger instance to use.
        agent_name: Name of the agent (included in log message).
        response_metadata: From ``AIMessage.response_metadata``, or None.
        label: Optional label e.g. ``"TM"`` for TaskManager invocations.
    """
    if not response_metadata:
        return
    token_usage = response_metadata.get("token_usage")
    if not token_usage or not isinstance(token_usage, dict):
        return

    usage = normalize_usage(None, token_usage)
    if usage is None:
        return

    total = usage.total_tokens
    tag = f"[{label}] " if label else ""

    if usage.cache_hit_tokens is not None and usage.prompt_tokens:
        pct = 100.0 * usage.cache_hit_tokens / usage.prompt_tokens
        logger.info(
            "%s%s prompt=%s cache_hit=%s miss=%s completion=%s total=%s (cache %.1f%%)",
            tag, agent_name, usage.prompt_tokens, usage.cache_hit_tokens,
            usage.cache_miss_tokens or 0,
            usage.completion_tokens, total, pct,
        )
    else:
        # No cache info available (different provider or old model)
        logger.info(
            "%s%s prompt=%s completion=%s total=%s (no cache info)",
            tag, agent_name, usage.prompt_tokens, usage.completion_tokens, total,
        )
