"""Utility functions for deep-worker CLI."""

import logging
import os
import sys


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

    Handles two cache-field conventions:
    - ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` (top-level)
    - ``prompt_tokens_details.cached_tokens`` (OpenAI-compatible format)

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

    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)

    # Try top-level cache fields first (LangChain-normalised)
    hit = token_usage.get("prompt_cache_hit_tokens")
    miss = token_usage.get("prompt_cache_miss_tokens")

    # Fallback to prompt_tokens_details.cached_tokens (OpenAI format)
    if hit is None and miss is None:
        details = token_usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if cached is not None:
                hit = cached
                miss = prompt_tokens - cached if prompt_tokens else 0

    total = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
    tag = f"[{label}] " if label else ""

    if hit is not None and prompt_tokens:
        pct = 100.0 * hit / prompt_tokens
        logger.info(
            "%s%s prompt=%s cache_hit=%s miss=%s completion=%s total=%s (cache %.1f%%)",
            tag, agent_name, prompt_tokens, hit, miss or 0,
            completion_tokens, total, pct,
        )
    else:
        # No cache info available (different provider or old model)
        logger.info(
            "%s%s prompt=%s completion=%s total=%s (no cache info)",
            tag, agent_name, prompt_tokens, completion_tokens, total,
        )
