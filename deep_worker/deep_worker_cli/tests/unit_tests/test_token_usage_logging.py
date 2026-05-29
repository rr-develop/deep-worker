"""Tests for token usage and cache hit logging."""

import logging

import pytest

from deep_worker_cli.utils import log_token_usage


class TestLogTokenUsage:
    """Tests for log_token_usage() utility function."""

    def test_logs_full_cache_hit(self, caplog):
        """When prompt_cache_hit_tokens > 0, log cache hit info."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="test_agent",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 3000,
                    "prompt_cache_hit_tokens": 2500,
                    "prompt_cache_miss_tokens": 500,
                    "completion_tokens": 100,
                    "total_tokens": 3100,
                }
            },
        )

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "test_agent" in msg
        assert "prompt=3000" in msg
        assert "cache_hit=2500" in msg
        assert "miss=500" in msg
        assert "83.3%" in msg or "83%" in msg

    def test_logs_cache_hit_from_prompt_tokens_details(self, caplog):
        """Fallback to prompt_tokens_details.cached_tokens."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="deep_agent",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 3000,
                    "prompt_tokens_details": {"cached_tokens": 2500},
                    "completion_tokens": 100,
                    "total_tokens": 3100,
                }
            },
        )

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "deep_agent" in msg
        assert "cache_hit=2500" in msg

    def test_no_cache_fields(self, caplog):
        """When no cache fields exist, still log prompt/completion totals."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="test_agent",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                }
            },
        )

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "prompt=150" in msg
        assert "completion=50" in msg
        # No cache info in fields without cache
        assert "cache_hit" not in msg or "cached_tokens=0" in msg

    def test_none_response_metadata(self, caplog):
        """None metadata is handled gracefully (no crash, no log)."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="test_agent",
            response_metadata=None,
        )

        # Should log nothing
        assert len(caplog.records) == 0

    def test_empty_token_usage(self, caplog):
        """Empty token_usage dict is handled gracefully."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="test_agent",
            response_metadata={"token_usage": {}},
        )

        # Should log nothing (no useful data)
        assert len(caplog.records) == 0

    def test_logs_at_info_level(self, caplog):
        """Message is logged at INFO level."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="test_agent",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_cache_hit_tokens": 800,
                    "prompt_cache_miss_tokens": 200,
                    "total_tokens": 1200,
                }
            },
        )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO

    def test_with_label(self, caplog):
        """Optional label is included in log message."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test_logger")

        log_token_usage(
            logger=logger,
            agent_name="tm_agent",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 500,
                    "prompt_cache_hit_tokens": 400,
                    "prompt_cache_miss_tokens": 100,
                    "completion_tokens": 30,
                }
            },
            label="TM",
        )

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "[TM]" in msg
        assert "tm_agent" in msg
