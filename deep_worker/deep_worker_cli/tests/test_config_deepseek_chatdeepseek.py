"""Regression: the deepseek provider must use ``langchain_deepseek.ChatDeepSeek``,
not a bare ``langchain_openai.ChatOpenAI``.

Incident (observed 2026-05-22): the default deepseek model became
``deepseek-v4-flash`` — a THINKING / reasoning model. It returns
``reasoning_content`` alongside ``content``, and DeepSeek's API requires that
``reasoning_content`` be PASSED BACK on subsequent turns. A plain
``ChatOpenAI`` does not understand the non-standard ``reasoning_content``
field, so it dropped it on the round-trip → the API rejected the next call:

    Error code: 400 - The `reasoning_content` in the thinking mode must be
    passed back to the API.

→ the agent went to ERROR. ``ChatDeepSeek`` natively round-trips
``reasoning_content`` (stores it in ``additional_kwargs`` and re-sends it),
so switching to it fixes the crash.

These tests pin: create_model(provider="deepseek") builds a ChatDeepSeek
configured against api.deepseek.com with the API key, the resolved model, and
the optional ``user`` KV-cache key — and that it does NOT use ChatOpenAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDeepseekUsesChatDeepSeek:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for k in (
            "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "DEEPSEEK_MODEL",
        ):
            monkeypatch.delenv(k, raising=False)

    def _settings(self):
        s = MagicMock()
        s.has_deepseek = True
        s.deepseek_api_key = "sk-test-deepseek"
        s.openai_api_key = None
        s.anthropic_api_key = None
        s.google_api_key = None
        s.google_cloud_project = None
        s.model_name = None
        s.model_provider = None
        return s

    def test_deepseek_builds_chatdeepseek(self) -> None:
        from deep_worker_cli.config import create_model

        with patch("deep_worker_cli.config.settings", self._settings()):
            with patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek") as mock_ds:
                mock_ds.return_value = MagicMock()
                create_model(provider="deepseek", model_name="deepseek-v4-flash")

                mock_ds.assert_called_once()
                kw = mock_ds.call_args[1]
                assert kw["model"] == "deepseek-v4-flash"
                # DeepSeek API endpoint, key, preserved.
                assert kw.get("api_base") == "https://api.deepseek.com"
                assert kw["api_key"] == "sk-test-deepseek"

    def test_deepseek_does_not_use_chatopenai(self) -> None:
        """The whole point: deepseek must NOT go through bare ChatOpenAI
        anymore (that's what dropped reasoning_content)."""
        from deep_worker_cli.config import create_model

        with patch("deep_worker_cli.config.settings", self._settings()):
            with patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek") as mock_ds:
                mock_ds.return_value = MagicMock()
                with patch("langchain_openai.ChatOpenAI") as mock_openai:
                    mock_openai.return_value = MagicMock()
                    create_model(provider="deepseek", model_name="deepseek-v4-flash")

                    mock_ds.assert_called_once()
                    mock_openai.assert_not_called()

    def test_deepseek_passes_user_kv_cache_key(self) -> None:
        from deep_worker_cli.config import create_model

        with patch("deep_worker_cli.config.settings", self._settings()):
            with patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek") as mock_ds:
                mock_ds.return_value = MagicMock()
                create_model(
                    provider="deepseek",
                    model_name="deepseek-v4-flash",
                    user="my_agent:tm",
                )
                kw = mock_ds.call_args[1]
                assert kw["model_kwargs"] == {"user": "my_agent:tm"}

    def test_deepseek_without_user_has_no_model_kwargs(self) -> None:
        from deep_worker_cli.config import create_model

        with patch("deep_worker_cli.config.settings", self._settings()):
            with patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek") as mock_ds:
                mock_ds.return_value = MagicMock()
                create_model(
                    provider="deepseek",
                    model_name="deepseek-v4-flash",
                    user=None,
                )
                kw = mock_ds.call_args[1]
                assert kw.get("model_kwargs") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
