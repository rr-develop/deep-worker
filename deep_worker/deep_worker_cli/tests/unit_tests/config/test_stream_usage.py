"""OpenAI-compatible providers must request usage on streaming responses.

SCOPE — this is a POLICY guard, not the incident fix. The 2026-08-03
``no_usage`` incident was caused by a normalizer gap (the aggregate
``AIMessage.usage_metadata`` shape was not understood); see
``test_token_stats.py::TestNormalizeUsage`` and
``test_stats_callback_usage_metadata.py``. Live probes showed DeepSeek streams
usage even WITHOUT ``stream_usage`` — so these tests must not be read as
proving causation. They only pin the wire option down.

Why the policy exists: ``stream_options={"include_usage": true}`` is sent only
when ``self.stream_usage`` is on, and ``langchain_openai`` auto-enables it ONLY
for the default base URL (``chat_models/base.py``: the check requires
``openai_api_base`` to be ``None``). We always pass an explicit ``api_base``,
so the flag would stay ``None`` and ``_should_stream_usage()`` would be
``False`` — leaving statistics at the mercy of the endpoint. Statistics are
wanted by default (user requirement, 2026-08-03), so ``create_model`` requests
usage explicitly for every OpenAI-compatible provider.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

from deep_worker_cli.config import create_model, settings


class TestStreamUsageEnabled:
    """``create_model`` requests streaming usage for OpenAI-compatible providers."""

    def setup_method(self) -> None:
        settings.model_context_limit = None
        settings.model_name = None
        settings.model_provider = None

    def _isolate_keys(self, **overrides: str | None) -> dict[str, str | None]:
        original = {
            "deepseek_api_key": settings.deepseek_api_key,
            "openai_api_key": settings.openai_api_key,
            "anthropic_api_key": settings.anthropic_api_key,
            "google_api_key": settings.google_api_key,
            "google_cloud_project": settings.google_cloud_project,
        }
        for key in original:
            setattr(settings, key, None)
        for key, value in overrides.items():
            setattr(settings, key, value)
        return original

    def _restore(self, original: dict[str, str | None]) -> None:
        for key, value in original.items():
            setattr(settings, key, value)

    @patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek")
    def test_deepseek_requests_stream_usage(self, mock_chat_class: Mock) -> None:
        """DeepSeek uses a custom api_base, so stream_usage must be explicit."""
        mock_chat_class.return_value = Mock(spec=["invoke"])

        original = self._isolate_keys(deepseek_api_key="test-key")
        try:
            create_model("deepseek-v4-flash")
        finally:
            self._restore(original)

        kwargs: dict[str, Any] = mock_chat_class.call_args.kwargs
        assert kwargs["stream_usage"] is True

    @patch("deep_worker_cli.deepseek_model.ReasoningChatDeepSeek")
    def test_deepseek_stream_usage_set_with_user(self, mock_chat_class: Mock) -> None:
        """The user/KV-cache branch must not drop the stream_usage flag."""
        mock_chat_class.return_value = Mock(spec=["invoke"])

        original = self._isolate_keys(deepseek_api_key="test-key")
        try:
            create_model("deepseek-v4-flash", user="agent:tm")
        finally:
            self._restore(original)

        kwargs: dict[str, Any] = mock_chat_class.call_args.kwargs
        assert kwargs["stream_usage"] is True
        assert kwargs["model_kwargs"] == {"user": "agent:tm"}

    @patch("langchain_openai.ChatOpenAI")
    def test_openai_requests_stream_usage(self, mock_chat_class: Mock) -> None:
        """OpenAI auto-enables the flag only in some configurations — be explicit."""
        mock_chat_class.return_value = Mock(spec=["invoke"])

        original = self._isolate_keys(openai_api_key="test-key")
        try:
            create_model("gpt-4o")
        finally:
            self._restore(original)

        kwargs: dict[str, Any] = mock_chat_class.call_args.kwargs
        assert kwargs["stream_usage"] is True


class TestStreamUsageReachesRequest:
    """The flag actually turns into ``stream_options`` on the wire payload."""

    def test_deepseek_stream_usage_true_yields_include_usage(self) -> None:
        """Real model object (no mocks): stream_usage=True → include_usage."""
        from deep_worker_cli.deepseek_model import ReasoningChatDeepSeek

        model = ReasoningChatDeepSeek(
            model="deepseek-v4-flash",
            api_base="https://api.deepseek.com",
            api_key="test-key",
            stream_usage=True,
        )
        assert model._should_stream_usage() is True

    def test_deepseek_without_flag_would_not_stream_usage(self) -> None:
        """Documents the fragile upstream default (NOT the incident cause).

        Live-verified 2026-08-03: DeepSeek streams ``usage_metadata`` even with
        this returning ``False``, so the flag being off did not cause the
        ``no_usage`` records — it merely left the guarantee to the endpoint.
        """
        from deep_worker_cli.deepseek_model import ReasoningChatDeepSeek

        model = ReasoningChatDeepSeek(
            model="deepseek-v4-flash",
            api_base="https://api.deepseek.com",
            api_key="test-key",
        )
        assert model._should_stream_usage() is False
