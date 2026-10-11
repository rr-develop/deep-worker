"""The model pin must be validated on RESUME too (round 8 #5).

The post-session gate (``validate_session_new_reply``) ran on
``session/new`` only. A resume after a restart skipped it, so a pin the
adapter had since DROPPED silently fell back to the CLI's default: the
agent kept running on a different model than the operator pinned, with
nothing logged.

The picker rotates BETWEEN processes — which is precisely when a resume
happens — so a resume is the likeliest moment for a pin to be stale.

Contract: same validator, same fail-loud behaviour, both entry points; and
the established fail-OPEN cases (no pin / adapter reported no picker) must
keep working, because a resume that hard-fails on a degenerate reply would
be worse than the bug.
"""
from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker.providers.claude_code_provider.profile import (
    AcpModelError,
    MODEL_ENV_VAR,
    validate_session_new_reply,
)


def _client(reply: dict, env: dict | None) -> AcpClient:
    client = object.__new__(AcpClient)
    client._env = env
    client._session_ready = {}
    client._session_new_validator = validate_session_new_reply

    async def _fake_call(method, params):
        return reply

    client._call = _fake_call
    return client


def _reply_with_picker(*values: str, current: str = "default") -> dict:
    return {
        "result": {
            "configOptions": [
                {
                    "id": "model",
                    "currentValue": current,
                    "options": [{"value": v} for v in values],
                }
            ]
        }
    }


class TestSessionLoadRunsTheGate:
    def test_dropped_pin_fails_loudly_on_resume(self) -> None:
        """THE DEFECT: a stale pin used to sail through session/load."""
        client = _client(
            _reply_with_picker("sonnet", "opus"),
            {MODEL_ENV_VAR: "retired-model"},
        )
        with pytest.raises(AcpModelError) as e:
            asyncio.run(client.session_load("s-1", cwd="/tmp"))
        msg = str(e.value)
        assert "retired-model" in msg
        assert "sonnet" in msg and "opus" in msg, (
            "error must list the LIVE picker values, not a remembered list"
        )

    def test_valid_pin_resumes_normally(self) -> None:
        client = _client(
            _reply_with_picker("sonnet", "opus"), {MODEL_ENV_VAR: "opus"}
        )
        assert asyncio.run(client.session_load("s-1", cwd="/tmp"))

    def test_case_insensitive_like_session_new(self) -> None:
        client = _client(
            _reply_with_picker("Opus"), {MODEL_ENV_VAR: "opus"}
        )
        assert asyncio.run(client.session_load("s-1", cwd="/tmp"))


class TestFailOpenCasesArePreserved:
    """A resume must not become MORE brittle than it was."""

    def test_no_pin_is_fine(self) -> None:
        client = _client(_reply_with_picker("sonnet"), {})
        assert asyncio.run(client.session_load("s-1", cwd="/tmp"))

    def test_reply_without_a_picker_is_fine(self) -> None:
        """Older adapters / degenerate replies: we cannot invent a whitelist,
        and hard-failing every resume would be worse than the bug."""
        client = _client({"result": {}}, {MODEL_ENV_VAR: "anything"})
        assert asyncio.run(client.session_load("s-1", cwd="/tmp"))

    def test_no_validator_configured_is_fine(self) -> None:
        client = _client(_reply_with_picker("sonnet"), {MODEL_ENV_VAR: "x"})
        client._session_new_validator = None
        assert asyncio.run(client.session_load("s-1", cwd="/tmp"))

    def test_readiness_future_is_still_armed(self) -> None:
        """Validation must not short-circuit the rest of the method."""
        client = _client(_reply_with_picker("opus"), {MODEL_ENV_VAR: "opus"})
        asyncio.run(client.session_load("s-9", cwd="/tmp"))
        assert "s-9" in client._session_ready
