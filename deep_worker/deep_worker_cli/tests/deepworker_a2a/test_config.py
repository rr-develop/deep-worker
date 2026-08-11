"""Unit tests for ``deepworker_a2a.config.A2AConfig``.

Covers the user-mandated invariants:
- localhost-only by default (host == 127.0.0.1, never 0.0.0.0),
- disabled by default (enabled is False unless explicitly turned on),
- the bind address / port / transport are configurable so a future
  cross-host switch does not require code changes.
"""

import os
from unittest import mock

import pytest

from deep_worker.providers.deepworker_a2a.config import A2AConfig


def test_defaults_are_localhost_and_disabled():
    cfg = A2AConfig()
    assert cfg.enabled is False, "A2A must be disabled by default (RC-7)"
    assert cfg.host == "127.0.0.1", "must bind localhost only by default"
    assert cfg.port == 0, "0 = OS picks a free port by default"
    assert cfg.transport == "jsonrpc"


def test_explicit_values_are_respected():
    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=9999, transport="jsonrpc")
    assert cfg.enabled is True
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999


def test_rejects_zero_value_external_bind_unless_allowed():
    # By default, binding to a non-loopback address is refused: nothing must
    # be exposed outside localhost without an explicit opt-in.
    with pytest.raises(ValueError):
        A2AConfig(host="0.0.0.0")
    with pytest.raises(ValueError):
        A2AConfig(host="")


def test_external_bind_allowed_only_with_explicit_optin():
    # Future cross-host: allowed ONLY when allow_external=True is explicitly set.
    cfg = A2AConfig(host="0.0.0.0", allow_external=True)
    assert cfg.host == "0.0.0.0"
    assert cfg.allow_external is True


def test_from_env_disabled_by_default(monkeypatch):
    for k in list(os.environ):
        if k.startswith("DEEP_WORKER_A2A"):
            monkeypatch.delenv(k, raising=False)
    cfg = A2AConfig.from_env("agentX")
    assert cfg.enabled is False
    assert cfg.host == "127.0.0.1"


def test_from_env_enables_and_reads_port(monkeypatch):
    monkeypatch.setenv("DEEP_WORKER_A2A_ENABLED", "1")
    monkeypatch.setenv("DEEP_WORKER_A2A_PORT", "9123")
    cfg = A2AConfig.from_env("agentX")
    assert cfg.enabled is True
    assert cfg.port == 9123
    assert cfg.host == "127.0.0.1"


def test_from_env_external_requires_optin(monkeypatch):
    monkeypatch.setenv("DEEP_WORKER_A2A_ENABLED", "1")
    monkeypatch.setenv("DEEP_WORKER_A2A_HOST", "0.0.0.0")
    # No DEEP_WORKER_A2A_ALLOW_EXTERNAL -> must refuse.
    with pytest.raises(ValueError):
        A2AConfig.from_env("agentX")


def test_advertised_url_uses_host_and_port():
    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=9999)
    assert cfg.advertised_url() == "http://127.0.0.1:9999/"
