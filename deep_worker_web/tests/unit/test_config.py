"""Tests for application configuration and security validation."""

import pytest

from app.config import Settings


class TestSettingsDefaults:
    """Test default settings values."""

    def test_default_host_is_localhost(self) -> None:
        """Test that default host is 127.0.0.1."""
        settings = Settings()
        assert settings.host == "127.0.0.1"

    def test_default_port_is_7887(self) -> None:
        """Test that default port is 7887."""
        settings = Settings()
        assert settings.port == 7887

    def test_default_debug_is_false(self) -> None:
        """Default ``debug`` must be False.

        Rationale: ``debug=True`` flips uvicorn to DEBUG log level,
        and uvicorn at DEBUG emits a record per ASGI / WS keepalive
        event. On a long-lived process with stuck CLOSE-WAIT sockets
        that translates to ~2700 ``logger.formatTime`` calls/sec, which
        fragments the glibc heap until RSS is in the multi-GB range
        even though no Python objects are leaking. See the analysis of the
        earlier debug-log-storm incident.
        """
        settings = Settings()
        assert settings.debug is False, (
            "debug must default to False — DEBUG logging causes the log-storm "
            "described in the earlier debug-log-storm incident analysis"
        )


class TestSecurityValidation:
    """Test security validation for host and port."""

    def test_validate_security_passes_with_correct_values(self) -> None:
        """Test that security validation passes with 127.0.0.1:7887."""
        settings = Settings(host="127.0.0.1", port=7887)
        result = settings.validate_security()
        assert result is True

    def test_validate_security_fails_with_wrong_host(self) -> None:
        """Test that security validation fails with non-localhost host."""
        # Test with 0.0.0.0
        settings = Settings(host="0.0.0.0", port=7887)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

        # Test with public IP
        settings = Settings(host="192.168.1.100", port=7887)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

        # Test with localhost hostname (should also fail - only 127.0.0.1 allowed)
        settings = Settings(host="localhost", port=7887)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

    def test_validate_security_accepts_a_configured_port(self) -> None:
        """A port set via DEEP_WORKER_WEB_PORT is accepted.

        Each checkout binds to one port so several checkouts can run side
        by side; which port that is, is configuration, not a constant.
        """
        settings = Settings(host="127.0.0.1", port=8000)
        assert settings.validate_security() is True

    def test_validate_security_fails_with_privileged_port(self) -> None:
        """Ports below 1024 need root to bind and are refused."""
        settings = Settings(host="127.0.0.1", port=80)
        with pytest.raises(ValueError, match="outside the unprivileged range"):
            settings.validate_security()

    def test_validate_security_error_messages_are_informative(self) -> None:
        """Test that security error messages explain the requirement."""
        settings = Settings(host="0.0.0.0", port=7887)
        with pytest.raises(ValueError) as exc_info:
            settings.validate_security()
        
        error_msg = str(exc_info.value)
        assert "127.0.0.1" in error_msg
        assert "NEVER" in error_msg or "never" in error_msg.lower()
        assert "external" in error_msg.lower()

    def test_settings_validates_on_import(self) -> None:
        """Test that settings are validated when config module is imported.
        
        This test verifies that the security validation happens automatically,
        not just when explicitly called.
        """
        # The settings object is created and validated at module import time
        from app.config import settings
        
        # If we got here without exception, validation passed
        assert settings.host == "127.0.0.1"
        assert settings.port == 7887
