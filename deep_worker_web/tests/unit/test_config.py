"""Tests for application configuration and security validation."""

import pytest

from app.config import Settings
# The port is assigned PER CHECKOUT: 7887 for the prod tree
# (the production tree), 7888 for the dev tree
# (the developer tree). Tests must derive the expected port from
# the tree they run in — hard-coding 7887 made the whole suite fail
# when run from the dev worktree (where the correct port is 7888).
from app.config import _DEFAULT_PORT as EXPECTED_PORT  # type: ignore[attr-defined]


class TestSettingsDefaults:
    """Test default settings values."""

    def test_default_host_is_localhost(self) -> None:
        """Test that default host is 127.0.0.1."""
        settings = Settings()
        assert settings.host == "127.0.0.1"

    def test_default_port_matches_checkout(self) -> None:
        """Default port equals the port assigned to the active checkout
        (7887 for the prod tree, 7888 for the dev tree)."""
        settings = Settings()
        assert settings.port == EXPECTED_PORT

    def test_default_debug_is_false(self) -> None:
        """Default ``debug`` must be False.

        Rationale: ``debug=True`` flips uvicorn to DEBUG log level,
        and uvicorn at DEBUG emits a record per ASGI / WS keepalive
        event. On a long-lived process with stuck CLOSE-WAIT sockets
        that translates to ~2700 ``logger.formatTime`` calls/sec, which
        fragments the glibc heap until RSS is in the multi-GB range
        even though no Python objects are leaking. See
        ``a related internal incident analysis (not included here)``.
        """
        settings = Settings()
        assert settings.debug is False, (
            "debug must default to False — DEBUG logging causes the log-storm "
            "described in a related internal incident analysis (not included here)"
        )


class TestSecurityValidation:
    """Test security validation for host and port."""

    def test_validate_security_passes_with_correct_values(self) -> None:
        """Validation passes with 127.0.0.1 and the checkout's port."""
        settings = Settings(host="127.0.0.1", port=EXPECTED_PORT)
        result = settings.validate_security()
        assert result is True

    def test_validate_security_fails_with_wrong_host(self) -> None:
        """Test that security validation fails with non-localhost host."""
        # Test with 0.0.0.0
        settings = Settings(host="0.0.0.0", port=EXPECTED_PORT)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

        # Test with public IP
        settings = Settings(host="192.168.1.100", port=EXPECTED_PORT)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

        # Test with localhost hostname (should also fail - only 127.0.0.1 allowed)
        settings = Settings(host="localhost", port=EXPECTED_PORT)
        with pytest.raises(ValueError, match="SECURITY ERROR: Host must be 127.0.0.1 only"):
            settings.validate_security()

    def test_validate_security_fails_with_wrong_port(self) -> None:
        """A port that does not match the tree's assigned port is rejected.

        The binding is per-checkout: the production tree → 7887,
        the developer tree → 7888. ``validate_security`` raises a
        "must bind to port <expected>" error whenever ``settings.port``
        diverges from the port assigned to the active checkout. (This
        test runs from one of the approved trees, so ``expected`` is
        that tree's port; we only assert the divergence is rejected.)
        """
        settings = Settings(host="127.0.0.1", port=8000)
        with pytest.raises(ValueError, match="must bind to port"):
            settings.validate_security()

        settings = Settings(host="127.0.0.1", port=80)
        with pytest.raises(ValueError, match="must bind to port"):
            settings.validate_security()

    def test_validate_security_error_messages_are_informative(self) -> None:
        """Test that security error messages explain the requirement."""
        settings = Settings(host="0.0.0.0", port=EXPECTED_PORT)
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
        assert settings.port == EXPECTED_PORT
