"""Tests for application configuration and security validation."""

import ast
import inspect
import textwrap

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
        """``Settings()`` takes its port from the module default.

        NOTE: ``EXPECTED_PORT`` *is* ``_DEFAULT_PORT``, so this asserts
        only that the setting is wired to the module-level default — it
        holds for any value that default happens to have, including a
        wrong one (``_DEFAULT_PORT = 9999`` keeps this green). The port
        numbers themselves are pinned with literals in
        ``TestDefaultPortLiterals``.
        """
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


class TestLoopbackInvariant:
    """The loopback binding is the ONE hard invariant of this config.

    Everything else in ``validate_security`` was a whitelist of absolute
    checkout paths, which cannot survive a pip install: there is no
    checkout to whitelist. That part is gone (see
    ``TestNoCheckoutWhitelistPolicy``). Refusing to bind anywhere but
    loopback is not a path policy and does not weaken — it must hold
    identically from source and from a wheel, and must not be reachable
    through any env var, flag or config key.
    """

    def test_loopback_passes_and_non_loopback_fails(self) -> None:
        """Positive control and negative case in ONE test.

        A test that only asserts the rejection cannot tell "the check
        works" from "the check rejects everything" — including the port
        this server is supposed to use.
        """
        allowed = Settings(host="127.0.0.1")
        assert allowed.validate_security() is True, (
            "127.0.0.1 is the one approved binding and must pass — if this "
            "fails, validate_security rejects everything and the negative "
            "case below proves nothing"
        )

        refused = Settings(host="0.0.0.0")
        with pytest.raises(ValueError) as exc_info:
            refused.validate_security()
        message = str(exc_info.value)
        assert "127.0.0.1" in message, (
            f"the rejection must name the only allowed host; got: {message!r}"
        )
        assert "external" in message.lower(), (
            f"the rejection must say why (external exposure); got: {message!r}"
        )

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.100", "10.0.0.5", "localhost", ""],
    )
    def test_every_non_loopback_host_is_refused(self, host: str) -> None:
        """Including ``localhost``: name resolution is not under our
        control, so only the literal address counts."""
        with pytest.raises(ValueError, match="must be 127.0.0.1 only"):
            Settings(host=host).validate_security()

    def test_loopback_check_is_not_overridable_by_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env var may relax the invariant.

        ``env_prefix`` is ``DEEP_WORKER_WEB_``, so the host itself is
        settable from the environment — that is fine, because the check
        runs on the EFFECTIVE value. What must not exist is a switch that
        turns the check off.
        """
        monkeypatch.setenv("DEEP_WORKER_WEB_HOST", "0.0.0.0")
        for off_switch in (
            "DEEP_WORKER_WEB_ALLOW_EXTERNAL",
            "DEEP_WORKER_WEB_SKIP_SECURITY",
            "DEEP_WORKER_WEB_INSECURE",
            "DEEP_WORKER_WEB_DISABLE_SECURITY_CHECK",
        ):
            monkeypatch.setenv(off_switch, "1")

        settings = Settings()
        assert settings.host == "0.0.0.0", (
            "precondition: the env var must actually reach the setting, "
            "otherwise this test proves nothing about the check"
        )
        with pytest.raises(ValueError, match="must be 127.0.0.1 only"):
            settings.validate_security()

    def test_loopback_check_runs_after_config_is_applied(self) -> None:
        """The check must read the EFFECTIVE host.

        If ``host`` ever becomes configurable through a new mechanism, the
        validation has to happen after that mechanism is applied — a check
        against the class default would pass while the server binds
        elsewhere.
        """
        settings = Settings()
        object.__setattr__(settings, "host", "0.0.0.0")
        with pytest.raises(ValueError, match="must be 127.0.0.1 only"):
            settings.validate_security()


class TestNoCheckoutWhitelistPolicy:
    """``validate_security`` must not police WHERE the code lives.

    The old implementation refused to start unless the active tree was one
    of four hard-coded checkout paths and unless the port equalled the
    one assigned to that tree. Both branches are unshippable: a pip user
    has no checkout, so the first raises unconditionally, and the second
    inverted its own intent — it rejected a FREE port and accepted a port
    already in use (measured: port 7888 occupied → True; free port 39947 →
    ValueError). Port collisions are reported by ``bind()``; this function
    is not a port allocator.
    """

    def test_unrecognised_checkout_is_not_a_failure(self) -> None:
        """Running from outside the known trees must be allowed."""
        import app.config as config_module

        original = config_module._CHECKOUT_ROOT
        try:
            config_module._CHECKOUT_ROOT = None  # as installed from a wheel
            settings = Settings(host="127.0.0.1")
            assert settings.validate_security() is True, (
                "a pip install has no checkout at all; refusing to start "
                "there is exactly the bug that keeps the web UI "
                "source-only"
            )
        finally:
            config_module._CHECKOUT_ROOT = original

    def test_port_is_not_validated_against_the_checkout(self) -> None:
        """Any port passes the policy; collisions surface at bind()."""
        settings = Settings(host="127.0.0.1", port=8000)
        assert settings.validate_security() is True, (
            "port 8000 differs from this checkout's assigned port, but "
            "validate_security must not veto it — it cannot know whether "
            "the port is free, and its predecessor got the answer exactly "
            "backwards"
        )

    def test_no_repos_path_literals_remain_in_the_policy(self) -> None:
        """The policy itself must not name absolute checkout paths.

        Textual, deliberately: the port table may keep its literals (it
        answers "which port does THIS checkout default to"), but the
        validation must not consult a path whitelist.
        """
        source = inspect.getsource(Settings.validate_security)
        tree = ast.parse(textwrap.dedent(source))

        # The docstring is excluded on purpose: it EXPLAINS why the
        # whitelist was removed, and naming the old paths there is the
        # point. Only executable string constants are policed.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]
        offenders = [s for s in literals if "/" + "repos" + "/" in s]
        assert not offenders, (
            f"validate_security still contains checkout path literals "
            f"{offenders}. A pip user's code does not live in a fixed checkout directory, so "
            f"any policy keyed off those paths refuses to start."
        )


class TestDefaultPortLiterals:
    """The default port, asserted with LITERALS.

    ``EXPECTED_PORT`` is imported from the module under test, so
    ``settings.port == EXPECTED_PORT`` is a tautology: it holds for any
    value the module computes, including a wrong one. Proven by mutation —
    setting ``_DEFAULT_PORT = 9999`` left that assertion green. This class
    names the numbers.
    """

    def test_dev_checkout_defaults_to_7888(self) -> None:
        from app.config import _DEV_TREE, _default_port_for

        assert _default_port_for(_DEV_TREE) == 7888

    def test_prod_checkout_defaults_to_7887(self) -> None:
        from app.config import _PROD_TREE, _default_port_for

        assert _default_port_for(_PROD_TREE) == 7887

    def test_unknown_tree_falls_back_to_7887(self) -> None:
        from pathlib import Path

        from app.config import _default_port_for

        assert _default_port_for(Path("/tmp/not-a-checkout")) == 7887

    def test_installed_without_checkout_falls_back_to_7887(self) -> None:
        """``None`` means "installed from a wheel" and takes the same
        branch as an unrecognised tree."""
        from app.config import _default_port_for

        assert _default_port_for(None) == 7887

    def test_this_tree_defaults_to_its_own_literal_port(self) -> None:
        """End-to-end through ``Settings()``, with the number spelled out.

        This file lives in the dev checkout, so the answer is 7888. Run
        from the prod tree the same assertion reads 7887 — the test is
        skipped there rather than made tautological again.
        """
        from app.config import _CHECKOUT_ROOT, _DEV_TREE

        if _CHECKOUT_ROOT != _DEV_TREE:
            pytest.skip(
                f"this assertion is written for the dev checkout "
                f"({_DEV_TREE}); running from {_CHECKOUT_ROOT}"
            )
        assert Settings().port == 7888
