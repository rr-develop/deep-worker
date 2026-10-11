"""Unit tests MUST NOT depend on the argv pytest itself was launched with.

The defect these pin down
=========================
``_explicit_cli_flags()`` defaults to reading the REAL ``sys.argv``. Under
pytest that is pytest's own command line, so any test calling
``merge_restart_config(saved, args)`` without an explicit ``explicit=``
silently inherits whatever flags the developer happened to type.

``argparse`` matches long options by unique PREFIX, and the detector
declares ``-pp/--preferred-provider``. Therefore running

    pytest tests/unit_tests -p no:randomly

makes ``-p`` match ``-pp``, ``_explicit_cli_flags()`` returns
``{"preferred_provider"}``, and ~32 restart/failover tests fail — while the
very same tests pass under a bare ``pytest``. The product code is fine; the
suite was simply not isolated from its own invocation.

That is the worst kind of red: it appears and disappears depending on how
the suite is launched, and it trains people to distrust real failures. The
autouse fixture in ``tests/conftest.py`` neutralises ``sys.argv`` for every
test; these tests fail without it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    merge_restart_config,
)

CLI_ROOT = Path(__file__).resolve().parents[3]


def test_explicit_cli_flags_is_empty_under_pytest() -> None:
    """Whatever flags pytest was given, none of them are OUR flags."""
    assert _explicit_cli_flags() == set()


def test_p_flag_does_not_leak_as_preferred_provider() -> None:
    """The exact prefix collision: ``-p`` must not become ``-pp``.

    Asserted against an explicitly supplied argv so the test states the
    rule directly rather than relying on how it was launched.
    """
    assert _explicit_cli_flags(["tests", "-q", "-p", "no:randomly"]) == {
        "preferred_provider"
    }, "prefix matching is real — which is exactly why sys.argv must be isolated"


def _saved() -> dict:
    from tests.unit_tests.process.test_failover_issue77 import (
        _make_saved_launch_config,
    )

    return _make_saved_launch_config(preferred_provider=None)


def _args():
    # Reuse the suite's own arg factory rather than hand-rolling a
    # Namespace: ``merge_restart_config`` reads many attributes, and a
    # partial stand-in would fail for reasons unrelated to argv isolation.
    from tests.unit_tests.process.test_failover_issue77 import _make_cli_args

    return _make_cli_args()


def test_merge_backfills_preferred_provider_under_pytest() -> None:
    """The canonical victim of the leak.

    With ``preferred_provider`` wrongly in the explicit set, the merge
    treats a ``None`` CLI value as a deliberate override and the backfill
    never happens.
    """
    merged = merge_restart_config(_saved(), _args())
    assert merged.preferred_provider == "claude-code-sdk"


def test_merge_is_unaffected_by_pytest_p_flag() -> None:
    """Same assertion, but proving the isolation rather than assuming it:
    a subprocess runs one restart test WITH the poisonous flag."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit_tests/process/test_failover_issue77.py"
            "::test_merge_restart_config_backfill_preferred",
            "-q",
            "-p",
            "no:randomly",
        ],
        cwd=str(CLI_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "restart tests must pass regardless of pytest's own flags.\n"
        f"stdout:\n{result.stdout[-3000:]}"
    )
