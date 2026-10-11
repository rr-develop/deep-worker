"""Anti-recurrence: the unit suite may not write to the real ``~/.deep_worker``.

Owner directive (2026-08-17), verbatim:

    "Tests must not mess up production configs. NEVER!!!!!!!!!!!!!!!!!"

What happened: ``test_handle_invoke_session_limit`` isolated only its
``SessionManager``, while the code under test reached
``_schedule_failover`` → ``AvailabilityTracker()`` with no ``data_dir``,
whose default is ``deep_worker_home()`` — the live runtime tree. One green
test rewrote the host-wide
``~/.deep_worker/provider_availability.json``, inventing usage-limit
records attributed to agents that do not exist (``test_session_limit``,
``test_failover77``). Those records mark providers unavailable, and
``AvailabilityTracker.is_available`` gates real failover decisions — so a
passing test silently degraded production's ability to fail over, and the
owner noticed a provider "blocked" while an agent was happily using it.

The fix is the autouse fixture in ``tests/conftest.py`` that repoints
``HOME`` (and therefore ``deep_worker_home()``) at a tmp directory. This
module guards the fixture itself: a fixture nobody checks is a fixture
that rots.

Note ``DEEP_WORKER_HOME`` deliberately does not exist as an env var —
``deep_worker_home()`` derives from ``Path.home()`` unconditionally, which
``test_restart_with_fable_alias`` relies on. So ``HOME`` is the only
correct lever, and patching it is what the fixture does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import deep_worker_home


def _real_home() -> Path:
    """The operator's home, resolved without consulting ``HOME``.

    ``HOME`` is what the fixture under test moves, so reading it here would
    make every assertion below vacuously true. ``pwd`` carries the account's
    real home independently of the environment. Deliberately computed rather
    than written as a literal: a hardcoded path is operator PII (the
    pre-commit scanner rejects it) and would only be correct on one machine.
    """
    import pwd

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):  # pragma: no cover — no passwd entry
        return Path(os.path.expanduser("~")).resolve()


REAL_HOME = _real_home()
PROD_TREE = REAL_HOME / ".deep_worker"


def _fixture_is_active(request) -> bool:
    """Whether this run collected the autouse HOME sandbox.

    Must be evaluated INSIDE a test, not at import time: the fixture is
    per-test, so at import HOME is still the operator's own and a
    module-level flag would read False always — silently skipping both
    assertions instead of checking them.

    The fixture lives in ``deep_worker_cli/tests/conftest.py`` and is not
    collected when pytest is rooted at the repository top, so its absence
    is a legitimate "not applicable" rather than a failure.
    """
    return "_sandbox_deep_worker_home" in request.fixturenames


def test_home_is_redirected_away_from_the_operators_tree(request):
    """Inside the suite, HOME must not be the operator's real home."""
    if not _fixture_is_active(request):
        pytest.skip("autouse HOME sandbox not collected from this rootdir")
    current = Path(os.path.expanduser("~")).resolve()
    assert current != REAL_HOME, (
        "HOME still points at the operator's home directory; the autouse "
        "isolation fixture in tests/conftest.py is not active"
    )


def test_deep_worker_home_resolves_into_a_temp_tree(request):
    """The accessor every writer goes through must land in tmp."""
    if not _fixture_is_active(request):
        pytest.skip("autouse HOME sandbox not collected from this rootdir")
    resolved = deep_worker_home().resolve()
    assert resolved != PROD_TREE.resolve(), (
        f"deep_worker_home() resolves to the production tree {resolved}"
    )
    assert "pytest" in str(resolved) or "tmp" in str(resolved).lower(), (
        f"expected a temp-dir sandbox, got {resolved}"
    )


def test_a_default_constructed_tracker_writes_inside_the_sandbox(monkeypatch, tmp_path):
    """The exact leak that happened: no ``data_dir`` passed.

    This is the regression test proper — it reproduces the original bug
    shape (construct the tracker the way production does, with no
    ``data_dir``) and proves the write lands in the sandbox.

    It re-asserts the HOME redirect locally instead of trusting the autouse
    fixture: this module lives under ``deep_worker_cli/tests``, but pytest
    can be invoked from the repository root (``pytest tests/…``), where
    that ``conftest.py`` is not collected and the fixture never runs. A
    guard that WRITES must not depend on the very protection it verifies —
    otherwise running it the wrong way round makes it the leak.
    """
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    tracker = AvailabilityTracker()
    tracker.record_failure(
        provider="grok",
        marker_text="unit-test marker — must never reach the real tree",
        agent_name="test_tests_never_touch_prod_state",
    )

    written = tracker._file.resolve()
    assert written != (PROD_TREE / "provider_availability.json").resolve(), (
        "a default-constructed AvailabilityTracker just wrote to the "
        "production availability file"
    )
    assert written.exists()


def test_the_production_availability_file_is_untouched_by_this_module():
    """Belt and braces: the real file's contents did not change.

    Reads the real path directly (bypassing the redirected HOME) and
    checks it does not contain this module's marker string.
    """
    prod_file = PROD_TREE / "provider_availability.json"
    if not prod_file.exists():
        pytest.skip("no production availability file on this host")
    assert "test_tests_never_touch_prod_state" not in prod_file.read_text(), (
        "this test module's marker leaked into the production file"
    )


def test_home_isolation_does_not_poison_a_foreign_interpreter(tmp_path):
    """The sandbox must not force this interpreter's user-site onto a child
    running a DIFFERENT Python version.

    Regression: the fixture exported the real user-site through
    ``PYTHONPATH``, which every child Python inherits regardless of its
    version. A child on another minor version then imported 3.12 extension
    modules and died with ``No module named 'pydantic_core._pydantic_core'``
    (an ABI mismatch, not a missing package). That broke the four
    ``test_cli_fast_return`` tests, which spawn the installed ``deep-worker``
    binary from a 3.14 virtualenv.

    ``PYTHONUSERBASE`` is the correct lever: the child derives its OWN
    version-specific user-site from it and runs ``.pth`` processing
    natively, so a version mismatch degrades to "no user packages" instead
    of "incompatible user packages".
    """
    forced = os.environ.get("PYTHONPATH", "")
    version_tag = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    poisoned = [
        part
        for part in forced.split(os.pathsep)
        if part and version_tag in part and part.startswith(str(Path.home().anchor))
    ]
    assert not poisoned, (
        "the HOME sandbox forced this interpreter's version-specific "
        f"site-packages onto PYTHONPATH ({poisoned}); any child Python on a "
        "different version will inherit it and fail on an ABI mismatch. "
        "Use PYTHONUSERBASE so the child resolves its own user-site."
    )


def test_home_isolation_still_reaches_user_installed_packages(tmp_path):
    """The counterweight to the test above: isolation must not silently
    strip dependencies from same-version children either.

    Whatever mechanism replaces PYTHONPATH has to keep a child on THIS
    interpreter able to import a user-installed package.

    Two assertions, because the behavioural one alone is not sufficient.
    Under ``uv-run.sh`` ``sys.executable`` is the ``.venv`` interpreter,
    whose own site-packages already carry every dependency — so the child
    imports fine whether or not the fixture forwards anything, and a fixture
    that forwards NOTHING still passes here. That gap is real: on an
    interpreter without a venv (CI, system Python) the same fixture would
    silently strip user-installed packages. So pin the mechanism directly
    as well as its effect.

    Credit: gap found by an M-nothing mutation, which this file
    passed 6/6 before this assertion existed.
    """
    assert os.environ.get("PYTHONUSERBASE"), (
        "the HOME sandbox did not export PYTHONUSERBASE; a child on an "
        "interpreter without a venv would resolve its user-site from the "
        "sandboxed HOME and lose every user-installed package"
    )

    result = subprocess.run(
        [sys.executable, "-c", "import langchain; print('ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and "ok" in result.stdout, (
        "a subprocess on this same interpreter lost its user-installed "
        f"packages under the HOME sandbox: {result.stderr!r}"
    )
