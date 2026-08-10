"""pytest-only ``sitecustomize.py`` shim — installs a fake claude SDK.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §5.2
("Chosen mechanism (round-3 BLOCKER fix)").

This module is **never** imported from production code. It ships under
``claude_code_provider/tests/_support/`` and is picked up exclusively by
CPython's standard ``site`` machinery when the pytest fixture prepends this
directory to a child interpreter's ``PYTHONPATH``. The
``test_no_prod_import_of_test_support`` lint test enforces the no-prod-import
invariant on every commit.

Double gate — both must be true or this module returns immediately with no
side effects:

  1. This directory is on ``sys.path`` (i.e. the pytest fixture prepended it
     to the child's ``PYTHONPATH``).
  2. Environment variable ``DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST=1`` is set
     (the pytest fixture sets it explicitly; production processes do not).

If either gate is missing, this shim is a no-op. Under the ``-S`` interpreter
flag Python does not import ``site`` at all — the fixture asserts on the
child's argv that ``-S`` is absent and, if the shim never runs, the canary
below never fires and the test fails with a clean "seam bootstrap did not
fire" message instead of an ambiguous timeout.
"""

from __future__ import annotations

import os as _os


def _install() -> None:
    if _os.environ.get("DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST") != "1":
        return
    # Import the heavy monkeypatch machinery lazily — only under the pytest
    # gate. Production imports of this file (should any regression add one)
    # therefore do not pay for it, and the no-prod-import lint test catches
    # the regression regardless.
    #
    # ``_fake_sdk_boot`` lives right next to us; because the parent
    # directory is on ``sys.path`` (that is exactly how ``site`` found this
    # ``sitecustomize`` module), the top-level module name resolves.
    # Reaching it as ``claude_code_provider.tests._support._fake_sdk_boot``
    # would require the whole ``claude_code_provider`` package to already
    # be importable, which is only true AFTER the agent's own sys.path
    # setup runs later — i.e. NOT at ``site`` bootstrap time. So we import
    # by the top-level name here.
    import _fake_sdk_boot  # noqa: PLC0415  (test-only module)
    _fake_sdk_boot.install()


_install()
