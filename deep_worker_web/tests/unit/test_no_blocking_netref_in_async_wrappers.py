"""AST guard: no blocking netref expression may sit in an argument position.

BACKLOG §12. Three production outages (2026-07-10, 2026-08-27, 2026-08-29)
came from one shape::

    fire_and_forget_async(
        root.get_last_response(),   # evaluated BEFORE the call
        timeout=5.0,                # cannot cover it
    )

Python evaluates arguments before entering the callee, so the RPC round-trip
happens on the caller's event loop, outside the task and outside ``timeout``.
``netref.__getattribute__`` is a synchronous request, and a peer that is
merely slow — not dead — freezes the loop inside attribute resolution.

Two guards now stand between this defect and production:

* ``fire_and_forget_call`` (rpyc_async) takes a factory and refuses a
  non-callable first argument at runtime;
* this test refuses the shape at rest, so it cannot even be committed.

The runtime guard alone is not enough: it only fires on a code path that
actually executes, and these call sites are error handlers and rarely-taken
branches. That is precisely how the same line survived two incidents.

Note that ``fire_and_forget_async`` was NOT changed: it is published API of
rpyc-async on PyPI, and its by-value signature stays valid for awaitables that
are cheap to create. The hazard is documented on it; the factory variant was
added alongside it. In this codebase every such call is remote, so both names
are scanned here.

Deliberately matched on the AST SHAPE (``<name>.<attr>(...)`` in an argument
of a known async wrapper), not on a variable being spelled ``root``. A guard
keyed to a name would be defeated by renaming the variable, which is exactly
the "test theatre" this repo has been burned by before.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Wrappers that must never be handed an already-evaluated remote call.
#
# Both names are scanned. ``fire_and_forget_call`` takes a factory and is the
# form the web tree uses. ``fire_and_forget_async`` is still valid published
# API of rpyc-async and is used deliberately in deep_worker_cli, where the
# argument is an AsyncResult already built by ``asyncreq`` — a local value,
# not a remote call, so it is not the shape this guard forbids.
#
# What the guard rejects is an EVALUATED ``x.y(...)`` in argument position,
# whichever wrapper receives it.
ASYNC_WRAPPERS = {"fire_and_forget_async", "fire_and_forget_call"}

# Locals that are known NOT to be netrefs. ``self``/``cls`` calls are ordinary
# local method calls; everything else is treated as suspect on purpose.
SAFE_RECEIVERS = {"self", "cls", "asyncio", "loop", "logger", "_logger", "json"}

WEB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WEB_ROOT.parent

SCANNED_FILES = [
    WEB_ROOT / "app" / "services" / "agent_service.py",
    WEB_ROOT / "app" / "services" / "mcp_service.py",
    WEB_ROOT / "app" / "routers" / "agents.py",
    REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli"
    / "process" / "agent_process.py",
    REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli"
    / "rpyc_service" / "agent_service.py",
]


def _offending_arguments(tree: ast.AST) -> list[tuple[int, str]]:
    """Find evaluated ``x.y(...)`` calls passed to an async wrapper."""
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if name not in ASYNC_WRAPPERS:
            continue

        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            # A lambda/def factory is the correct form — its body is not
            # evaluated here, so it is fine whatever it contains.
            if isinstance(arg, (ast.Lambda, ast.Name, ast.Constant)):
                continue
            if not isinstance(arg, ast.Call):
                continue
            inner = arg.func
            if not isinstance(inner, ast.Attribute):
                continue
            receiver = inner.value
            if isinstance(receiver, ast.Name) and receiver.id in SAFE_RECEIVERS:
                continue
            found.append((arg.lineno, f"{ast.unparse(arg)[:80]}"))

    return found


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda p: p.name)
def test_no_evaluated_remote_call_in_async_wrapper_arguments(path: Path):
    assert path.exists(), f"guard is pointed at a missing file: {path}"
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = _offending_arguments(tree)
    assert not offenders, (
        f"{path}: a remote call is evaluated in an argument to an async "
        f"wrapper. It runs on the event loop BEFORE the wrapper is entered, "
        f"so `timeout=` cannot cover it (BACKLOG §12).\n"
        + "\n".join(f"  line {ln}: {src}" for ln, src in offenders)
        + "\nUse a factory instead:\n"
        "  fire_and_forget_call(lambda: async_call(root, 'method', arg), ...)"
    )


def test_guard_detects_the_original_defect():
    """The guard must actually catch the shape that caused the outages.

    Without this, a guard that silently matches nothing would "pass" forever.
    """
    defective = (
        "fire_and_forget_async(\n"
        "    root.get_last_response(),\n"
        "    timeout=5.0,\n"
        ")\n"
    )
    offenders = _offending_arguments(ast.parse(defective))
    assert offenders, "guard failed to flag the exact 2026-08-29 defect"

    # The same mistake made against the new wrapper must also be caught.
    also_defective = (
        "fire_and_forget_call(\n"
        "    root.get_last_response(),\n"
        "    timeout=5.0,\n"
        ")\n"
    )
    assert _offending_arguments(ast.parse(also_defective)), (
        "guard must flag an evaluated remote call to fire_and_forget_call too"
    )

    fixed = (
        "fire_and_forget_call(\n"
        "    lambda: async_call(root, 'get_last_response'),\n"
        "    timeout=5.0,\n"
        ")\n"
    )
    assert not _offending_arguments(ast.parse(fixed)), (
        "guard flags the corrected factory form — it would block the fix"
    )
