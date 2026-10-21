"""Package-wide guard for the Linux 4.19 floor in ``docs/REQUIREMENTS.md``.

``test_web_daemon.py`` guards the web path. This guards every module in
``deep_worker_cli``: a post-4.19 syscall introduced anywhere fails here,
at author time, rather than on a WSL2 user's 4.19 kernel.

Two separate rules, because the two kinds of call differ:

* Calls with **no fallback in this tree** (``clone3``, ``openat2``,
  ``close_range``, ...) are banned outright. None are used today, so the
  rule costs nothing until someone reaches for one.
* ``pidfd_open`` / ``pidfd_send_signal`` ARE used — they make child-death
  detection event-driven — so they are allowed only in modules that
  demonstrably handle their absence. The allowlist below is the list of
  such modules, and the second test checks each entry really does catch
  ``AttributeError`` alongside ``OSError``: ENOSYS from the kernel and a
  CPython built without ``<sys/pidfd.h>`` are different exceptions, and
  catching only the first is the bug this whole document exists for.

Reaching the syscall indirectly does not escape the second rule. A
module that routes through ``child_death.default_pidfd_open``, or that
binds the call to an attribute (``self._pidfd_open``) and invokes it
there, is held to exactly the same obligation — the thing that can go
wrong is unchanged by the spelling. Both forms are in the tree, and a
module quietly leaving the guard's scope by adopting one of them is the
failure this file is written against.

What still bounds the oracle is name resolution: it matches the call by
name rather than resolving it, so a call assembled at runtime
(``getattr(os, "openat2")()``) would slip past. It catches the direct
form and the two indirections above, which is how every one of these has
actually been written.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

import deep_worker_cli


PACKAGE_ROOT = pathlib.Path(deep_worker_cli.__file__).parent


#: Post-4.19 syscalls with no guarded user in this tree. Adding one means
#: either writing the fallback (and moving it to the allowlist above) or
#: raising the floor documented in ``docs/REQUIREMENTS.md``.
FORBIDDEN = {
    "P_PIDFD": "waitid(P_PIDFD), Linux 5.4",
    "clone3": "Linux 5.3",
    "close_range": "Linux 5.9",
    "openat2": "Linux 5.6",
    "faccessat2": "Linux 5.8",
    "pidfd_getfd": "Linux 5.6",
    "memfd_secret": "Linux 5.14",
}

#: Modules allowed to call ``pidfd_open`` / ``pidfd_send_signal``, each
#: because it falls back when the call is unavailable. Keyed by path
#: relative to the package root.
PIDFD_ALLOWED = {
    "lsp/service/child_death.py",
    "process/web_daemon.py",
    "lsp/service/supervisor.py",
    "lsp/service/serena_backend.py",
    "semantic_memory/service/supervisor.py",
    "lsp/service/serena_supervisor.py",
}

#: Helpers that ARE the syscall, just spelled differently. Routing
#: ``os.pidfd_open`` through ``child_death.default_pidfd_open`` does not
#: change what can go wrong — the caller still has to handle both
#: ENOSYS and the missing attribute — so a call to one of these counts
#: exactly as a direct call for every rule below. Without this, a module
#: could adopt the helper and silently drop out of the guard's scope.
#: ``blocking_waitpid`` is deliberately absent: it is the fallback, not
#: the syscall, and carries no such obligation.
PIDFD_HELPERS = {"default_pidfd_open"}

_PIDFD_NAMES = {"pidfd_open", "pidfd_send_signal"} | PIDFD_HELPERS

#: Matches the call by its ATTRIBUTE name, whatever the module is bound
#: as. Anchoring on ``child_death.`` would miss
#: ``from ... import child_death as _cd`` — the module would adopt the
#: helper and quietly leave the guard's scope, which is the failure this
#: whole rule exists to prevent.
_PIDFD_CALL = re.compile(
    r"\.(?:pidfd_open|pidfd_send_signal|default_pidfd_open)\s*\("
)


def _package_sources() -> list[pathlib.Path]:
    return [
        p
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]


def test_no_unguardable_post_floor_syscall_in_the_package() -> None:
    """No module names a post-4.19 syscall that has no fallback here."""
    offenders: dict[str, list[str]] = {}
    for path in _package_sources():
        source = path.read_text()
        named = [
            f"{name} ({why})"
            for name, why in FORBIDDEN.items()
            if name in source
        ]
        if named:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = named

    assert not offenders, (
        f"{offenders} postdate the Linux 4.19 floor in "
        "docs/REQUIREMENTS.md; either guard the call with a fallback and "
        "list the module in PIDFD_ALLOWED, or raise the documented floor"
    )


def test_pidfd_is_called_only_where_a_fallback_exists() -> None:
    """``pidfd_*`` appears only in modules that handle its absence."""
    callers = {
        str(path.relative_to(PACKAGE_ROOT))
        for path in _package_sources()
        if _PIDFD_CALL.search(path.read_text())
    }

    unlisted = callers - PIDFD_ALLOWED
    assert not unlisted, (
        f"{sorted(unlisted)} call pidfd_* without being listed as having a "
        "fallback. A missing pidfd must degrade, never abort — see "
        "docs/REQUIREMENTS.md — so add the fallback and list the module."
    )

    stale = PIDFD_ALLOWED - callers
    assert not stale, (
        f"{sorted(stale)} are allowlisted but no longer call pidfd_*; "
        "drop them so the list keeps meaning something"
    )


@pytest.mark.parametrize("relpath", sorted(PIDFD_ALLOWED))
def test_each_pidfd_caller_catches_attributeerror_too(relpath: str) -> None:
    """Every guarded caller treats a missing attribute as a missing call.

    ``os.pidfd_open`` is absent entirely on a CPython built without
    ``<sys/pidfd.h>`` — that is ``AttributeError``, not ``OSError``.
    A handler that catches only ``OSError`` still crashes there, which is
    precisely the regression this file guards.
    """
    path = PACKAGE_ROOT / relpath
    tree = ast.parse(path.read_text())
    aliases = frozenset(_pidfd_aliases(tree))

    # EVERY try block wrapping a pidfd call must catch AttributeError.
    # Checking "the module has at least one such handler" would let a
    # correctly-guarded site vouch for a narrowed one next to it.
    narrowed: list[int] = []
    guarded = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if _nested_try_handles_pidfd(node, aliases):
            # An outer block wrapping an inner guarded one. The inner
            # block is what protects the call; judging the outer one
            # would flag code that is already correct.
            continue
        if not _directly_calls_pidfd(node.body, aliases):
            continue
        caught: set[str] = set()
        for handler in node.handlers:
            caught |= _exception_names(handler.type)
        if "AttributeError" in caught or "Exception" in caught:
            guarded += 1
        else:
            narrowed.append(node.lineno)

    if guarded == 0 and not narrowed:
        # The call is routed through a helper (e.g. child_death); the
        # helper itself is allowlisted and checked on its own.
        pytest.skip(f"{relpath} calls pidfd_* outside a try block it owns")

    assert not narrowed, (
        f"{relpath} guards a pidfd call at line(s) {narrowed} without "
        "catching AttributeError; a CPython built without <sys/pidfd.h> "
        "raises that, not OSError"
    )


def test_child_death_helper_stays_import_light() -> None:
    """Importing the helper must not drag in the agent stack.

    ``lsp/service/supervisor.py`` is on the LSP bridge's import path, and
    the bridge is asserted to load no ML/agent packages
    (``test_bridge_thin_entry.py``). The helper first lived under
    ``deep_worker_cli/process/``, whose ``__init__`` imports
    ``AgentProcess`` — that alone pulled torch and langchain into the
    bridge and turned those assertions red. Keeping it in a package with
    an empty ``__init__`` is what makes it safe to import from there.
    """
    source = (
        PACKAGE_ROOT / "lsp" / "service" / "child_death.py"
    ).read_text()
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module:
                imported.add(node.module.split(".")[0])

    # __future__ is compiler machinery, not a dependency.
    imported.discard("__future__")

    assert imported <= {"os"}, (
        f"child_death imports {sorted(imported)}; it is reachable from the "
        "LSP bridge, which must stay free of the agent stack. Keep it to "
        "the standard library."
    )


def _pidfd_aliases(tree: ast.AST) -> set[str]:
    """Attribute names a module binds the syscall to.

    ``SerenaSupervisor`` stores it as ``self._pidfd_open`` so tests can
    inject a stub, then calls through that attribute. Matching only
    ``os.pidfd_open`` would miss the guarded call entirely and the module
    would drop out of scope — which is exactly how a narrowed handler
    there went undetected. Any attribute assigned from the syscall (or
    from a helper that is the syscall) is treated as the syscall.
    """
    # Functions whose body calls the syscall are themselves the syscall
    # for this purpose: binding one to an attribute makes that attribute
    # a call site. ``SerenaSupervisor`` does exactly this, via
    # ``self._pidfd_open = pidfd_open_fn or self._default_pidfd_open``.
    wrappers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_mentions_pidfd(stmt) for stmt in node.body)
    }

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (_mentions_pidfd(node.value) or _mentions(node.value, wrappers)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                aliases.add(target.attr)
            elif isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _mentions(node: ast.AST, names: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in names:
            return True
        if isinstance(child, ast.Name) and child.id in names:
            return True
    return False


def _mentions_pidfd(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in _PIDFD_NAMES:
            return True
        if isinstance(child, ast.Name) and child.id in _PIDFD_NAMES:
            return True
    return False


def _directly_calls_pidfd(
    body: list[ast.stmt], aliases: frozenset[str] = frozenset()
) -> bool:
    """True if ``body`` calls pidfd_* outside any nested try block."""
    names = _PIDFD_NAMES | set(aliases)
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Try):
                # Belongs to the inner block, judged on its own.
                continue
            if isinstance(node, ast.Attribute) and node.attr in names:
                if not _inside_nested_try(stmt, node):
                    return True
    return False


def _inside_nested_try(stmt: ast.stmt, target: ast.AST) -> bool:
    for node in ast.walk(stmt):
        if isinstance(node, ast.Try):
            for inner in ast.walk(node):
                if inner is target:
                    return True
    return False


def _nested_try_handles_pidfd(
    node: ast.Try, aliases: frozenset[str] = frozenset()
) -> bool:
    """True if a try block nested inside ``node``'s body guards pidfd_*."""
    for stmt in node.body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Try) and _directly_calls_pidfd(
                child.body, aliases
            ):
                return True
    return False


def _exception_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        out: set[str] = set()
        for elt in node.elts:
            out |= _exception_names(elt)
        return out
    return set()
