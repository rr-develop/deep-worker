"""Child-death detection that works below the supported kernel floor.

``os.pidfd_open`` is the primitive every supervisor in this tree wants:
it turns "the child died" into a readable descriptor, so waiting is
event-driven and needs no polling.  It is also the primitive most likely
to be missing.  It needs the syscall (Linux >= 5.3) *and*
``<sys/pidfd.h>`` at CPython build time; WSL2 still ships 4.19 kernels
and uv-installed interpreters are built without the header, so either
half can be absent on a system that is otherwise fine.  The kernel
answers ``ENOSYS`` (an ``OSError``); a CPython without the header has no
attribute at all (an ``AttributeError``).

``docs/REQUIREMENTS.md`` states the rule: a post-4.19 syscall may be an
optimisation, never a requirement.  This module holds the two pieces
every caller needs to honour it — the call-time lookup that makes both
absences raise, and the ``waitpid`` fallback they all fall back to — so
the supervisors share one implementation instead of each carrying a copy.

The fallback is ``waitpid`` on a worker thread.  That is still
event-driven from asyncio's point of view — one Future resolves exactly
once, when the child dies — and it uses no ``WNOHANG`` loop, so it
complies with the no-polling policy enforced by ``test_no_polling_policy``.
"""

from __future__ import annotations

import os


def blocking_waitpid(pid: int) -> None:
    """Block in a worker thread until ``pid`` exits.

    Runs to completion once the kernel reaps the child; ``WNOHANG`` is
    NOT used, so we never busy-loop.  Idempotent against ``ECHILD``: by
    the time this is called the child has either been reaped already
    (``ChildProcessError``, swallowed) or still belongs to us and we
    block until it dies.
    """
    while True:
        try:
            waited_pid, _status = os.waitpid(pid, 0)
        except ChildProcessError:
            # Already reaped (e.g. by a previous test teardown).
            return
        if waited_pid == pid:
            return


def default_pidfd_open(pid: int, flags: int = 0) -> int:
    """``os.pidfd_open``, looked up at call time.

    Indirected so that a CPython built without ``<sys/pidfd.h>`` raises
    ``AttributeError`` from the call rather than at import time — which
    is what lets callers treat "missing syscall" (``OSError``/ENOSYS) and
    "missing build support" (``AttributeError``) as the single condition
    that selects :func:`blocking_waitpid`.
    """
    return os.pidfd_open(pid, flags)
