"""Suite-wide safety net: cap how much memory a test run may consume.

Why this exists
---------------
On 2026-08-29 a single test drove one pytest process to 19.5 GB RSS and locked
the host up twice, hard enough to need a power cycle both times. The test itself
allocated almost nothing: it recursed until ``RecursionError``, and pytest's
traceback formatting then compared thousands of stack frames whose locals were
``MagicMock``s, each comparison allocating more mocks. See a related internal
incident analysis (not included here).

The specific defect is fixed and has regression tests. This cap exists because
the *class* of failure is not specific to that test: any runaway allocation
inside pytest can take the whole machine down, and a developer's first warning
should be a dead test process, not an unresponsive host.

``RLIMIT_AS`` is inherited by child processes, so tests that spawn subprocesses
are covered too. The limit is on address space rather than RSS because that is
what a process can be prevented from *requesting* — the allocation fails with
``MemoryError`` and pytest reports an ordinary failure, instead of the kernel
having to reclaim memory that was already handed out.

Override with ``DEEPAGENTS_TEST_MEMORY_LIMIT_GB``; set it to ``0`` to disable
(e.g. when profiling something legitimately large).
"""

from __future__ import annotations

import os
import resource

_DEFAULT_LIMIT_GB = 8.0


def _configure_memory_limit() -> None:
    raw = os.environ.get("DEEPAGENTS_TEST_MEMORY_LIMIT_GB")
    try:
        limit_gb = _DEFAULT_LIMIT_GB if raw is None else float(raw)
    except ValueError:
        limit_gb = _DEFAULT_LIMIT_GB
    if limit_gb <= 0:
        return

    limit_bytes = int(limit_gb * 1024**3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    # Never raise an existing limit, and never exceed the hard limit: an outer
    # harness (a cgroup runner, CI) may have set something stricter on purpose.
    if soft != resource.RLIM_INFINITY and soft <= limit_bytes:
        return
    if hard != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, hard)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except (ValueError, OSError):
        # Not fatal: the suite must still run where the limit cannot be applied.
        pass


_configure_memory_limit()
