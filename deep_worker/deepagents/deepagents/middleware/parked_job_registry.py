"""THE single source of truth for background-job ids — one namespace: ``job_N``.

HARD CONTRACT (a past incident, 2026-07-31): whatever id is advertised to the
model as ``job_id`` MUST resolve in this registry. Advertising an id that
``wait_for_job``/``kill_job`` cannot honour is the defect this module exists to
make impossible.

Before this module there were TWO registries with TWO id namespaces:

  * ``BoundedToolCallMiddleware`` parked under the originating ``tool_call_id``
    (``toolu_…``) in a private dict nothing in production ever read;
  * ``CLIShellBackend.execute`` parked under ``job_N`` in its own dict, which
    is the ONLY one ``wait_for_job``/``kill_job`` resolved.

Both bounds default to 120s, and the middleware bound also covers the CLI
stage-A dispatch wait — so on CLI providers the middleware fired FIRST, handed
the model a ``toolu_…``, and the shell backend never minted the matching
``job_N``. The model then lost control of live background work.

Design notes:

  * ONE counter per agent process → ids minted by the middleware and by the
    shell backend can never collide.
  * Every registration may carry ONE alias (the originating ``tool_use_id``),
    so a model that echoes ``toolu_…`` still reaches the job. This preserves
    the alias-fix contract (2026-06-17).
  * ``drop`` purges the canonical key AND every alias pointing at it — no
    stale alias, no leak.
  * Process-local, NOT cross-agent global: one agent process owns one
    registry. This is not a second wait site, so POLICIES.md §7.1 (the single
    bounded wait point lives in ``BoundedToolCallMiddleware.awrap_tool_call``)
    is untouched.
  * Event-driven only. Nothing here polls.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

JobKind = Literal["tool", "shell"]


@dataclass
class ParkedJob:
    """A unit of work that outlived its bound and keeps running.

    ``handle`` is deliberately untyped: the middleware parks an
    ``asyncio.Task``, the shell backend parks its own job record. Each owner
    knows how to interpret its own handle; the registry only owns identity.
    """

    job_id: str
    kind: JobKind
    handle: Any
    tool_name: str | None = None
    aliases: set[str] = field(default_factory=set)


class ParkedJobRegistry:
    """Identity for parked jobs. One namespace (``job_N``), two lookup keys."""

    def __init__(self) -> None:
        self._jobs: dict[str, ParkedJob] = {}
        self._aliases: dict[str, str] = {}
        self._counter = itertools.count(1)

    def allocate(self) -> str:
        """Mint the next canonical id. The ONLY way a ``job_N`` is created."""
        return f"job_{next(self._counter)}"

    def register(
        self,
        job_id: str,
        *,
        kind: JobKind,
        handle: Any,
        alias: str | None = None,
        tool_name: str | None = None,
    ) -> ParkedJob:
        """Record a parked job under ``job_id``, optionally reachable by
        ``alias`` (the originating ``tool_use_id``).

        An alias equal to the canonical id is ignored, and an alias is never
        allowed to shadow a canonical id.
        """
        job = self._jobs.get(job_id)
        if job is None:
            job = ParkedJob(
                job_id=job_id, kind=kind, handle=handle, tool_name=tool_name
            )
            self._jobs[job_id] = job
        else:
            job.kind = kind
            job.handle = handle
            if tool_name is not None:
                job.tool_name = tool_name
        if alias and alias != job_id and alias not in self._jobs:
            self._aliases[alias] = job_id
            job.aliases.add(alias)
        return job

    def resolve(self, job_id: str) -> ParkedJob | None:
        """Look up by canonical ``job_N`` OR by ``tool_use_id`` alias."""
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        canonical = self._aliases.get(job_id)
        if canonical is not None:
            return self._jobs.get(canonical)
        return None

    def resolve_by_alias(self, alias: str) -> ParkedJob | None:
        """Look up ONLY by alias. Used to avoid minting a second id for a job
        that the other layer already parked under the same ``tool_use_id``."""
        canonical = self._aliases.get(alias)
        if canonical is None:
            return None
        return self._jobs.get(canonical)

    def drop(self, job_id: str) -> None:
        """Remove a finished/killed job and purge every alias pointing at it.

        Accepts either namespace and is idempotent.
        """
        job = self.resolve(job_id)
        if job is None:
            self._aliases.pop(job_id, None)
            return
        self._jobs.pop(job.job_id, None)
        for alias in list(job.aliases):
            self._aliases.pop(alias, None)
        job.aliases.clear()

    def clear(self) -> None:
        """Drop everything. For tests; production never needs it."""
        self._jobs.clear()
        self._aliases.clear()


_REGISTRY: ParkedJobRegistry | None = None


def get_parked_job_registry() -> ParkedJobRegistry:
    """The process-local registry shared by every parking site.

    Deliberately a module singleton: handing each middleware instance its own
    registry is exactly how the two-namespace defect arose.
    """
    global _REGISTRY  # noqa: PLW0603 — process-local singleton by design
    if _REGISTRY is None:
        _REGISTRY = ParkedJobRegistry()
    return _REGISTRY
