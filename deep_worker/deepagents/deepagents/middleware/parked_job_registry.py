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
from collections.abc import Callable
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
    #: Invoked once the job is erased, for an owner with bookkeeping of its
    #: own to clear. Set by ``park``; see the note there before relying on it.
    on_release: Callable[[str], None] | None = None


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

    def park(
        self,
        handle: Any,
        *,
        kind: JobKind = "tool",
        alias: str | None = None,
        tool_name: str | None = None,
        on_release: "Callable[[str], None] | None" = None,
    ) -> ParkedJob:
        """Park a live unit of work. THE single entry point for parking.

        This owns the whole lifetime, which is why every caller must come
        through here instead of hand-rolling allocate + register + a
        done-callback of its own (POLICIES.md §14):

        * reuses the id already parked under ``alias``, so two layers parking
          the same call cannot mint two ids;
        * allocates a canonical ``job_N`` otherwise;
        * attaches the completion hook ITSELF, so no caller can get the
          "erase on completion" rule wrong again. Completion RETIRES the job
          (it stays reachable, holding a result nobody has collected yet);
          only ``complete`` erases it.

        ``on_release`` is invoked when the job is finally erased, for owners
        that must clear bookkeeping of their own. Prefer not needing it: a
        second table shadowing this registry is exactly what §14 forbids.
        """
        existing = self.resolve_by_alias(alias) if alias else None
        job_id = existing.job_id if existing is not None else self.allocate()
        job = self.register(
            job_id,
            kind=kind,
            handle=handle,
            alias=alias,
            tool_name=tool_name,
        )
        job.on_release = on_release
        # The completion hook lives on the handle itself, or on the awaitable
        # a wrapper handle carries. Finding it is the registry's job, so a
        # caller never has to know which shape it holds — that knowledge
        # leaking outwards is how five call sites each grew their own version.
        awaitable = getattr(handle, "task", handle)
        add_done_callback = getattr(awaitable, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(lambda _h: self.retire(job_id))
        return job

    def complete(self, job_id: str) -> None:
        """Erase a job whose result has been delivered to its caller.

        The ONLY legitimate reason to erase a record: someone now holds the
        result. Erasing on completion instead was a past defect —
        the work finished, the record vanished, and the caller was told "no
        background job with id" (POLICIES.md §14).
        """
        job = self.resolve(job_id)
        if job is None:
            self._aliases.pop(job_id, None)
            return
        release = job.on_release
        self.drop(job.job_id)
        if release is not None:
            release(job.job_id)

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

    def retire(self, job_id: str) -> None:
        """Mark a finished job terminal WITHOUT making it unreachable.

        Completion must not erase identity. A job whose work is done still
        owes the caller an answer: ``wait_for_job`` has to be able to resolve
        it and hand back the result it produced.

        This is a past incident (2026-08-28): the sync parking site
        called ``drop`` from ``add_done_callback``, so finishing deleted the
        entry and the completed result became unreachable — the agent was
        told "no background job with id" and could never confirm the work.
        Erasure is ``drop``'s job, and only a caller that has already been
        served (or an explicit kill) may do it.

        Deliberately does NOT mutate the entry: reachability via ``resolve``
        IS the terminal state, and the handle already knows whether its work
        finished. A separate ``retired`` flag would be written and never read
        — a second source of truth that could disagree with the handle.

        The method exists so the parking site states its intent explicitly
        ("this job is done, keep it reachable") instead of silently omitting
        the ``drop`` that used to be here.

        Idempotent; accepts either namespace.
        """

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


def resolve_parked_job_key(
    call_id: str, jobs: "dict[str, Any]"
) -> str | None:
    """Find a provider's local job key from either id namespace.

    A late result arrives under the originating ``call_id``, while the local
    table is keyed by canonical ``job_N``; the shared registry knows the alias
    that links them. Shared because all three provider proxy servers need
    exactly this, and three hand-written copies had already drifted apart —
    one returned a bool, two returned nothing (POLICIES.md §14).

    Returns the key into ``jobs``, or ``None`` if this call has no parked job.
    """
    if call_id in jobs:
        return call_id
    parked = get_parked_job_registry().resolve(call_id)
    if parked is not None and parked.job_id in jobs:
        return parked.job_id
    return None


def complete_parked_job(job_id: str) -> None:
    """Erase a parked job whose result has just been delivered.

    Thin by design: the point is that no caller decides for itself when a
    record dies.
    """
    get_parked_job_registry().complete(job_id)
