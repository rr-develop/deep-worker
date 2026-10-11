"""Delivery barrier: spawn the restart driver only after the result is recorded.

WHAT DELIVERY MEANS
-------------------
A tool result is delivered when, and ONLY when, it has been written into the
model's context — that is, persisted to the corresponding checkpoint. Nothing
else is delivery, and none of the following may be used as a substitute for it:
a successful transport write, a hook attached to any transport-level event, the
absence of an interrupt/stop/restart/exception, teardown or end-of-turn state,
or an acknowledgement from a provider, CLI or MCP transport. A transport can
accept bytes that never reach the model's context; keying the barrier to the
transport would certify something other than what it claims to certify, and
would reintroduce the very defect this module exists to prevent.

THE PROBLEM
-----------
``restart_self`` must do two things that are in direct tension:

  * hand its tool result back, so the transcript records what actually
    happened; and
  * get this process stopped, which is what a restart means.

The stop is performed by ``deep-worker --restart``, a SEPARATE process. If it
is spawned before the result is recorded, the two race — and the driver's route
to SIGTERM is short. ``return``-ing from the tool handler creates no
happens-before edge with delivery; it only means the *Python function*
finished.

Losing that race reproduces the exact defect this module exists to prevent: the
CLI sees a ``tool_use`` with no matching result, classifies the dangling call
under its default ``toolDenialKind: "user-rejected"``, and feeds the model a
fabricated "The user doesn't want to proceed with this tool use" plus a
"[Request interrupted by user for tool use]" line. Both are replayed into the
next process on resume. The user denied nothing. (5 of 5 historical restarts
were mis-recorded this way, one of them stamped 170 ms AFTER the replacement
process had already started — proof the pairing is a timing artifact.)

THE FIX
-------
Invert the order. The tool ARMS a restart instead of launching one:

    restart_self  ──arm_self_restart(argv, call_id)──▶  [pending]
    result → aput → await conn.commit()                 ← delivery
    commit listener  ──fire_armed_self_restart(ids)──▶  Popen(driver)
                                                        driver SIGTERMs us

The signal is the checkpointer's commit listener
(``DeltaSqliteCheckpointer.add_commit_listener``), which fires strictly after
``await self.conn.commit()`` and carries the ``tool_call_id`` of every
``ToolMessage`` in the committed checkpoint. That checkpointer is the single
persistence path for EVERY provider, so the barrier is provider-complete: no
transport hook, no vendored SDK patch, no per-provider gate.

Nothing can stop this process until its own tool result is committed, because
the only thing that stops it is spawned after that point. The barrier is
one-shot and process-local: the armed command is consumed when fired.

CORRELATION
-----------
Tool calls can be committed concurrently, so the barrier records the
``tool_call_id`` of the ``restart_self`` call — injected into the tool by
LangChain, so it is known at arming time — and fires only when that exact id
appears among the committed ids. Firing on a commit that does not carry our
result would stop the process while it was still unrecorded — the same
dangling ``tool_use``, reached by a different route.

END OF TURN
-----------
If the turn ends with the restart still armed, the result was never recorded.
``disarm_undelivered_self_restart`` then drops it and logs an ERROR; it does
NOT spawn. Spawning at that point would stop the process on precisely the path
where the transcript may already carry a dangling ``tool_use`` — reinstating
the defect. An agent that visibly failed to restart is recoverable; a
fabricated denial replayed into the next process is not.
"""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 - spawning the restart driver is the point
import threading

logger = logging.getLogger(__name__)

# Process-local, one-shot. An agent process restarts itself at most once —
# after the driver lands SIGTERM there is no "next" restart to hold.
#
# ``_armed_call_id`` is the ``tool_call_id`` of the ``restart_self`` call that
# armed us. It is REQUIRED, not decorative: tool calls run concurrently, so a
# barrier that fired on any commit would spawn the driver when some OTHER
# tool's result was committed first — while restart_self's own result was
# still unrecorded. That is the original dangling-``tool_use`` race,
# reintroduced. Only a commit carrying our id may fire.
_lock = threading.Lock()
_armed: list[str] | None = None
_armed_call_id: str | None = None

# True once a checkpointer with a commit listener is wired up. This is NOT a
# provider capability — every provider persists through the same
# checkpointer. It is false only where there is no durable commit at all
# (``InMemorySaver`` under --memory-only or tests), and there a restart
# cannot be confirmed, so the tool must fail instead of promising one.
_delivery_available = False


def mark_self_restart_delivery_available() -> None:
    """Record that a commit-listening checkpointer is wired up."""
    global _delivery_available
    with _lock:
        _delivery_available = True


def self_restart_delivery_available() -> bool:
    """True when a durable commit signal exists for this process."""
    with _lock:
        return _delivery_available


def set_self_restart_delivery_available_for_test(available: bool) -> None:
    global _delivery_available
    with _lock:
        _delivery_available = available


def arm_self_restart(argv: list[str], call_id: str) -> None:
    """Record the driver command to run once this call's result is delivered.

    ``call_id`` is the ``tool_call_id`` of the ``restart_self`` call itself,
    injected by LangChain (``InjectedToolCallId``). Arming is therefore
    BOUND from the start: the tool knows its own identity and does not need
    a transport layer to supply one.

    Deliberately does NOT spawn anything. See module docstring for why the
    spawn must not race the result.
    """
    global _armed, _armed_call_id
    with _lock:
        _armed = list(argv)
        _armed_call_id = call_id
    logger.info(
        f"Self-restart armed for tool_call_id={call_id!r}; the driver will "
        f"be spawned once THAT call's result has been committed."
    )


def self_restart_armed() -> bool:
    """True when a restart has been armed and not yet fired."""
    with _lock:
        return _armed is not None


def armed_self_restart_call_id() -> str | None:
    """The tool_call_id the pending restart is waiting on, if any."""
    with _lock:
        return _armed_call_id


def fire_armed_self_restart(committed_call_ids: frozenset[str] | None = None) -> bool:
    """Spawn the armed driver iff the armed call is among the committed ids.

    Called once per durable commit, with the ``tool_call_id`` of every
    ``ToolMessage`` in that checkpoint. Returns True only when the driver
    was actually spawned.

    Correlation is mandatory. A commit that does not carry OUR call must
    NOT fire the restart: restart_self's own result may still be
    unrecorded, and stopping the process then strands its ``tool_use`` —
    which the CLI records as a fabricated user rejection. An empty or
    absent set is likewise not a match.
    """
    global _armed, _armed_call_id
    with _lock:
        if _armed is None:
            return False
        if _armed_call_id is None or _armed_call_id not in (committed_call_ids or frozenset()):
            # Our result is not in this commit — leave the restart armed
            # for the commit that actually carries it.
            return False
        argv = _armed
        _armed = None
        _armed_call_id = None

    try:
        # S603: argv is built by ``restart_self`` from ``shutil.which`` plus
        # the agent's own name and a locally-minted token — never from model
        # output or any external caller.
        subprocess.Popen(  # noqa: S603
            argv,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        # The tool already reported success to the model, so this is the only
        # place the failure can surface. Loud, not silent.
        logger.error(
            f"Armed self-restart failed to spawn ({exc}). The agent is still "
            f"running and was NOT restarted."
        )
        return False

    logger.info(
        "Self-restart driver spawned after the result was committed to the "
        "checkpoint."
    )
    return True


def disarm_undelivered_self_restart() -> bool:
    """Drop a still-armed restart at end of turn. NEVER spawns.

    Returns True if something was armed (i.e. the restart did not happen).

    ⚠️ This deliberately does NOT fire the driver, and an earlier revision
    that did was wrong. Reaching end of turn with the restart still armed
    means the ``restart_self`` result was never confirmed delivered —
    provider error, cancelled turn, dead transport. Spawning here would
    stop the process on exactly the path where the transcript may already
    hold a dangling ``tool_use``, which is the fabricated-denial defect
    this module exists to prevent. "The turn is over" does not imply "the
    result got out".

    So the failure is made loud instead of papered over: the agent keeps
    running, the operator sees an ERROR, and the model can ask again. A
    visible non-restart beats an invisible false denial.
    """
    global _armed, _armed_call_id
    with _lock:
        was_armed = _armed is not None
        call_id = _armed_call_id
        _armed = None
        _armed_call_id = None

    if was_armed:
        logger.error(
            f"Self-restart for tool_call_id={call_id!r} was armed but its "
            f"result was never confirmed delivered; the restart has been "
            f"ABANDONED rather than spawned. Stopping the process with an "
            f"undelivered result is what fabricates a 'user-rejected' "
            f"denial. The agent is still running."
        )
    return was_armed


def reset_armed_self_restart_for_test() -> None:
    """Clear module state between tests. Not used in production."""
    global _armed, _armed_call_id
    with _lock:
        _armed = None
        _armed_call_id = None
