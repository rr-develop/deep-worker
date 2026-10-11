"""Durable spool for inbound inter-agent messages.

Fixes a related internal incident analysis (not included here) per
``docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md`` (signed off, revision 6).

Before this module, ``exposed_enqueue_message`` acknowledged a delivery right
after ``asyncio.create_task(_bg_turn())``. At that instant the payload existed
only as a captured local in a coroutine that had not started: the recipient's
death (``restart_self`` → ``os._exit``) destroyed it with no trace, while the
sender had already been told ``success: true``.

This spool makes the acknowledgement honest: a record is fsynced to disk
*before* the ack is produced, so an accepted message survives abrupt process
death and is recovered by the startup drain.

Layout (design §3.1)::

    <data_dir>/inbox/                    0o700
        <seq>-<uuid4hex>.json            0o600   pending
        .tmp-<uuid4hex>                  0o600   in-flight write, never published
        corrupt/<original name>          0o600   unparseable, never retried

``seq`` is a 12-digit zero-padded integer allocated ONLY by the recipient
process, so lexicographic order is arrival order.

Durability contract (design §3.2.1), deliberately not overclaimed:

* **Guaranteed** — survives abrupt process death (``SIGKILL``, ``os._exit``),
  which is the incident's actual failure mode.
* **Best-effort** — survives OS crash / power loss. The fsync chain is built
  for it, but that depends on the filesystem and storage stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_logger = logging.getLogger(__name__)


class InboxDuplicateMessageError(Exception):
    """Raised when a message already in the agent's context is sent again.

    Not a transport failure: the publish was refused ON PURPOSE, because
    automatic resends of an already-received message are forbidden. Carries its
    own type so the RPyC layer can report it as a distinct, non-retryable
    outcome instead of an unexpected error with a traceback.
    """

_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600

#: Schema version of a spool record. Bump when the body shape changes so a
#: future reader can migrate rather than misparse.
RECORD_VERSION: Final = 1

_SEQ_WIDTH: Final = 12
_TMP_PREFIX: Final = ".tmp-"
_CORRUPT_DIRNAME: Final = "corrupt"
_EXPIRED_DIRNAME: Final = "expired"
_LEDGER_NAME: Final = "delivered.log"
#: 32 hex chars + newline. Fixed width, so only the tail can ever be torn.
_LEDGER_LINE_WIDTH: Final = 33

#: Ledger of CONTENT fingerprints of messages already recorded in the agent's
#: context. Separate file from ``delivered.log`` on purpose: that one answers
#: "has this record been delivered" (identity = ``delivery_id``), this one
#: answers "has this MESSAGE been received" (identity = content + origin).
#: Merging them would overload one ledger with two different notions of
#: identity and two different lifetimes.
_RECEIVED_NAME: Final = "received.log"
#: 64 hex chars (sha256) + newline. Fixed width, as above.
_RECEIVED_LINE_WIDTH: Final = 65

#: Failed delivery attempts before a record is retired. Only a genuine turn
#: FAILURE counts; a user interrupt never does (design §3.1).
MAX_FAILURES: Final = int(os.environ.get("DEEP_WORKER_INBOX_MAX_FAILURES", "5"))
#: Wall-clock age, from the record's ``ts``, before retirement. 72h: the spool
#: carries more than review verdicts and weekend-offline delivery is legitimate.
RECORD_TTL: Final = float(os.environ.get("DEEP_WORKER_INBOX_RECORD_TTL", "259200"))
#: Age at which a still-deliverable record is marked stale in its body, so the
#: recipient MODEL can discount it. Logging alone cannot achieve that.
STALE_MARK: Final = float(os.environ.get("DEEP_WORKER_INBOX_STALE_MARK", "3600"))
#: Fail-loud tripwire on the PRE-compaction ledger. NOT a correctness knob: it
#: never truncates the correctness set (design §3.1).
LEDGER_CAP: Final = int(os.environ.get("DEEP_WORKER_INBOX_LEDGER_CAP", "50000"))

#: ``<seq>-<uuid4hex>.json`` — the published form.
_RECORD_RE: Final = re.compile(r"^(\d{%d})-([0-9a-f]{32})\.json$" % _SEQ_WIDTH)


def inbox_dir(data_dir: Path) -> Path:
    """Spool directory for ``data_dir``."""
    return Path(data_dir) / "inbox"


def corrupt_dir(data_dir: Path) -> Path:
    """Quarantine directory for unparseable records."""
    return inbox_dir(data_dir) / _CORRUPT_DIRNAME


def expired_dir(data_dir: Path) -> Path:
    """Retirement directory: records past the TTL or the failure cap.

    Retired records are MOVED here, never deleted — a silent drop is forbidden
    (POLICIES §9/§12). This directory is the durable half of the loud WARNING
    emitted by :func:`retire`.
    """
    return inbox_dir(data_dir) / _EXPIRED_DIRNAME


def ledger_path(data_dir: Path) -> Path:
    """Path of the dedup ledger: one 32-hex ``delivery_id`` per line."""
    return inbox_dir(data_dir) / _LEDGER_NAME


def received_path(data_dir: Path) -> Path:
    """Path of the received-content ledger: one 64-hex fingerprint per line."""
    return inbox_dir(data_dir) / _RECEIVED_NAME


def content_fingerprint(message: str, reply_back: bool) -> str:
    """Stable identity of a MESSAGE, as opposed to of a delivery record.

    ``delivery_id`` is minted per send (``uuid4``), so two sends of identical
    text are, to the receiver, two unrelated records: dedup on it can never
    recognise a resend. This fingerprint is derived from what the message IS,
    so the same text sent twice fingerprints the same both times.

    Origin is included WITHOUT parsing it back out of the display text. The
    sender has already prefixed the payload as ``"<sender> -> <receiver> :>
    <text>"`` (``tools.py``), and ``sender_name`` there is bound by the tool
    factory and cannot be forged by the model — so hashing the prefixed string
    covers sender, receiver and payload together. That keeps the "no sender
    field on the wire" property of :func:`build_record` intact (design §3.1.2).

    ``reply_back`` is part of the identity because it changes what the message
    ASKS FOR: the same text with and without a required reply are two different
    requests, and collapsing them would swallow the second.
    """
    digest = hashlib.sha256()
    digest.update(b"v1\n")
    digest.update(message.encode("utf-8"))
    digest.update(b"\n")
    digest.update(b"reply_back=1" if reply_back else b"reply_back=0")
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/create in it is itself durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_spool(data_dir: Path) -> Path:
    """Create ``inbox/`` eagerly and make its directory entry durable.

    Called once at startup (design §3.2.1) rather than lazily on first enqueue.
    Creating it up front means the per-record fsync chain in :func:`publish` is
    sufficient on its own, and removes the create-race between two concurrent
    first enqueues.

    NOTE: :func:`publish` also calls this, so it doubles as a LAZY FALLBACK if
    startup initialisation failed (``init_inbox_spool`` logs and continues on
    ``OSError``) or if the directory was removed underneath a running process.
    The design describes creation as eager; this fallback means a message is
    still persisted rather than rejected in those degraded cases. It is
    idempotent, so the normal path pays only an ``mkdir(exist_ok=True)``.
    The eager call is what guarantees the parent-directory fsync happens
    exactly once, off the delivery path.
    """
    inbox = inbox_dir(data_dir)
    existed = inbox.is_dir()
    inbox.mkdir(parents=True, exist_ok=True)
    os.chmod(inbox, _DIR_MODE)
    if not existed:
        # The directory entry for inbox/ lives in data_dir; sync the parent so
        # the spool itself cannot vanish while its contents are durable.
        _fsync_dir(Path(data_dir))
    return inbox


def cleanup_tmp(data_dir: Path) -> int:
    """Remove ``.tmp-*`` leftovers from an interrupted write.

    By construction these were never published, so discarding them cannot lose
    an acknowledged message. Returns the number removed.
    """
    inbox = inbox_dir(data_dir)
    if not inbox.is_dir():
        return 0
    removed = 0
    for path in inbox.iterdir():
        if path.is_file() and path.name.startswith(_TMP_PREFIX):
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover - best effort
                _logger.warning("[INBOX] could not remove stale tmp %s", path)
    return removed


def pending_records(data_dir: Path) -> list[Path]:
    """Published records, sorted by ``seq`` (i.e. arrival order).

    Intake is NAME-GATED: only ``<seq>-<uuid4hex>.json`` is a candidate record.
    Previously any non-``.tmp-`` file was returned, so a file another component
    wrote into the spool was read, failed to parse, and was silently moved to
    ``corrupt/`` with nobody told (observed: a 0644 review-request ``.md``
    in another component's spool). Foreign files are now reported by
    :func:`foreign_entries` and LEFT IN PLACE — moving another component's file
    is how you lose someone else's data. ``corrupt/`` keeps its meaning: a
    correctly-named record whose CONTENTS are unparseable.
    """
    inbox = inbox_dir(data_dir)
    if not inbox.is_dir():
        return []
    # Match ONCE and keep it: re-matching inside the sort key would both
    # re-run the regex per comparison and hand the type checker a fresh
    # Match|None it cannot narrow.
    matched = [
        (match, p)
        for p in inbox.iterdir()
        if p.is_file() and (match := _RECORD_RE.match(p.name)) is not None
    ]
    return [p for _, p in sorted(matched, key=lambda mp: int(mp[0].group(1)))]


def foreign_entries(data_dir: Path) -> list[Path]:
    """Files in the spool that are not delivery records and not ours.

    Returned so the caller can WARN about them once per name per process.
    ``.tmp-*`` (our own in-flight writes) and the ledger are excluded.
    """
    inbox = inbox_dir(data_dir)
    if not inbox.is_dir():
        return []
    return sorted(
        p for p in inbox.iterdir()
        if p.is_file()
        and not p.name.startswith(_TMP_PREFIX)
        and p.name != _LEDGER_NAME
        and p.name != _RECEIVED_NAME
        and not _RECORD_RE.match(p.name)
    )


def _max_seq_in(directory: Path) -> int:
    """Highest ``seq`` present in ``directory``, or -1 if none."""
    if not directory.is_dir():
        return -1
    best = -1
    for path in directory.iterdir():
        m = _RECORD_RE.match(path.name)
        if m is not None:
            best = max(best, int(m.group(1)))
    return best


def next_seq(data_dir: Path) -> int:
    """Compute the next ``seq`` at startup.

    Scans pending, quarantined **and** retired entries (design §3.1.1) so
    neither a quarantined nor a retired record can have its name reused by a
    later arrival.
    """
    return max(
        _max_seq_in(expired_dir(data_dir)),
        _max_seq_in(inbox_dir(data_dir)),
        _max_seq_in(corrupt_dir(data_dir)),
    ) + 1


def build_record(message: str, reply_back: bool) -> dict[str, Any]:
    """Build a v1 record body.

    Note there is deliberately **no** ``sender`` field: the RPyC wire carries
    only ``(message, reply_back)``, and the sender name exists solely inside
    the already-prefixed display text. Recovering it would mean parsing
    ``<sender> -> <recipient> :>`` back out of human-facing text — a bad
    protocol that breaks the moment the format changes (design §3.1.2).
    """
    return {
        "v": RECORD_VERSION,
        "delivery_id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "reply_back": bool(reply_back),
        # Two INDEPENDENT counters (design §3.1). `attempted` drives ORDERING
        # only and is set on every outcome except drain-cancellation, so an
        # interrupted record stops overtaking fresh mail. `failure_attempts`
        # drives RETIREMENT only and is never incremented by a user interrupt.
        # Merging them re-creates the defect: either an interrupt expires a
        # message nobody read, or an interrupted record stays "fresh" forever.
        "attempted": False,
        "failure_attempts": 0,
    }


def publish(data_dir: Path, record: dict[str, Any], seq: int) -> Path:
    """Atomically publish ``record`` and return its final path.

    Sequence (design §3.2) — a torn file is never observable:

    1. write the body to ``.tmp-<uuid>`` (0o600)
    2. ``flush`` + ``fsync`` the file
    3. ``os.replace`` to ``<seq>-<uuid>.json``
    4. ``fsync`` the inbox directory, so the rename itself is durable

    Only after step 4 returns may the caller acknowledge the send.
    """
    inbox = ensure_spool(data_dir)
    token = uuid.uuid4().hex
    tmp_path = inbox / f"{_TMP_PREFIX}{token}"
    final_path = inbox / f"{seq:0{_SEQ_WIDTH}d}-{token}.json"

    payload = json.dumps(record, ensure_ascii=False)
    # 0o600 from creation — never a window where the file is world-readable.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with _suppressed_oserror():
            tmp_path.unlink()
        raise

    os.replace(tmp_path, final_path)
    # Step 4: make the RENAME itself durable, not just the file contents.
    #
    # NOTE FOR MAINTAINERS: removing this line does not fail any test, and
    # that is expected — its effect is only observable across an OS crash or
    # power loss, which the design scopes as BEST-EFFORT (§3.2.1) and which the
    # suite deliberately does not simulate. It is retained because the cost is
    # one fsync per delivery and the failure it prevents (a published record
    # whose directory entry never reached disk) is silent and unrecoverable.
    # Do not delete it on the grounds that "no test covers it".
    _fsync_dir(inbox)
    return final_path


_DELIVERY_ID_RE: Final = re.compile(r"^[0-9a-f]{32}$")
#: sha256 hex, as produced by :func:`content_fingerprint`.
_FINGERPRINT_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def read_record(path: Path) -> dict[str, Any] | None:
    """Parse and VALIDATE a spool record, or ``None`` if it is unusable.

    Fails closed. A record is only delivered when it matches the v1 schema
    exactly; anything else is quarantined by the caller rather than run.

    Rationale (design §3.1.2): ``v`` exists so a future change can *migrate
    rather than misparse*, and ``delivery_id`` is the stable identity that
    makes an at-least-once duplicate diagnosable. Neither promise holds if a
    record with an unknown ``v`` or a missing/garbage ``delivery_id`` is
    delivered anyway and then unlinked on success — the evidence would be gone
    precisely when it is needed. A record written by a NEWER version must be
    left alone, not guessed at.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    # Unknown/absent schema version: refuse rather than misparse.
    if body.get("v") != RECORD_VERSION:
        return None
    if not isinstance(body.get("message"), str):
        return None
    delivery_id = body.get("delivery_id")
    if not isinstance(delivery_id, str) or not _DELIVERY_ID_RE.match(delivery_id):
        return None
    if not isinstance(body.get("ts"), str):
        return None
    if not isinstance(body.get("reply_back"), bool):
        return None
    return body


def quarantine(data_dir: Path, path: Path) -> Path | None:
    """Move an unparseable record into ``inbox/corrupt/``.

    Quarantined records are never retried and never rescanned as pending, so a
    corrupt file cannot wake the drain forever or block later records
    (design §3.3.5). Returns the new path, or ``None`` if the move failed.
    """
    target_dir = corrupt_dir(data_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(target_dir, _DIR_MODE)
        target = target_dir / Path(path).name
        os.replace(path, target)
        _fsync_dir(target_dir)
    except OSError:
        _logger.exception("[INBOX] failed to quarantine %s", path)
        return None
    _logger.warning("[INBOX] quarantined unparseable record %s", target)
    return target


def _verify_ledger_fd(fd: int, path: Path) -> None:
    """Fail CLOSED unless the open fd is our own private regular file.

    Applied on BOTH the append and the load path. On load it is if anything
    more important: a symlinked or planted ledger read at startup injects
    arbitrary ``delivery_id``s into the suppression set, and the sweep then
    unlinks REAL undelivered messages — a silent loss with no trace.

    Scope, as at the append site: this catches accidental corruption and casual
    substitution, NOT a same-UID adversary racing the pathname after the open.
    That rests on the single-owner precondition (design §3.1.1).
    """
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise OSError(f"ledger is not a regular file: {path}")
    if st.st_uid != os.getuid():
        raise OSError(f"ledger not owned by us: {path}")
    if stat.S_IMODE(st.st_mode) != _FILE_MODE:
        raise OSError(f"ledger has unexpected mode {stat.S_IMODE(st.st_mode):o}")
    if st.st_nlink != 1:
        raise OSError(f"ledger is hardlinked (nlink={st.st_nlink}): {path}")


def _open_ledger_for_append(
    data_dir: Path,
    path: Path | None = None,
    line_width: int = _LEDGER_LINE_WIDTH,
) -> tuple[int, bool]:
    """Open a ledger for append. Returns ``(fd, created)``.

    Create detection must be RACE-FREE, and ``fstat`` cannot provide it: it
    says nothing about whether *this* ``open`` created the file. ``O_EXCL`` is
    therefore the only correct probe, and the only one used here — taking the
    other branch would skip the initial-create directory fsync and silently
    reopen the loss window this whole mechanism exists to close.

    An unexpected error is NEVER recovered into a weaker branch: it raises, the
    caller does not unlink, and the delivery replays instead of vanishing.

    ``path``/``line_width`` are parameters so the received-content ledger reuses
    this exact durability sequence instead of growing a second, subtly different
    copy of it. Defaults keep the ``delivered.log`` behaviour unchanged.
    """
    if path is None:
        path = ledger_path(data_dir)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC,
            _FILE_MODE,
        )
        return fd, True
    except FileExistsError:
        pass  # fall through to the existing-file branch

    # O_RDWR, not O_WRONLY: _truncate_torn_tail must pread through this same
    # descriptor. O_APPEND still forces every write to the end.
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        # Fail CLOSED, and BEFORE any write: a rejected file is never appended
        # to and the record stays pending.
        _verify_ledger_fd(fd, path)
        # REPAIR A TORN TAIL BEFORE APPENDING. A previous append may have
        # written a partial id and then failed; appending after it would fuse
        # the fragment and the new id into ONE line (`<partial><full>`), which
        # the parser rejects — so the id would be silently absent from the
        # ledger while its record had already been unlinked. Truncating the
        # incomplete tail first is safe: an unconfirmed append means that
        # delivery's record is still pending (design §3.1.1).
        _truncate_torn_tail(fd, path, line_width)
    except BaseException:
        os.close(fd)
        raise
    return fd, False


def _truncate_torn_tail(
    fd: int, path: Path, line_width: int = _LEDGER_LINE_WIDTH
) -> None:
    """Drop an incomplete final line, durably, before appending after it."""
    size = os.fstat(fd).st_size
    if size == 0:
        return
    # Read through the SAME fd we are about to truncate, via pread. Reopening
    # by pathname would inspect a file that is not necessarily the one this fd
    # refers to, and would re-open the symlink/substitution hole that
    # _verify_ledger_fd just closed on this descriptor. pread also leaves the
    # O_APPEND write offset untouched.
    tail_start = max(0, size - line_width)
    tail = os.pread(fd, size - tail_start, tail_start)
    if tail.endswith(b"\n"):
        return  # intact
    keep = size - len(tail.rsplit(b"\n", 1)[-1])
    os.ftruncate(fd, keep)
    os.fsync(fd)
    _logger.warning(
        "[INBOX] repaired torn tail in dedup ledger %s (%d -> %d bytes); the "
        "affected delivery's record is still pending and will be retried",
        path, size, keep,
    )


def append_delivered(data_dir: Path, delivery_id: str) -> None:
    """Record ``delivery_id`` as delivered, durably, BEFORE the caller unlinks.

    Ordering is the load-bearing property (design §3.1.1)::

        write all bytes -> fsync file -> fsync DIR if we created it -> unlink

    ``fsync`` on a newly created file does NOT make its directory entry
    durable, so the create branch must also fsync the directory; otherwise
    ``create -> write+fsync -> unlink -> crash`` recovers with the record gone
    and the ledger pathname absent — a silent loss.

    Any failure here propagates and the caller must NOT unlink: that costs one
    visible duplicate, never a lost message.

    PRECONDITION (design §3.1.1): one writer per agent. The ledger is appended
    only from ``_deliver_inbox_record`` on the single drain task, and one
    process owns a ``data_dir``. ``O_APPEND`` makes each individual ``write``
    atomic, but a loop resuming after a short write could interleave with
    another appender — so a future second appender MUST take an exclusive lock.
    """
    ensure_spool(data_dir)
    payload = f"{delivery_id}\n".encode("ascii")
    fd, created = _open_ledger_for_append(data_dir)
    try:
        # Write ALL bytes: a short write would leave a torn line.
        view = memoryview(payload)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:  # EINTR — retry, do not lose bytes
                continue
            if written <= 0:
                raise OSError("short write to inbox ledger made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if created:
        # The step whose absence was the silent-loss window.
        _fsync_dir(inbox_dir(data_dir))


def load_delivered(data_dir: Path) -> set[str]:
    """Read the ledger, streaming, dropping a torn final line.

    Discarding an incomplete last line is safe for exactly one reason: a torn
    line means the write loop or the file fsync did not complete, so the caller
    never reached the ``unlink`` and the record is STILL PENDING. The cost is
    at most one replay of a record that still exists — never a loss.
    """
    path = ledger_path(data_dir)
    seen: set[str] = set()
    lines = 0
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return set()
    # Validate BEFORE trusting a single byte, and let failure PROPAGATE. A
    # symlinked or planted ledger injects ids into the suppression set, and the
    # startup sweep then unlinks real, undelivered messages. Returning an empty
    # set here would be the wrong kind of safe: it hides a spool that has been
    # tampered with or corrupted, and the next successful delivery would
    # rebuild a ledger on top of it. Refusing to start is the honest outcome.
    try:
        _verify_ledger_fd(fd, path)
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:  # streaming: never load the whole file
                lines += 1
                if not line.endswith("\n"):
                    break  # torn final line — see _truncate_torn_tail
                candidate = line.strip()
                if _DELIVERY_ID_RE.match(candidate):
                    seen.add(candidate)
    except BaseException:
        with _suppressed_oserror():
            os.close(fd)
        raise
    if lines > LEDGER_CAP:
        _logger.error(
            "[INBOX] dedup ledger has %d lines, over LEDGER_CAP=%d — appends are "
            "outrunning compaction; NOT truncating (that could drop a "
            "suppressing id and re-arm a replay)", lines, LEDGER_CAP,
        )
    return seen


def append_received(data_dir: Path, fingerprint: str) -> None:
    """Record that this MESSAGE has been sent to the agent, durably.

    Written immediately after :func:`publish` puts the record in the inbox,
    because that is the instant the message counts as sent: the sender is
    acknowledged on the fsynced record, not on the turn that later consumes it.
    A fingerprint present here therefore means "this message was queued for the
    agent", which is exactly the condition the no-resend rule is stated over.

    Same durability sequence as :func:`append_delivered`, via the same helper.
    """
    ensure_spool(data_dir)
    payload = f"{fingerprint}\n".encode("ascii")
    fd, created = _open_ledger_for_append(
        data_dir, received_path(data_dir), _RECEIVED_LINE_WIDTH
    )
    try:
        view = memoryview(payload)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:  # EINTR — retry, do not lose bytes
                continue
            if written <= 0:
                raise OSError("short write to received ledger made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if created:
        _fsync_dir(inbox_dir(data_dir))


def load_received(data_dir: Path) -> set[str]:
    """Read the received-content ledger, dropping a torn final line.

    Unlike :func:`load_delivered`, a failure here is NOT fatal. This ledger
    only ever REFUSES a publish; losing it re-admits a duplicate, which is
    visible and recoverable, whereas refusing to start the agent because a
    secondary ledger is unreadable would turn a duplicate-suppression aid into
    an availability risk. A missing file is simply an empty set.
    """
    path = received_path(data_dir)
    seen: set[str] = set()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return set()
    except OSError:
        _logger.warning("[INBOX] could not open received ledger %s", path)
        return set()
    try:
        _verify_ledger_fd(fd, path)
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:  # streaming: never load the whole file
                if not line.endswith("\n"):
                    break  # torn final line
                candidate = line.strip()
                if _FINGERPRINT_RE.match(candidate):
                    seen.add(candidate)
    except BaseException:
        with _suppressed_oserror():
            os.close(fd)
        _logger.warning(
            "[INBOX] received ledger %s is unreadable; treating as empty "
            "(a duplicate may be admitted, nothing is lost)", path,
        )
        return set()
    return seen


def compact_ledger(data_dir: Path, keep: set[str]) -> None:
    """Rewrite the ledger to exactly ``keep``.

    ``keep`` is the intersection of the ledger with the ids of records that are
    STILL PENDING (plus any in-flight delivery) — relevance, never recency. An
    id can only suppress a record that still exists, so this is exact: every id
    that can suppress something is kept and every id that cannot is dropped.
    Position in the file is irrelevant; truncating by recency would evict an
    old suppressing id and re-arm the very replay this prevents.
    """
    path = ledger_path(data_dir)
    if not path.exists() and not keep:
        return
    inbox = ensure_spool(data_dir)
    token = uuid.uuid4().hex
    tmp_path = inbox / f"{_TMP_PREFIX}{token}"
    body = "".join(f"{i}\n" for i in sorted(keep))
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with _suppressed_oserror():
            tmp_path.unlink()
        raise
    os.replace(tmp_path, path)
    _fsync_dir(inbox)


def update_record(path: Path, record: dict[str, Any]) -> None:
    """Rewrite a record in place, atomically, PRESERVING its filename.

    The name carries ``seq``, so rewriting under the same name keeps arrival
    order intact while persisting the counters across restarts.
    """
    path = Path(path)
    tmp_path = path.parent / f"{_TMP_PREFIX}{uuid.uuid4().hex}"
    payload = json.dumps(record, ensure_ascii=False)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with _suppressed_oserror():
            tmp_path.unlink()
        raise
    os.replace(tmp_path, path)


def record_age_seconds(record: dict[str, Any]) -> float:
    """Age from the record's ``ts``. Unparseable timestamps read as age 0.

    Failing to 0 is deliberate: a record whose ``ts`` we cannot trust must not
    be retired on the strength of that same untrusted field.
    """
    try:
        ts = datetime.fromisoformat(str(record.get("ts")))
    except (TypeError, ValueError):
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _format_age(seconds: float) -> str:
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def _sender_of(record: dict[str, Any]) -> str:
    """Best-effort sender, FOR LOGGING ONLY.

    The record has no ``sender`` field by design (§3.1.2) — the wire carries
    only ``(message, reply_back)``. Parsing the display prefix would be a bad
    protocol, but it is fine in a diagnostic string, and degrades to ``?``.
    """
    message = record.get("message")
    if isinstance(message, str):
        for line in message.splitlines():
            marker = line.find(" -> ")
            if marker > 0 and " :> " in line:
                return line[:marker].strip() or "?"
    return "?"


def retire(data_dir: Path, path: Path, record: dict[str, Any], reason: str) -> Path | None:
    """Move a record to ``expired/`` and say so LOUDLY.

    Never a silent drop: the record is preserved on disk and the WARNING
    carries delivery_id, seq, sender, age and the failure count, so a message
    that was never delivered can always be found and explained afterwards.
    """
    target_dir = expired_dir(data_dir)
    name = Path(path).name
    match = _RECORD_RE.match(name)
    seq = match.group(1).lstrip("0") or "0" if match else "?"
    age = _format_age(record_age_seconds(record))
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(target_dir, _DIR_MODE)
        target = target_dir / name
        os.replace(path, target)
        # BOTH directories: the rename adds an entry to expired/ AND removes
        # one from inbox/. Syncing only the destination leaves the source
        # removal non-durable, so after a power loss the record can resurrect
        # as pending and be DELIVERED — defeating the bounded-retirement
        # guarantee this function exists to provide.
        _fsync_dir(target_dir)
        _fsync_dir(inbox_dir(data_dir))
    except OSError:
        _logger.exception("[INBOX] failed to retire %s", path)
        return None
    _logger.warning(
        "[INBOX] RETIRED delivery_id=%s seq=%s sender=%s age=%s "
        "failure_attempts=%s attempted=%s reason=%s — moved to expired/, "
        "NOT delivered",
        record.get("delivery_id"), seq, _sender_of(record), age,
        record.get("failure_attempts", 0), record.get("attempted", False),
        reason,
    )
    return target


class _suppressed_oserror:
    """Tiny context manager: ignore OSError during best-effort cleanup."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)
