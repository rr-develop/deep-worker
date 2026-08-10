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

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_logger = logging.getLogger(__name__)

_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600

#: Schema version of a spool record. Bump when the body shape changes so a
#: future reader can migrate rather than misparse.
RECORD_VERSION: Final = 1

_SEQ_WIDTH: Final = 12
_TMP_PREFIX: Final = ".tmp-"
_CORRUPT_DIRNAME: Final = "corrupt"

#: ``<seq>-<uuid4hex>.json`` — the published form.
_RECORD_RE: Final = re.compile(r"^(\d{%d})-([0-9a-f]{32})\.json$" % _SEQ_WIDTH)


def inbox_dir(data_dir: Path) -> Path:
    """Spool directory for ``data_dir``."""
    return Path(data_dir) / "inbox"


def corrupt_dir(data_dir: Path) -> Path:
    """Quarantine directory for unparseable records."""
    return inbox_dir(data_dir) / _CORRUPT_DIRNAME


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

    ``.tmp-*`` files and the ``corrupt/`` subdirectory are excluded.
    """
    inbox = inbox_dir(data_dir)
    if not inbox.is_dir():
        return []
    entries = [
        p for p in inbox.iterdir()
        if p.is_file() and not p.name.startswith(_TMP_PREFIX)
    ]
    # Sort by parsed seq where possible; unparseable names sort last but are
    # still returned so the drain can quarantine them.
    def _key(p: Path) -> tuple[int, int, str]:
        m = _RECORD_RE.match(p.name)
        if m is None:
            return (1, 0, p.name)
        return (0, int(m.group(1)), p.name)

    return sorted(entries, key=_key)


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

    Scans pending **and** quarantined entries (design §3.1.1) so a quarantined
    record can never have its name reused by a later arrival.
    """
    return max(
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


class _suppressed_oserror:
    """Tiny context manager: ignore OSError during best-effort cleanup."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)
