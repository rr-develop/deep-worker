"""One-time capability token proving a stop request came from the agent itself.

``stop_agent(..., self_requested=True)`` waives the "actively running" guard.
That guard is the only thing standing between an external caller and a turn it
knows nothing about, so the waiver must not be reachable by simply *asserting*
it on a command line: ``deep-worker --restart <busy-agent> --self-requested``
must NOT be enough. Hiding the flag from ``--help`` is obscurity, not
authorization.

The sanction therefore travels as a capability, not as a claim:

  1. ``restart_self`` (running INSIDE the target agent process) mints a random
     token and writes it to ``.self_restart_token.json`` in that agent's own
     data directory, mode 0600. Only something already executing as the agent —
     i.e. with write access to the agent's private tree — can create it.
  2. The token is handed to the driver on the command line.
  3. ``stop_agent`` consumes it: the file is deleted FIRST, then compared. A
     token is valid once and only once.

What this buys, concretely: an external ``--self-requested`` with a guessed or
replayed token is refused, because no matching unconsumed token exists on disk.
What it does not claim to buy: defence against an attacker who can already
write to the agent's private directory as the agent's uid — such an attacker
can restart the agent by a dozen other routes, so it is outside this boundary.

Consume-once matters for a specific failure: a token left on disk after a
restart would let ANY later external ``--self-requested`` reuse it. Deleting
before comparing also means a crash mid-verification fails closed.

See docs/DESIGN_RESTART_SELF_FALSE_DENIAL.md.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_FILENAME = ".self_restart_token.json"  # noqa: S105 - a filename, not a secret

# 32 bytes of urandom, hex-encoded. Not guessable within the lifetime of the
# token (which is one restart).
_TOKEN_NBYTES = 32


def mint_self_restart_token(data_dir: Path) -> str:
    """Create and persist a fresh one-time self-restart token.

    Called from inside the agent process by ``restart_self``. Overwrites any
    previous unconsumed token — a newer self-restart supersedes an older one
    that never got used, and leaving stale tokens around would only widen the
    replay window.

    Returns the token string to hand to the driver.
    """
    from deep_worker_cli.storage_paths import ensure_private_file

    data_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(_TOKEN_NBYTES)
    token_path = data_dir / TOKEN_FILENAME
    payload = json.dumps({"token": token}).encode("utf-8")
    # O_CREAT|O_EXCL|O_NOFOLLOW 0600: the secret must not exist even
    # briefly as a umask-created 0666 file. A leftover token from a
    # previous unused restart is unlinked first, then created exclusive.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)

    def _create() -> int:
        return os.open(token_path, flags, 0o600)

    try:
        fd = _create()
    except FileExistsError:
        token_path.unlink()
        fd = _create()
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, payload)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            token_path.unlink()
        except OSError:
            pass
        raise
    os.close(fd)
    ensure_private_file(token_path)
    return token


def consume_self_restart_token(data_dir: Path, presented: str | None) -> bool:
    """Verify ``presented`` against the on-disk token and consume it.

    Returns True only if a token file existed, was readable, was
    successfully DELETED, and matched. Deletion happens before the
    comparison and is a precondition of the result: a token that could not
    be removed is refused, because authorizing it would leave a replayable
    capability on disk.

    A missing file, an unreadable/corrupt file, a failed unlink, or a
    mismatch all return False — fail closed.
    """
    if not presented:
        return False

    token_path = data_dir / TOKEN_FILENAME
    try:
        raw = token_path.read_text()
    except (OSError, ValueError):
        return False

    # Successful deletion is a PRECONDITION of authorization, not a
    # best-effort cleanup. If the file survives, the same token would still
    # authorize the next caller — "valid exactly once" would be a false
    # claim and the capability would be replayable. So a failed unlink
    # denies the waiver even when the token itself matches: fail closed.
    try:
        token_path.unlink()
    except OSError as exc:
        logger.error(
            f"Refusing self-restart: could not consume the capability token "
            f"at {token_path} ({exc}). Authorizing without consuming it "
            f"would leave a replayable token on disk."
        )
        return False

    try:
        stored = json.loads(raw).get("token")
    except (ValueError, AttributeError):
        return False

    if not isinstance(stored, str) or not stored:
        return False

    # Constant-time compare: the token is a secret and the caller controls the
    # presented value, so avoid leaking a prefix match through timing.
    return secrets.compare_digest(stored, presented)
