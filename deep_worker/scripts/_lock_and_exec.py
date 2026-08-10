#!/usr/bin/env python3
"""Open a lock file safely, hold an exclusive lock on it, and exec the caller.

Helper for ``uv-run.sh``. Not useful on its own.

Why this exists
---------------
``uv-run.sh`` must serialize per project root, and the lock file it uses lives
in a directory under ``$TMPDIR``/``$XDG_RUNTIME_DIR``. Doing this purely in bash
is unsafe:

* ``exec 9>"$path"`` follows symlinks, so validating ``$path`` and *then*
  redirecting to it is a TOCTOU — the path can be swapped for a symlink between
  the two steps, and the shell will happily write through it.
* Even validating with ``stat`` and re-opening by name re-introduces the race,
  because the second open resolves the name again.

The fix is to do the open, the validation and the lock on **one** file
descriptor:

1. ``os.open`` with ``O_NOFOLLOW`` — refuses outright if the final component is
   a symlink, atomically.
2. ``os.fstat`` on the resulting fd — validates the object we actually opened
   (regular file, our uid, mode 0600), not a name we looked at earlier.
3. ``flock`` on that same fd.
4. ``os.execv`` the original script, which inherits the still-locked fd.

``flock`` is held by the open file description, so it survives ``exec`` and is
released automatically by the kernel when the process dies — including on
SIGKILL, so a crashed run cannot wedge the next one.

Usage:
    _lock_and_exec.py <lock-file> <expected-uid> <script> [args...]
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import sys

#: Must match the fd number ``uv-run.sh`` expects to inherit.
LOCK_FD = 9


def _fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    sys.stderr.write(f"ERROR (uv-run.sh): {message}\n")
    raise SystemExit(1)


def _verify_held(lock_path: str, expected_uid: int) -> int:
    """Exit 0 iff inherited fd 9 IS the validated, flocked lock for *lock_path*.

    ``uv-run.sh`` must not decide "am I already holding the lock?" from an
    environment variable: the payload inherits our environment, so a nested
    wrapper (for a different project!) would think it already holds a lock, and
    a caller could set the marker by hand to skip locking entirely.

    Identity is therefore established against the kernel:

    * fd 9 must exist and be a regular file owned by us with mode 0600;
    * it must be the SAME INODE as ``lock_path`` -- so a lock held for another
      project root cannot be mistaken for this one;
    * a non-blocking ``LOCK_EX`` on **fd 9 itself** must succeed.

    The last point is the subtle one. Probing through a *separate* ``open``
    only proves "some open file description holds this inode" -- it cannot tell
    "fd 9 holds it" from "another wrapper holds it while fd 9 holds nothing".
    A caller could open the correct lock file on fd 9 without locking it while a
    real wrapper owns the lock; the separate probe would get EAGAIN and we would
    wrongly conclude we may proceed, running concurrently with that wrapper.

    Locking fd 9 itself resolves all three cases correctly, because ``flock`` is
    per open file description and re-locking one's own description is a no-op:

    ==========================  =========================  ==================
    state of fd 9               flock(fd9, EX|NB)          verdict
    ==========================  =========================  ==================
    already owns the lock       succeeds (no-op re-lock)   held -> proceed
    open but unlocked, free     succeeds (acquires it)     safe to proceed
    open but unlocked, taken    EAGAIN                     not ours -> re-exec
    ==========================  =========================  ==================
    """
    try:
        info = os.fstat(LOCK_FD)
    except OSError:
        return 1  # no fd 9 at all -> not held

    if not stat.S_ISREG(info.st_mode):
        return 1
    if info.st_uid != expected_uid:
        return 1
    if stat.S_IMODE(info.st_mode) != 0o600:
        return 1

    try:
        on_disk = os.stat(lock_path)
    except OSError:
        return 1
    if (on_disk.st_dev, on_disk.st_ino) != (info.st_dev, info.st_ino):
        # fd 9 is some other file -- e.g. another project's lock inherited
        # through a nested invocation.
        return 1

    # Operate on fd 9 ITSELF -- see the table above. Never release afterwards:
    # if this succeeded, fd 9 now holds the lock and must keep holding it for
    # the rest of the run.
    try:
        fcntl.flock(LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            # Someone else's open file description owns it. We do NOT hold the
            # lock, so the caller must go through the helper and block.
            return 1
        # EINTR or anything else is an unknown state, not evidence either way.
        # Treating it as "held" would let the run proceed unlocked.
        return 1
    return 0


def main(argv: list[str]) -> "NoReturn":  # type: ignore[valid-type]
    if len(argv) >= 4 and argv[1] == "--verify-held":
        raise SystemExit(_verify_held(argv[2], int(argv[3])))

    if len(argv) < 4:
        _fail("usage: _lock_and_exec.py <lock-file> <uid> <script> [args...]")

    lock_path, expected_uid_raw, script = argv[1], argv[2], argv[3]
    script_args = argv[4:]

    try:
        expected_uid = int(expected_uid_raw)
    except ValueError:
        _fail(f"invalid uid argument: {expected_uid_raw!r}")

    # O_NOFOLLOW makes this fail with ELOOP if the final component is a symlink,
    # atomically -- there is no window between "is it a symlink?" and "open it".
    try:
        fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        _fail(f"cannot safely open lock file {lock_path}: {exc}")

    # Validate the OPENED OBJECT, not the path. Anything surprising is a hard
    # stop: we do not repair permissions or ownership, because a lock file we
    # did not create the way we expect means something is wrong that a human
    # should look at.
    try:
        info = os.fstat(fd)
    except OSError as exc:
        _fail(f"cannot stat opened lock file {lock_path}: {exc}")

    if not stat.S_ISREG(info.st_mode):
        _fail(f"lock path is not a regular file: {lock_path}")
    if info.st_uid != expected_uid:
        _fail(
            f"lock file is owned by uid {info.st_uid}, not {expected_uid}: "
            f"{lock_path}"
        )
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o600:
        # Ownership alone is not enough: a 0666 lock file lets any local user
        # hold or corrupt the lock, which defeats the serialization that the
        # ".venv is mine to delete" claim depends on.
        _fail(f"lock file mode is {mode:04o}, expected 0600: {lock_path}")

    # Blocks in the kernel until the current holder releases -- event-driven,
    # no polling.
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        _fail(f"failed to acquire lock on {lock_path}: {exc}")

    # Hand the locked fd to the script at the well-known number. os.dup2 clears
    # FD_CLOEXEC on the duplicate, so it survives the exec below.
    if fd != LOCK_FD:
        try:
            os.dup2(fd, LOCK_FD)
        except OSError as exc:
            _fail(f"cannot place lock fd at {LOCK_FD}: {exc}")
        os.close(fd)

    try:
        os.execv("/usr/bin/env", ["env", "bash", script, *script_args])
    except OSError as exc:
        _fail(f"cannot exec {script}: {exc}")


if __name__ == "__main__":
    main(sys.argv)
