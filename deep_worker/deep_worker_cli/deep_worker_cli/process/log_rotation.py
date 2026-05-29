"""In-process hard cap + rotation for the agent daemon's log files.

Why this exists
---------------
Agent daemons redirect stdout/stderr to the agent's log files
(``{name}.{out,err}.log`` under the agent's data directory)
via ``os.dup2`` in ``deep_worker_cli.cli.agent_command.daemonize``. Nothing
bounded the size of those files, so a buggy hot path could fill the disk:
an older build (predating the get_history log-storm fix) wrote
~100 GB/day into a single ``.err.log`` and took the host to 100% disk
(incident observed 2026-05-25; recurrence of a related internal incident
analysis (not included here)).

External logrotate was rejected: the cap must be enforced BY THE PROCESS
ITSELF, with zero system config, so it holds even if nobody set anything up.

How it works
------------
``daemonize`` points fd 1/2 at the write end of an ``os.pipe()`` instead of
straight at the log file. A daemon writer thread drains the read end and
appends through a ``RotatingLogWriter`` that enforces a hard per-file cap. This
captures ``print``, the ``logging`` module, AND C-level writes to fd 1/2 alike.

Hard invariant (no compression, simple rotation)
------------------------------------------------
On-disk total per stream = active file (<= THRESHOLD) + ``BACKUP_COUNT`` rotated
segments (each <= THRESHOLD). With BACKUP_COUNT=1 that is two segments.

Cap proof. ``write`` rotates BEFORE appending whenever the chunk would push the
active file past THRESHOLD (rotate-before-overflow). A single chunk is at most
``PIPE_READ_CHUNK_BYTES`` (``os.read`` never returns more than requested), which
is far below THRESHOLD. So for every real chunk the active file stays
``<= THRESHOLD``:

    active_max = THRESHOLD
    .1_max     = THRESHOLD          (a former active file, frozen)
    SUM_max    = 2 * THRESHOLD

With THRESHOLD = 512 MiB - 2*CHUNK, SUM_max = 1 GiB - 4*CHUNK < 1 GiB, so
(active + .1) is ALWAYS strictly below the 1 GiB ceiling. (The CHUNK slack in
THRESHOLD is not needed for the bound above — it is kept only so that even a
pathological single chunk larger than a pipe read could not push a fresh active
file over a full 512 MiB.) The only way a single file exceeds THRESHOLD is one
``write`` whose chunk alone is ``> THRESHOLD``; such a chunk lands by itself in
a freshly-rotated active file and cannot come from the 64 KiB pipe path.
"""

from __future__ import annotations

import os
import threading

# One pipe-buffer fill. ``os.read(fd, PIPE_READ_CHUNK_BYTES)`` is the largest a
# single ``write`` chunk can be, which bounds the rotation overshoot.
PIPE_READ_CHUNK_BYTES = 65536  # 64 KiB

# The hard ceiling the user demands: per stream, the sum of all on-disk log
# segments must NEVER exceed this.
HARD_BUDGET_BYTES = 1024 * 1024 * 1024  # 1 GiB

# Active-file rotation threshold. 512 MiB minus two read-chunks of slack so the
# worst-case (active + one backup) overshoot stays strictly under HARD_BUDGET
# (see the overshoot proof in the module docstring). Keep the ``- 2*CHUNK`` term
# explicit so the relationship to the proof is obvious.
ROTATE_THRESHOLD_BYTES = 512 * 1024 * 1024 - 2 * PIPE_READ_CHUNK_BYTES

# Number of rotated ``.N`` segments kept. 1 => active + .1, total <= 1 GiB.
BACKUP_COUNT = 1

# How many extra ``.N`` segments to scan-and-prune at startup, beyond the kept
# range, to clean up sparse leftovers from prior runs/scheme changes.
_STARTUP_PRUNE_SCAN = 64


class RotatingLogWriter:
    """Append-only writer with a hard size cap via simple file rotation.

    Owns one binary, unbuffered, append-mode file object. Tracks the active
    file's size internally (seeded once via ``os.stat`` at open), so no per-write
    ``stat`` is needed and the counter exactly matches on-disk bytes. When the
    active file reaches ``threshold``, it is rotated: the oldest backup is
    deleted, the rest shift down, the active file becomes ``.1``, and a fresh
    empty active file is opened.

    Pure file I/O — no pipes, threads, or fork. Unit-testable directly.
    """

    def __init__(
        self,
        path: str,
        threshold: int = ROTATE_THRESHOLD_BYTES,
        backups: int = BACKUP_COUNT,
    ) -> None:
        self._path = path
        self._threshold = max(1, int(threshold))
        self._backups = max(0, int(backups))
        self._fobj = None  # type: ignore[assignment]  # binary append file object
        self._size = 0  # bytes currently in the ACTIVE file
        self._open_and_recover()

    def _backup_path(self, n: int) -> str:
        # n >= 1, e.g. ".../agent.err.log.1"
        return f"{self._path}.{n}"

    def _open_and_recover(self) -> None:
        """Open the active file, converging any pre-existing logs to budget.

        A daemon may start on top of logs left by a previous run — possibly a
        huge active file (e.g. the 56 GB leftover from an old build) or stray
        backups beyond what the current scheme keeps. Prune the extras and
        rotate an oversized active file BEFORE writing a single new byte.
        """
        # 1. Prune backups beyond what we keep (.{backups+1}, .{backups+2}, ...).
        #    Scan a bounded range and delete every match — do NOT stop at the
        #    first gap, since a previous run may have left a sparse set (e.g.
        #    only a stray .3 with no .2).
        n = self._backups + 1
        while n <= self._backups + _STARTUP_PRUNE_SCAN:
            p = self._backup_path(n)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
            n += 1

        # 2. Discard any KEPT backup that is itself oversized. A backup left by
        #    a prior (buggy) run could exceed the threshold; rotating around it
        #    would keep it on disk and blow the budget. The hard cap wins over
        #    retaining history, so drop it.
        for i in range(1, self._backups + 1):
            bp = self._backup_path(i)
            try:
                if os.stat(bp).st_size > self._threshold:
                    os.remove(bp)
            except OSError:
                pass

        # 3. Seed size from the existing active file (the only os.stat we do).
        try:
            existing = os.stat(self._path).st_size
        except OSError:
            existing = 0

        # 4. An oversized leftover active file (e.g. a 56 GB file from an old
        #    build) CANNOT be retained within budget — rotating it would just
        #    move the over-budget bytes into ``.1``. The hard cap is absolute,
        #    so discard it outright rather than preserve it.
        if existing > self._threshold:
            try:
                os.remove(self._path)
            except OSError:
                pass
            existing = 0

        # 5. Open the active file for unbuffered binary append; seed the counter.
        self._fobj = open(self._path, "ab", buffering=0)
        self._size = existing

    def _rotate(self) -> None:
        """Rotate the active file into ``.1``, dropping the oldest backup.

        Leaves a fresh, empty active file open and ``_size`` reset to 0.
        """
        if self._backups <= 0:
            # No backups kept: discard the active file entirely.
            if self._fobj is not None:
                try:
                    self._fobj.close()
                except OSError:
                    pass
                self._fobj = None
            try:
                os.remove(self._path)
            except OSError:
                pass
            self._fobj = open(self._path, "ab", buffering=0)
            self._size = 0
            return

        # Delete the oldest kept backup, then shift the rest down by one.
        oldest = self._backup_path(self._backups)
        if os.path.exists(oldest):
            try:
                os.remove(oldest)
            except OSError:
                pass
        for i in range(self._backups - 1, 0, -1):
            src = self._backup_path(i)
            dst = self._backup_path(i + 1)
            if os.path.exists(src):
                try:
                    os.replace(src, dst)
                except OSError:
                    pass

        # Close the active file before renaming so we reopen a clean inode.
        if self._fobj is not None:
            try:
                self._fobj.close()
            except OSError:
                pass
            self._fobj = None
        if os.path.exists(self._path):
            try:
                os.replace(self._path, self._backup_path(1))
            except OSError:
                pass

        self._fobj = open(self._path, "ab", buffering=0)
        self._size = 0

    def write(self, chunk: bytes) -> None:
        """Append ``chunk``, rotating first if it would overflow the threshold."""
        if not chunk:
            return
        # Rotate-before-overflow: if adding this chunk would push the active file
        # past the threshold, rotate first. For chunks no larger than the
        # threshold (always true for pipe reads, since chunk <= PIPE_READ_CHUNK
        # << THRESHOLD) the active file therefore never exceeds the threshold,
        # so on-disk total = active + .1 <= 2 * threshold, exactly. Only the
        # pathological case of a single chunk > threshold can overshoot, and it
        # lands alone in a freshly-rotated active file.
        if self._size and self._size + len(chunk) > self._threshold:
            self._rotate()
        if self._fobj is None:  # defensive; _rotate/_open always leave it open
            self._fobj = open(self._path, "ab", buffering=0)
            self._size = 0
        self._fobj.write(chunk)  # buffering=0 -> single unbuffered os.write
        self._size += len(chunk)

    def flush(self) -> None:
        if self._fobj is not None:
            try:
                self._fobj.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._fobj is not None:
            try:
                self._fobj.flush()
            except OSError:
                pass
            try:
                self._fobj.close()
            except OSError:
                pass
            self._fobj = None


def start_pipe_writer(
    path: str,
    threshold: int = ROTATE_THRESHOLD_BYTES,
    backups: int = BACKUP_COUNT,
    chunk_size: int = PIPE_READ_CHUNK_BYTES,
    thread_name: str | None = None,
) -> tuple[int, threading.Thread]:
    """Create a pipe + draining writer thread for one log stream.

    Returns ``(write_fd, thread)``. The caller ``dup2``'s ``write_fd`` onto
    stdout/stderr and then closes its own copy; fd 1/2 keep a dup of the write
    end alive for the life of the process, so the thread sees EOF (read returns
    ``b""``) only at process teardown. The read end is owned and closed by the
    thread.

    The thread is a daemon thread and its drain loop NEVER raises: on a write
    error it discards the chunk but keeps reading, so the producer's ``write``
    can never block forever on a full pipe buffer.
    """
    read_fd, write_fd = os.pipe()
    writer = RotatingLogWriter(path, threshold=threshold, backups=backups)

    def _drain() -> None:
        try:
            while True:
                try:
                    data = os.read(read_fd, chunk_size)
                except InterruptedError:  # EINTR -> retry
                    continue
                except OSError:  # read end broke; producer is gone too
                    break
                if not data:  # EOF: all write ends closed (process teardown)
                    break
                try:
                    writer.write(data)
                except Exception:
                    # Disk full / FS error / anything: drop this chunk but KEEP
                    # DRAINING. Losing log bytes is acceptable; blocking the
                    # agent's write() forever on a full pipe is not.
                    pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
            try:
                os.close(read_fd)
            except OSError:
                pass

    thread = threading.Thread(
        target=_drain,
        name=thread_name or f"logwriter:{os.path.basename(path)}",
        daemon=True,  # never block interpreter exit
    )
    thread.start()
    return write_fd, thread
