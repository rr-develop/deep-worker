"""Regression: agent log files must never exceed a hard on-disk size cap.

Incident (observed 2026-05-25, recurrence of the get_history log storm): a
daemon running a pre-fix build wrote ~100 GB/day into ``agent.err.log`` and
filled the host disk to 100%. Nothing bounded the log size, and external
logrotate is too easy to forget. The fix enforces the cap IN-PROCESS:
``deep_worker_cli.process.log_rotation.RotatingLogWriter`` rotates the active
file at a threshold and keeps a fixed number of backups, so

    on-disk total = active (<= THRESHOLD) + backups * (<= THRESHOLD)

is bounded with NO system config.

Contract under test:
  * the on-disk total (active + .1) NEVER exceeds 2 * threshold, at every point;
  * rotation actually happens (a .1 appears) and only BACKUP_COUNT backups are
    kept (no .2 with backups=1);
  * data within the retained window is intact (no corruption/interleaving);
  * a pre-existing oversized active file / stray backups converge to budget at
    startup, before any new byte is written;
  * the pipe-fed path (start_pipe_writer) honours the cap and exits cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

from deep_worker_cli.process.log_rotation import (
    RotatingLogWriter,
    start_pipe_writer,
)


def _ondisk_total(path: str, backups: int) -> int:
    """Sum the active file plus every backup segment currently on disk."""
    total = 0
    for p in [path] + [f"{path}.{i}" for i in range(1, backups + 2)]:
        try:
            total += os.stat(p).st_size
        except OSError:
            pass
    return total


def test_rotation_never_exceeds_budget(tmp_path: Path):
    """After EVERY write, active + .1 must stay <= 2 * threshold."""
    threshold = 1000
    path = str(tmp_path / "x.log")
    w = RotatingLogWriter(path, threshold=threshold, backups=1)

    rotated = False
    chunk = b"A" * 137  # non-divisor of threshold to exercise boundaries
    for _ in range(500):
        w.write(chunk)
        total = _ondisk_total(path, backups=1)
        assert total <= 2 * threshold, f"on-disk total {total} exceeded budget {2 * threshold}"
        if os.path.exists(f"{path}.1"):
            rotated = True
        # Only ONE backup is ever kept.
        assert not os.path.exists(f"{path}.2"), "a second backup (.2) must never exist with backups=1"
    w.close()

    assert rotated, "rotation never happened — .1 was never created"


def test_data_integrity_within_window(tmp_path: Path):
    """The retained window (.1 then active) is a contiguous suffix of the stream."""
    threshold = 500
    path = str(tmp_path / "y.log")
    w = RotatingLogWriter(path, threshold=threshold, backups=1)

    expected = bytearray()
    lines = []
    for i in range(2000):
        line = b"LINE_%05d\n" % i
        lines.append(line)
        expected += line
        w.write(line)
    w.close()

    kept = bytearray()
    if os.path.exists(f"{path}.1"):
        with open(f"{path}.1", "rb") as fh:
            kept += fh.read()
    with open(path, "rb") as fh:
        kept += fh.read()

    # Everything we kept must be a contiguous suffix of the full intended stream
    # (no bytes from outside the window, no interleaving/corruption).
    assert bytes(expected).endswith(bytes(kept)), "retained data is not a contiguous suffix of the stream"
    # And the most recent line must always survive.
    assert kept.endswith(lines[-1]), "the newest line was lost"


def test_startup_recovery_oversized_active(tmp_path: Path):
    """A pre-existing oversized active file + stray backups converge at startup.

    An oversized leftover (e.g. a huge file from a pre-fix build) CANNOT be kept
    within budget — rotating it would just move the over-budget bytes into .1 —
    so it is discarded outright. The hard cap is absolute.
    """
    threshold = 1000
    path = str(tmp_path / "z.log")

    # Leftover from a previous (buggy) run: huge active file + a stray .3.
    with open(path, "wb") as fh:
        fh.write(b"B" * 2500)
    with open(f"{path}.3", "wb") as fh:
        fh.write(b"stale")

    w = RotatingLogWriter(path, threshold=threshold, backups=1)
    w.write(b"new data\n")
    w.close()

    assert not os.path.exists(f"{path}.3"), "stray backup beyond the scheme must be pruned at startup"
    # The oversized active file is discarded, not rotated into .1 (keeping a
    # >threshold .1 would blow the budget). So .1 stays absent here.
    assert not os.path.exists(f"{path}.1"), "an oversized leftover must be discarded, not preserved in .1"
    assert _ondisk_total(path, backups=1) <= 2 * threshold, "did not converge to budget after recovery"
    # The fresh active file holds only the new data.
    with open(path, "rb") as fh:
        assert fh.read() == b"new data\n"


def test_startup_recovery_with_existing_backup(tmp_path: Path):
    """Oversized active AND an existing .1 still converge to <= budget."""
    threshold = 1000
    path = str(tmp_path / "w.log")
    with open(path, "wb") as fh:
        fh.write(b"C" * 1800)
    with open(f"{path}.1", "wb") as fh:
        fh.write(b"D" * 900)

    w = RotatingLogWriter(path, threshold=threshold, backups=1)
    w.write(b"x")
    w.close()

    assert os.path.exists(f"{path}.1"), ".1 should exist (the rotated active)"
    assert not os.path.exists(f"{path}.2"), "no .2 should be created with backups=1"
    assert _ondisk_total(path, backups=1) <= 2 * threshold


def test_backups_zero_truncates(tmp_path: Path):
    """With backups=0 no .1 is ever created and active stays bounded."""
    threshold = 1000
    path = str(tmp_path / "t.log")
    w = RotatingLogWriter(path, threshold=threshold, backups=0)

    chunk = b"E" * 300
    for _ in range(50):
        w.write(chunk)
        assert not os.path.exists(f"{path}.1"), "no backup must exist with backups=0"
        assert os.stat(path).st_size <= threshold + len(chunk), "active overshot beyond one chunk"
    w.close()


def test_start_pipe_writer_real_pipe(tmp_path: Path):
    """End-to-end through a real pipe (no fork): cap holds and thread exits."""
    threshold = 1000
    path = str(tmp_path / "pipe.log")
    write_fd, thread = start_pipe_writer(
        path, threshold=threshold, backups=1, chunk_size=256
    )

    payload = b"F" * 5000
    os.write(write_fd, payload)
    os.close(write_fd)  # EOF -> thread drains the rest and exits

    thread.join(timeout=5)
    assert not thread.is_alive(), "writer thread did not exit after EOF"

    assert os.path.exists(f"{path}.1"), "rotation should have produced a .1"
    assert _ondisk_total(path, backups=1) <= 2 * threshold, "pipe path exceeded the budget"

    # Data integrity across the pipe round-trip: kept window is a suffix of payload.
    kept = bytearray()
    with open(f"{path}.1", "rb") as fh:
        kept += fh.read()
    with open(path, "rb") as fh:
        kept += fh.read()
    assert payload.endswith(bytes(kept)), "data corrupted across the pipe round-trip"
