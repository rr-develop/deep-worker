"""Turn-text accumulation must be O(1) per notification, not O(n).

ROOT CAUSE of the burst defect. ``_append_bounded_text`` is called once per
``agent_message_chunk``. It used to re-derive the byte total with
``sum(len(c) for c in acc)`` on every call, so a turn's accumulation was
quadratic. Measured on the real function with 200-byte chunks:

    accumulated chunks   per-notification cost
    0                    2.4 us
    1000                 40.4 us
    4000                 153.8 us
    8000                 300.7 us      (125x the empty-buffer cost)

and that ``sum()`` was 97-101% of the call while the ``append`` itself cost
0.15 us. The reader routes an arriving notification for ~3.4 us, so from ~1000
accumulated chunks on the consumer is an order of magnitude slower than arrival
and the 1000-slot sink fills — the overflow is a *consequence* of this, which is
why raising the capacity only relocated the cliff.

The same arithmetic is the "a trivial tool call hangs" symptom: the finalization
drain holds the event loop with no await between notifications, so an unrelated
coroutine saw 1441 ms of latency at 8000 notifications and 3.49 s of solid CPU
at the 10000-chunk cap.

These tests assert the SHAPE of the cost curve, not an absolute duration, so
they do not depend on how fast the host is. They compare a cost ratio against an
input ratio, with a tolerance wide enough for scheduler noise but far below the
quadratic signature (which is ~4x per doubling, measured).
"""

from __future__ import annotations

import time

import pytest

from deep_worker.providers.acp.chat_model import (
    _BoundedTextAcc,
    _append_bounded_text,
    _text_acc_total,
)
from deep_worker.providers.acp.custom_types import (
    QUEUED_TEXT_MAX_BYTES,
    QUEUED_TEXT_MAX_CHUNKS,
)

CHUNK = "x" * 200


def _cost_per_append(prefilled: int, samples: int = 200) -> float:
    """Seconds per append against a buffer that already holds ``prefilled``."""
    acc = _BoundedTextAcc([CHUNK] * prefilled)
    best = None
    for _ in range(3):  # best-of-3: resist a scheduler hiccup, not a trend
        start = time.perf_counter()
        for _ in range(samples):
            _append_bounded_text(acc, CHUNK)
        elapsed = (time.perf_counter() - start) / samples
        best = elapsed if best is None else min(best, elapsed)
        del acc[-samples:]
        acc._total -= samples * len(CHUNK)
    assert best is not None
    return best


class TestPerNotificationCostIsFlat:
    """FAILS on pre-fix code: the cost grew ~linearly with the buffer."""

    def test_cost_does_not_grow_with_accumulated_volume(self) -> None:
        small = _cost_per_append(500)
        large = _cost_per_append(8000)
        # Pre-fix this ratio tracked 8000/500 = 16x (measured 21.1 -> 300.7 us,
        # i.e. 14x). A flat implementation sits near 1x; allow 4x for noise on a
        # loaded host, which still separates the two regimes by a wide margin.
        assert large < small * 4, (
            f"per-notification cost grew with volume: {small * 1e6:.1f}us at 500 "
            f"chunks -> {large * 1e6:.1f}us at 8000 ({large / small:.1f}x). "
            "Accumulation is not O(1) per notification."
        )

    def test_whole_turn_cost_is_not_quadratic(self) -> None:
        """Doubling the turn must roughly double the work, not quadruple it."""

        def whole_turn(n: int) -> float:
            best = None
            for _ in range(3):
                acc = _BoundedTextAcc()
                start = time.perf_counter()
                for _ in range(n):
                    _append_bounded_text(acc, CHUNK)
                elapsed = time.perf_counter() - start
                best = elapsed if best is None else min(best, elapsed)
            assert best is not None
            return best

        base = whole_turn(2000)
        doubled = whole_turn(4000)
        # Measured pre-fix: 4.08x for a 2x input. Linear is ~2x; 3x is the
        # midpoint that cleanly separates them.
        assert doubled < base * 3, (
            f"turn cost scaled {doubled / base:.2f}x for a 2x input "
            f"({base:.4f}s -> {doubled:.4f}s) — still quadratic."
        )


class TestCapsStillEnforced:
    """A faster accumulator that forgets the caps would be a worse bug: the
    buffers exist to be bounded (DESIGN §4.5)."""

    def test_chunk_cap_still_drops_and_logs(self, caplog) -> None:
        acc = _BoundedTextAcc([CHUNK] * QUEUED_TEXT_MAX_CHUNKS)
        with caplog.at_level("ERROR"):
            _append_bounded_text(acc, CHUNK)
        assert len(acc) == QUEUED_TEXT_MAX_CHUNKS
        assert "chunk cap" in caplog.text

    def test_byte_cap_still_drops_and_logs(self, caplog) -> None:
        big = "y" * (QUEUED_TEXT_MAX_BYTES // 2)
        acc = _BoundedTextAcc()
        _append_bounded_text(acc, big)
        _append_bounded_text(acc, big)
        with caplog.at_level("ERROR"):
            _append_bounded_text(acc, "z" * 1024)
        assert _text_acc_total(acc) <= QUEUED_TEXT_MAX_BYTES
        assert "byte cap" in caplog.text

    def test_empty_chunk_is_ignored(self) -> None:
        acc = _BoundedTextAcc()
        _append_bounded_text(acc, "")
        assert list(acc) == []
        assert _text_acc_total(acc) == 0


class TestCarriedTotalStaysTruthful:
    """The carried total must equal a real sum at all times — a drifting total
    would silently move the byte cap, which is worse than being slow."""

    def test_total_matches_a_real_sum(self) -> None:
        acc = _BoundedTextAcc()
        for i in range(500):
            _append_bounded_text(acc, "a" * (i % 17 + 1))
        assert _text_acc_total(acc) == sum(len(c) for c in acc)

    def test_total_is_correct_when_seeded_from_an_existing_list(self) -> None:
        seed = ["abc", "de", "f"]
        acc = _BoundedTextAcc(seed)
        assert _text_acc_total(acc) == 6
        _append_bounded_text(acc, "gh")
        assert _text_acc_total(acc) == 8
        assert list(acc) == ["abc", "de", "f", "gh"]

    def test_plain_list_still_works_via_the_fallback(self) -> None:
        """``state.queued_text`` can be a plain list (dataclass default, or a
        restored checkpoint). Correctness must not depend on the subclass."""
        acc: list[str] = ["abc", "de"]
        assert _text_acc_total(acc) == 5
        _append_bounded_text(acc, "f")
        assert list(acc) == ["abc", "de", "f"]
        assert _text_acc_total(acc) == 6

    def test_join_still_produces_the_full_text(self) -> None:
        """Readers use ``"".join(acc)`` — the subclass must stay a real list."""
        acc = _BoundedTextAcc()
        for part in ("Hello", ", ", "world", "!"):
            _append_bounded_text(acc, part)
        assert "".join(acc) == "Hello, world!"
        assert isinstance(acc, list)


class TestEventLoopIsNotHeld:
    """The drain must not monopolise the loop. Asserted as a shape: the work for
    a long turn stays small enough that a concurrent coroutine is not starved."""

    @pytest.mark.asyncio
    async def test_long_turn_leaves_the_loop_responsive(self) -> None:
        import asyncio

        acc = _BoundedTextAcc()
        latencies: list[float] = []
        stop = False

        async def pinger() -> None:
            while not stop:
                mark = time.perf_counter()
                await asyncio.sleep(0)
                latencies.append(time.perf_counter() - mark)

        task = asyncio.create_task(pinger())
        await asyncio.sleep(0)
        for _ in range(8000):
            _append_bounded_text(acc, CHUNK)
        stop = True
        await asyncio.sleep(0)
        task.cancel()

        worst = max(latencies) if latencies else 0.0
        # Pre-fix this measured 1.44 SECONDS at this turn size. A linear
        # accumulator keeps the whole drain in the low milliseconds; 250 ms is
        # far above any plausible linear cost and far below the quadratic one.
        assert worst < 0.25, (
            f"the accumulation held the event loop for {worst * 1e3:.0f} ms — "
            "a concurrent tool call would appear hung."
        )
