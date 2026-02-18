"""Tests for scroll-based pagination via ChatScroll widget.

This module tests the event-driven pagination system that replaces
the previous polling-based approach (_check_scroll_position with
set_interval(0.5, ...)).

Architecture being tested:
- ChatScroll.watch_scroll_y detects upward scroll and posts ScrolledUp
- InteractiveApp._maybe_load_more_history checks cached state and
  topmost visible message widget index to decide whether to paginate
- Pagination only triggers when fewer than 10 already-loaded messages
  exist above the topmost visible message in the DOM
- History availability check uses cached _history_offset/_history_total
  (no extra RPyC calls)

See ARCHITECTURE.md section "History Pagination" for full design rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add arpyc to path (same pattern as other test files)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / "arpyc"))

from textual.containers import VerticalScroll

from deep_worker_cli.cli.interactive import ChatScroll, InteractiveApp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(history_offset: int, history_total: int, loading: bool = False) -> InteractiveApp:
    """Create a minimal InteractiveApp with pagination state pre-set."""
    app = InteractiveApp(agent_name="test-agent")
    app._history_offset = history_offset
    app._history_total = history_total
    app._loading_history = loading
    return app


def _make_children(region_ys: list[int]) -> list[MagicMock]:
    """Create mock message widgets with given screen Y coordinates.

    Args:
        region_ys: List of widget.region.y values. Negative = above viewport,
                   non-negative = visible at or below top of viewport.
    """
    children = []
    for y in region_ys:
        child = MagicMock()
        child.region.y = y
        children.append(child)
    return children


# ---------------------------------------------------------------------------
# Tests: ChatScroll.ScrolledUp message emission
# ---------------------------------------------------------------------------

class TestChatScrollScrolledUp:
    """ChatScroll must post ScrolledUp only when the user scrolls upward.

    watch_scroll_y fires on ALL scroll input (mouse wheel, keyboard Page Up /
    arrow keys, scrollbar drag) because it watches the scroll_y reactive.
    This is more reliable than the ScrollUp event, which only fires on
    scrollbar arrow clicks.
    """

    def test_scroll_up_posts_scrolled_up(self):
        """Decreasing scroll_y (new < old) must post ScrolledUp."""
        scroll = ChatScroll()
        posted: list = []
        with patch.object(scroll, "post_message", side_effect=posted.append):
            scroll.watch_scroll_y(100.0, 50.0)

        assert len(posted) == 1
        assert isinstance(posted[0], ChatScroll.ScrolledUp)

    def test_scroll_down_does_not_post(self):
        """Increasing scroll_y (new > old) must NOT post ScrolledUp."""
        scroll = ChatScroll()
        posted: list = []
        with patch.object(scroll, "post_message", side_effect=posted.append):
            scroll.watch_scroll_y(50.0, 100.0)

        assert len(posted) == 0

    def test_no_change_does_not_post(self):
        """Unchanged scroll_y (new == old) must NOT post ScrolledUp."""
        scroll = ChatScroll()
        posted: list = []
        with patch.object(scroll, "post_message", side_effect=posted.append):
            scroll.watch_scroll_y(100.0, 100.0)

        assert len(posted) == 0

    def test_scroll_to_zero_posts_scrolled_up(self):
        """Scrolling all the way to the top (new=0) must still post ScrolledUp."""
        scroll = ChatScroll()
        posted: list = []
        with patch.object(scroll, "post_message", side_effect=posted.append):
            scroll.watch_scroll_y(30.0, 0.0)

        assert len(posted) == 1
        assert isinstance(posted[0], ChatScroll.ScrolledUp)

    def test_multiple_scroll_ups_post_multiple_messages(self):
        """Each upward step must produce one ScrolledUp message."""
        scroll = ChatScroll()
        posted: list = []
        with patch.object(scroll, "post_message", side_effect=posted.append):
            scroll.watch_scroll_y(100.0, 80.0)
            scroll.watch_scroll_y(80.0, 60.0)
            scroll.watch_scroll_y(60.0, 40.0)

        assert len(posted) == 3
        assert all(isinstance(m, ChatScroll.ScrolledUp) for m in posted)


# ---------------------------------------------------------------------------
# Tests: super().watch_scroll_y() is always called
# ---------------------------------------------------------------------------

class TestChatScrollCallsSuper:
    """ChatScroll.watch_scroll_y MUST call super() on every invocation.

    The parent VerticalScroll.watch_scroll_y does three things that are
    critical for correct rendering:
      1. Updates the visual scrollbar thumb position.
      2. Runs the anchor check (_check_anchor) for sticky-scroll behaviour.
      3. Calls _refresh_scroll() which redraws the scroll content at the
         new scroll offset.

    Without the super() call:
      - The display is NOT updated when scroll_y changes, so the input area
        border appears missing (the screen is "stuck" at the old position).
      - widget.region.y values become stale, causing _maybe_load_more_history
        to see a wrong topmost-visible index and spuriously load history,
        which duplicates messages in the chat.
    """

    def test_super_called_on_scroll_up(self):
        """super().watch_scroll_y() must be called when scrolling up."""
        scroll = ChatScroll()
        super_calls: list = []

        with patch.object(
            VerticalScroll,
            "watch_scroll_y",
            side_effect=lambda old, new: super_calls.append((old, new)),
        ), patch.object(scroll, "post_message", lambda _: None):
            scroll.watch_scroll_y(100.0, 50.0)

        assert len(super_calls) == 1
        assert super_calls[0] == (100.0, 50.0)

    def test_super_called_on_scroll_down(self):
        """super().watch_scroll_y() must ALSO be called when scrolling down.

        The parent's _refresh_scroll() must run regardless of direction so
        that the scrollbar and content always stay in sync.
        """
        scroll = ChatScroll()
        super_calls: list = []

        with patch.object(
            VerticalScroll,
            "watch_scroll_y",
            side_effect=lambda old, new: super_calls.append((old, new)),
        ), patch.object(scroll, "post_message", lambda _: None):
            scroll.watch_scroll_y(50.0, 100.0)

        assert len(super_calls) == 1
        assert super_calls[0] == (50.0, 100.0)

    def test_super_called_before_post_message(self):
        """super() must be called BEFORE post_message so Textual state is
        updated (region.y refreshed) before _maybe_load_more_history reads it.
        """
        scroll = ChatScroll()
        call_order: list = []

        with patch.object(
            VerticalScroll,
            "watch_scroll_y",
            side_effect=lambda old, new: call_order.append("super"),
        ), patch.object(
            scroll,
            "post_message",
            side_effect=lambda _: call_order.append("post_message"),
        ):
            scroll.watch_scroll_y(100.0, 50.0)

        assert call_order == ["super", "post_message"], (
            f"super() must be called before post_message; got order: {call_order}"
        )


# ---------------------------------------------------------------------------
# Tests: InteractiveApp._maybe_load_more_history guard conditions
# ---------------------------------------------------------------------------

class TestMaybeLoadMoreHistoryGuards:
    """_maybe_load_more_history must short-circuit without triggering RPyC
    when the cached state already tells us there is nothing to do."""

    def test_no_trigger_when_history_fully_loaded(self):
        """offset >= total means all history is loaded; must not paginate."""
        app = _make_app(history_offset=100, history_total=100)
        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append):
            app._maybe_load_more_history()
        assert len(workers) == 0

    def test_no_trigger_when_offset_exceeds_total(self):
        """Defensive: offset > total must also be treated as fully loaded."""
        app = _make_app(history_offset=110, history_total=100)
        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append):
            app._maybe_load_more_history()
        assert len(workers) == 0

    def test_no_trigger_when_already_loading(self):
        """Concurrent load guard: if _loading_history is True, skip."""
        app = _make_app(history_offset=20, history_total=100, loading=True)
        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append):
            app._maybe_load_more_history()
        assert len(workers) == 0

    def test_no_crash_when_messages_container_not_found(self):
        """NoMatches from query_one must be swallowed gracefully."""
        from textual.css.query import NoMatches

        app = _make_app(history_offset=20, history_total=100)
        with patch.object(app, "query_one", side_effect=NoMatches()):
            # Must not raise
            app._maybe_load_more_history()

    def test_no_trigger_when_container_is_empty(self):
        """Empty messages container means nothing is visible; skip."""
        app = _make_app(history_offset=20, history_total=100)
        mock_container = MagicMock()
        mock_container.children = []

        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append), \
             patch.object(app, "query_one", return_value=mock_container):
            app._maybe_load_more_history()

        assert len(workers) == 0

    def test_no_trigger_when_all_messages_scrolled_above_viewport(self):
        """If all region.y < 0 (all above viewport), no visible anchor; skip."""
        app = _make_app(history_offset=20, history_total=100)
        # All messages are above the viewport
        mock_container = MagicMock()
        mock_container.children = _make_children([-100, -80, -60, -40, -20])

        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append), \
             patch.object(app, "query_one", return_value=mock_container):
            app._maybe_load_more_history()

        assert len(workers) == 0


# ---------------------------------------------------------------------------
# Tests: _maybe_load_more_history threshold logic (topmost visible index)
# ---------------------------------------------------------------------------

class TestMaybeLoadMoreHistoryThreshold:
    """Pagination triggers when fewer than 10 loaded messages exist above
    the topmost visible widget.

    widget.region.y >= 0  →  visible at or below the top of the viewport
    widget.region.y < 0   →  scrolled above the viewport (not visible)

    The index of the first widget with region.y >= 0 is the "topmost visible
    index".  If that index is < 10, we load the next history page.
    """

    def _run(self, region_ys: list[int], *, offset: int = 20, total: int = 100) -> int:
        """Run _maybe_load_more_history and return number of run_worker calls."""
        app = _make_app(history_offset=offset, history_total=total)
        mock_container = MagicMock()
        mock_container.children = _make_children(region_ys)

        workers: list = []
        with patch.object(app, "run_worker", side_effect=workers.append), \
             patch.object(app, "query_one", return_value=mock_container):
            app._maybe_load_more_history()

        return len(workers)

    # --- below threshold: must trigger ---

    def test_topmost_at_index_0_triggers(self):
        """Topmost visible at index 0 (all messages visible) → trigger."""
        # region_ys: all non-negative → topmost at index 0
        assert self._run([0, 10, 20, 30]) == 1

    def test_topmost_at_index_5_triggers(self):
        """Topmost visible at index 5 (5 above) → trigger."""
        assert self._run([-50] * 5 + [0, 10, 20]) == 1

    def test_topmost_at_index_9_triggers_boundary(self):
        """Boundary: topmost visible at index 9 (9 < 10) → trigger."""
        assert self._run([-50] * 9 + [0, 10]) == 1

    # --- at or above threshold: must NOT trigger ---

    def test_topmost_at_index_10_no_trigger_boundary(self):
        """Boundary: topmost visible at index 10 (10 == 10) → no trigger."""
        assert self._run([-50] * 10 + [0, 10]) == 0

    def test_topmost_at_index_15_no_trigger(self):
        """Topmost visible at index 15 (15 > 10) → no trigger."""
        assert self._run([-50] * 15 + [0, 10]) == 0

    def test_topmost_at_index_50_no_trigger(self):
        """Large buffer (50 messages above) → no trigger."""
        assert self._run([-50] * 50 + [0, 10]) == 0

    # --- single message ---

    def test_single_visible_message_at_index_0_triggers(self):
        """Single visible message at index 0 → trigger (0 < 10)."""
        assert self._run([0]) == 1

    def test_single_message_above_viewport_no_trigger(self):
        """Single message above viewport (no visible anchor) → no trigger."""
        assert self._run([-10]) == 0


# ---------------------------------------------------------------------------
# Tests: on_chat_scroll_scrolled_up handler wiring
# ---------------------------------------------------------------------------

class TestScrolledUpHandlerWiring:
    """The Textual event handler must delegate to _maybe_load_more_history."""

    def test_handler_calls_maybe_load_more_history(self):
        """on_chat_scroll_scrolled_up must call _maybe_load_more_history."""
        app = InteractiveApp(agent_name="test-agent")
        called: list = []
        with patch.object(app, "_maybe_load_more_history", side_effect=lambda: called.append(1)):
            event = ChatScroll.ScrolledUp()
            app.on_chat_scroll_scrolled_up(event)

        assert len(called) == 1

    def test_handler_does_not_call_run_worker_directly(self):
        """Handler must NOT call run_worker directly; delegation is through
        _maybe_load_more_history which decides whether to call it."""
        app = InteractiveApp(agent_name="test-agent")
        run_worker_calls: list = []
        with patch.object(app, "_maybe_load_more_history", lambda: None), \
             patch.object(app, "run_worker", side_effect=run_worker_calls.append):
            event = ChatScroll.ScrolledUp()
            app.on_chat_scroll_scrolled_up(event)

        assert len(run_worker_calls) == 0
