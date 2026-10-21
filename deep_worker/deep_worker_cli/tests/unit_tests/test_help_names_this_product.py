"""Help screens must name the program the user actually has.

The Rich help screens were inherited from the ancestor and still printed
``deepagents`` — a command that does not exist in this product. So every
usage line and every worked example instructed the reader to run something
that fails. These tests render each screen and assert the name.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from deep_worker_cli import ui

if TYPE_CHECKING:
    from collections.abc import Callable

_HELP_SCREENS: list[tuple[str, Callable[[], None]]] = [
    ("top-level", ui.show_help),
    ("list", ui.show_list_help),
    ("reset", ui.show_reset_help),
    ("skills", ui.show_skills_help),
    ("skills-list", ui.show_skills_list_help),
    ("skills-create", ui.show_skills_create_help),
    ("skills-info", ui.show_skills_info_help),
    ("threads", ui.show_threads_help),
    ("threads-list", ui.show_threads_list_help),
    ("threads-delete", ui.show_threads_delete_help),
]


def _render(screen: Callable[[], None], monkeypatch: pytest.MonkeyPatch) -> str:
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", Console(file=buf, highlight=False, width=120))
    screen()
    return buf.getvalue()


@pytest.mark.parametrize(
    ("name", "screen"), _HELP_SCREENS, ids=[n for n, _ in _HELP_SCREENS]
)
def test_help_screen_never_names_the_ancestor(
    name: str, screen: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No help screen may mention the ancestor product or its docs site."""
    output = _render(screen, monkeypatch)

    assert "deepagents" not in output.lower(), (
        f"the {name} help screen still names the ancestor"
    )
    assert "docs.langchain.com" not in output, (
        f"the {name} help screen still points at the ancestor's documentation"
    )


@pytest.mark.parametrize(
    ("name", "screen"), _HELP_SCREENS, ids=[n for n, _ in _HELP_SCREENS]
)
def test_help_screen_names_this_product(
    name: str, screen: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each screen tells the reader the real command name.

    Paired with the negative test above on purpose: "says nothing about the
    ancestor" is also satisfied by a screen that names no command at all,
    which would be a different way of being useless.
    """
    output = _render(screen, monkeypatch)

    assert "deep-worker" in output, (
        f"the {name} help screen names no runnable command"
    )
