"""``LocalContextMiddleware`` is skipped when built-in tools are switched off.

A selection that denies the built-in tools (``--tools none,<tool>``, or a bare
allow-list, both of which ``parse_tool_selection`` normalises to ``deny_all``)
produces a single-purpose worker. It holds one tool and cannot act on a
directory listing at all, so the ``## Local Context`` section is prompt noise
the model may still read as an instruction.

``rg_cross_file_updater`` ran that way on 2026-10-01 (``--tools none,<submit>``)
and its prompt carried a tree naming a stale ``AGENTS.override.md`` left behind
by a killed codex run — see ``local_context.py:600-618`` for the file-list and
tree sections, and ``agent.py`` for the gate.

The provider is deliberately NOT part of the condition: the tool surface
decides, so this holds for deepseek and claude-code alike.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.local_context import LocalContextMiddleware
from deep_worker_cli.tool_selection import parse_tool_selection

#: ``rg_cross_file_updater`` launches with
#: ``--tools none,repo_graph_cross_file_update__submit_...`` (its
#: ``session_metadata.json`` ``cli_args``). That MCP tool needs a live server
#: to appear in the inventory, and ``validate_against_inventory`` rejects a
#: name it cannot see — so the kept tool here is a built-in. What matters for
#: the gate is the SHAPE (``none`` plus one kept tool → ``deny_all``), which
#: is identical either way; ``test_deny_all_predicts_local_context_absence``
#: pins that equivalence on the parse result.
KEPT_TOOL = "write_file"


def _mock_settings(tmp_path: Path, agent_dir: Path) -> Mock:
    settings = Mock()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    settings.ensure_user_skills_dir.return_value = skills_dir
    settings.get_project_skills_dir.return_value = None
    settings.get_built_in_skills_dir.return_value = Path("/fake/built-in")
    settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
    settings.get_project_agent_md_path.return_value = None
    settings.get_user_agents_dir.return_value = tmp_path / "agents"
    settings.get_project_agents_dir.return_value = None
    settings.model_name = None
    settings.model_provider = None
    settings.model_context_limit = None
    settings.project_root = None
    settings.user_langchain_project = None
    return settings


def _build_and_capture(
    tmp_path: Path, spec: list[str] | None
) -> tuple[list[Any], str]:
    """Build an agent and return ``(middleware_chain, system_prompt)``.

    The chain is the real list handed to ``create_deep_agent`` — nothing is
    patched out, so membership of ``LocalContextMiddleware`` is observed on
    the production wiring rather than on a stand-in.
    """
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    working_dir = tmp_path / "workspace"
    working_dir.mkdir(exist_ok=True)

    captured: dict[str, Any] = {}

    def capture_create_deep_agent(**kwargs: Any) -> Mock:
        captured.update(kwargs)
        agent = Mock()
        agent.with_config.return_value = agent
        return agent

    selection = parse_tool_selection(spec) if spec is not None else None

    with (
        patch("deep_worker_cli.agent.settings", _mock_settings(tmp_path, agent_dir)),
        patch(
            "deepagents.create_deep_agent",
            side_effect=capture_create_deep_agent,
        ),
    ):
        create_cli_agent(
            model="fake-model",
            assistant_id="test",
            enable_shell=True,
            enable_memory=False,
            enable_skills=False,
            project_memory_dir=working_dir,
            tool_selection=selection,
        )

    chain = list(captured.get("middleware") or ())
    prompt = str(
        captured.get("system_prompt")
        or captured.get("instructions")
        or captured.get("prompt")
        or ""
    )
    return chain, prompt


def _has_local_context(chain: list[Any]) -> bool:
    return any(isinstance(mw, LocalContextMiddleware) for mw in chain)


class TestBuiltInToolsOffSkipsLocalContext:
    """(a) ``none,<tool>`` → no LocalContextMiddleware, no ``## Local Context``."""

    def test_none_plus_one_tool_drops_local_context(self, tmp_path: Path) -> None:
        chain, prompt = _build_and_capture(tmp_path, ["none", KEPT_TOOL])

        assert not _has_local_context(chain), (
            "LocalContextMiddleware must be absent when the selection denies "
            "the built-in tools; it was in the chain: "
            f"{[type(mw).__name__ for mw in chain]}"
        )
        assert "## Local Context" not in prompt
        assert "**Tree** (3 levels)" not in prompt

    def test_bare_none_drops_local_context(self, tmp_path: Path) -> None:
        """Deny-everything with nothing kept is the same shape."""
        chain, _ = _build_and_capture(tmp_path, ["none"])
        assert not _has_local_context(chain)

    def test_explicit_deny_all_form_drops_local_context(
        self, tmp_path: Path
    ) -> None:
        """``-*`` is the canonical spelling ``none`` aliases."""
        chain, _ = _build_and_capture(tmp_path, ["-*"])
        assert not _has_local_context(chain)

    def test_bare_allow_list_drops_local_context(self, tmp_path: Path) -> None:
        """A bare allow-list also switches the built-ins off.

        ``parse_tool_selection`` sets ``deny_all`` for a bare list (see its
        ``if allow and not deny_all`` branch) precisely because writing names
        out means "keep only these" — so such an agent is single-purpose too.
        """
        chain, _ = _build_and_capture(tmp_path, ["read_file"])
        assert not _has_local_context(chain)


class TestBuiltInToolsOnKeepsLocalContext:
    """(b) Without a deny-all selection the behaviour is exactly as before."""

    def test_no_selection_keeps_local_context(self, tmp_path: Path) -> None:
        chain, _ = _build_and_capture(tmp_path, None)
        assert _has_local_context(chain), (
            "The default path must be untouched: LocalContextMiddleware was "
            f"missing from {[type(mw).__name__ for mw in chain]}"
        )

    def test_deny_one_tool_keeps_local_context(self, tmp_path: Path) -> None:
        """``-execute`` removes one tool; the built-ins are still on."""
        chain, _ = _build_and_capture(tmp_path, ["-execute"])
        assert _has_local_context(chain)

    def test_star_minus_one_keeps_local_context(self, tmp_path: Path) -> None:
        """``*,-execute`` is every tool except one — still not deny-all."""
        chain, _ = _build_and_capture(tmp_path, ["*", "-execute"])
        assert _has_local_context(chain)

    def test_increment_only_keeps_local_context(self, tmp_path: Path) -> None:
        """An incremental rule alone does not switch the built-ins off."""
        chain, _ = _build_and_capture(tmp_path, ["+glob"])
        assert _has_local_context(chain)


class TestGateUsesTheFilterSRepresentation:
    """The gate reads ``deny_all`` — the same field the tool filter uses.

    Guards against the gate drifting onto a re-parse of the raw spec string,
    which would be a second, divergent reading of ``--tools``.
    """

    @pytest.mark.parametrize(
        ("spec", "deny_all"),
        [
            (["none", KEPT_TOOL], True),
            (["none"], True),
            (["-*"], True),
            (["read_file"], True),
            (["-execute"], False),
            (["*", "-execute"], False),
            (["+glob"], False),
        ],
    )
    def test_deny_all_predicts_local_context_absence(
        self, tmp_path: Path, spec: list[str], deny_all: bool
    ) -> None:
        assert parse_tool_selection(spec).deny_all is deny_all
        chain, _ = _build_and_capture(tmp_path, spec)
        assert _has_local_context(chain) is (not deny_all)
