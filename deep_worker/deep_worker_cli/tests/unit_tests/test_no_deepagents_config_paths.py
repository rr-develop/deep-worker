"""POLICIES §16: configuration lives in ``~/.deep_worker``, never ``.deepagents``.

The regression this guards against is specific: the resolvers were spread
across six modules, each rebuilding the ``.deepagents`` literal itself, so
fixing one and leaving the others silently reintroduced the foreign path.
These tests pin the resolved VALUES rather than grepping for the string,
because a value assertion is what actually fails when an agent boots with
config from the wrong tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.config_store import scope_to_path
from deep_worker_cli.storage_paths import (
    agent_config_dir,
    config_home,
    project_config_dir,
)


def _assert_no_deepagents(path: Path) -> None:
    assert ".deepagents" not in path.parts, f"{path} points at the foreign tree"


def test_config_home_is_deep_worker(tmp_path: Path) -> None:
    resolved = config_home(tmp_path)
    assert resolved == tmp_path / ".deep_worker"
    _assert_no_deepagents(resolved)


def test_project_config_dir_is_deep_worker(tmp_path: Path) -> None:
    resolved = project_config_dir(tmp_path)
    assert resolved == tmp_path / ".deep_worker"
    _assert_no_deepagents(resolved)


def test_agent_config_dir_lives_under_agents_subdir(tmp_path: Path) -> None:
    """Per-agent config is a child of ``agents/``, not a top-level sibling.

    The top level is reserved for infrastructure (``logs/``, ``env.d/``, …);
    an agent dir placed there would collide with it.
    """
    resolved = agent_config_dir("bob", home=tmp_path)
    assert resolved == tmp_path / ".deep_worker" / "agents" / "bob"
    _assert_no_deepagents(resolved)


@pytest.mark.parametrize("scope", ["user", "agent:bob"])
def test_every_home_scope_resolves_into_deep_worker(scope: str, tmp_path: Path) -> None:
    _assert_no_deepagents(scope_to_path(scope, home=tmp_path))


def test_project_scope_resolves_into_deep_worker(tmp_path: Path) -> None:
    _assert_no_deepagents(scope_to_path("project", project_root=tmp_path))


def test_cli_scope_resolvers_agree_with_the_shared_one() -> None:
    """``deep-worker mcp`` and ``deep-worker lsp`` must not re-derive paths.

    Both CLIs previously carried their own copy of ``_scope_to_path``. Any
    divergence means ``mcp add`` writes where the agent does not read.
    """
    from deep_worker_cli.cli import lsp_command, mcp_command

    for scope in ("user", "agent:bob"):
        shared = scope_to_path(scope)
        assert mcp_command._scope_to_path(scope) == shared
        assert lsp_command._scope_to_path(scope) == shared
        _assert_no_deepagents(shared)


def test_agent_boot_reads_the_same_path_the_cli_writes() -> None:
    """The end-to-end contract: what ``mcp add --scope agent:x`` writes is
    exactly what agent ``x`` loads at boot. This is the pairing whose breakage
    produced a silently empty MCP layer.
    """
    from deep_worker_cli.cli import mcp_command

    written = mcp_command._scope_to_path("agent:some_agent")
    read_at_boot = scope_to_path("agent:some_agent")
    assert written == read_at_boot
    _assert_no_deepagents(written)
