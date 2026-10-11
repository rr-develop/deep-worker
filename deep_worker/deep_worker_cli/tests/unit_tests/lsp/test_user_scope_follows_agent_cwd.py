"""A user-scope bridge entry navigates the AGENT's own project.

``--scope user`` means "every project", so baking one literal
``--project-root`` into that layer made a single project the navigation
target for every agent on the machine (measured: 241 of 268 agents
resolved to one foreign repo). The user-scope entry now carries the
placeholder ``${agent:working_dir}``, which the loader substitutes with
each agent's own workspace root at start.

See ``docs/design/lsp_user_scope_follows_agent_cwd.md``.

Every test here fails on the pre-fix code, and for a substantive
reason — a wrong resolved ROOT, not a missing symbol:

* the three ``load_mcp_config`` tests fail because the old loader left
  ``${agent:working_dir}`` unsubstituted, so the root is the literal
  placeholder instead of the agent's directory;
* the writer tests fail because ``lsp enable --scope user`` used to
  write a fixed literal path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.config import (
    AGENT_WORKING_DIR_PLACEHOLDER,
    UnresolvedAgentPlaceholderError,
    load_mcp_config,
)
from deep_worker_cli.lsp.bridge import build_bridge_mcp_entry
from deep_worker_cli.lsp.enable import SERENA_SERVER_NAME, enable_lsp_at_path
from deep_worker_cli.lsp.migrate import migrate_mcp_config


def _write(path: Path, servers: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def _user_scope_entry() -> dict:
    """The user-scope entry as ``lsp enable --scope user`` now writes it."""
    return {
        "serena": {
            "type": "stdio",
            "command": "deep-worker",
            "args": [
                "lsp",
                "bridge",
                "--project-root",
                AGENT_WORKING_DIR_PLACEHOLDER,
            ],
            "auto_approve": True,
        }
    }


def _root_arg(args: list[str]) -> str:
    """The value following ``--project-root`` in a bridge argv."""
    return list(args)[list(args).index("--project-root") + 1]


def _root_of(spec) -> str:
    return _root_arg(spec.args)


def _git_project(tmp_path: Path, name: str) -> Path:
    """A directory that looks like a project root (has ``.git``)."""
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    return root


def test_user_scope_entry_resolves_to_the_agents_own_dir(tmp_path):
    """The whole point: one shared entry, root = this agent's project."""
    user_cfg = _write(tmp_path / "user.json", _user_scope_entry())
    wd = _git_project(tmp_path, "my_project")

    merged = load_mcp_config(
        user_path=user_cfg, project_path=None, agent_path=None, working_dir=wd
    )

    assert _root_of(merged["serena"]) == str(wd.resolve())


def test_two_agents_get_different_roots_from_one_entry(tmp_path):
    """Distinct axis: the substitution is per-load, not cached globally.

    A single stored entry must yield a different root per agent —
    otherwise the fix would merely relocate the original bug (one
    project for everyone) rather than remove it.
    """
    user_cfg = _write(tmp_path / "user.json", _user_scope_entry())
    a = _git_project(tmp_path, "project_a")
    b = _git_project(tmp_path, "project_b")

    root_a = _root_of(
        load_mcp_config(
            user_path=user_cfg, project_path=None, agent_path=None, working_dir=a
        )["serena"]
    )
    root_b = _root_of(
        load_mcp_config(
            user_path=user_cfg, project_path=None, agent_path=None, working_dir=b
        )["serena"]
    )

    assert root_a == str(a.resolve())
    assert root_b == str(b.resolve())
    assert root_a != root_b


def test_non_git_working_dir_resolves_to_that_dir(tmp_path):
    """No silent substitution when the directory is not a git project.

    Design §4: the bridge still starts and answers from that directory.
    The root must be that absolute path — never empty, never the config
    file's location.
    """
    user_cfg = _write(tmp_path / "user.json", _user_scope_entry())
    plain = tmp_path / "not_a_repo"
    plain.mkdir()

    merged = load_mcp_config(
        user_path=user_cfg, project_path=None, agent_path=None, working_dir=plain
    )

    assert _root_of(merged["serena"]) == str(plain.resolve())


def test_project_scope_still_overrides_the_user_entry(tmp_path):
    """Precedence is unchanged: a same-named higher layer fully wins."""
    user_cfg = _write(tmp_path / "user.json", _user_scope_entry())
    pinned = tmp_path / "pinned_project"
    pinned.mkdir()
    proj_cfg = _write(
        tmp_path / "proj.json",
        {
            "serena": {
                "type": "stdio",
                "command": "deep-worker",
                "args": ["lsp", "bridge", "--project-root", str(pinned)],
                "auto_approve": True,
            }
        },
    )
    wd = _git_project(tmp_path, "agent_cwd")

    merged = load_mcp_config(
        user_path=user_cfg,
        project_path=proj_cfg,
        agent_path=None,
        working_dir=wd,
    )

    assert len(merged) == 1, merged
    assert _root_of(merged["serena"]) == str(pinned)


def test_differently_named_entry_adds_a_second_server(tmp_path):
    """Access to ANOTHER project is additive, not a substitution.

    This is the owner's rule: reaching a foreign project must not cost
    you navigation over your own. A second name means a second server.
    """
    other = tmp_path / "other_project"
    other.mkdir()
    servers = _user_scope_entry()
    servers["serena_other"] = {
        "type": "stdio",
        "command": "deep-worker",
        "args": ["lsp", "bridge", "--project-root", str(other)],
        "auto_approve": True,
    }
    user_cfg = _write(tmp_path / "user.json", servers)
    wd = _git_project(tmp_path, "own_project")

    merged = load_mcp_config(
        user_path=user_cfg, project_path=None, agent_path=None, working_dir=wd
    )

    assert set(merged) == {"serena", "serena_other"}
    assert _root_of(merged["serena"]) == str(wd.resolve())
    assert _root_of(merged["serena_other"]) == str(other)


def test_placeholder_without_working_dir_fails_loudly(tmp_path):
    """A caller with no agent context must not get a blank root.

    Same loud-by-design contract as ``${env:VAR}``: resolving to an
    empty path would index the wrong tree, which is the defect the
    placeholder exists to prevent.
    """
    user_cfg = _write(tmp_path / "user.json", _user_scope_entry())

    with pytest.raises(UnresolvedAgentPlaceholderError):
        load_mcp_config(
            user_path=user_cfg,
            project_path=None,
            agent_path=None,
            working_dir=None,
        )


def test_enable_user_scope_writes_the_placeholder_not_a_literal(tmp_path):
    """``lsp enable --scope user`` must not pin the invoking cwd."""
    cfg = tmp_path / "user.json"
    invoked_from = _git_project(tmp_path, "whatever_cwd")

    enable_lsp_at_path(
        cfg, workspace_root=invoked_from, follow_agent_cwd=True
    )

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"][
        SERENA_SERVER_NAME
    ]
    assert _root_arg(entry["args"]) == AGENT_WORKING_DIR_PLACEHOLDER
    assert str(invoked_from) not in json.dumps(entry)


def test_enable_project_scope_keeps_the_literal_root(tmp_path):
    """Project scope is unchanged — naming one project is its purpose."""
    cfg = tmp_path / "project.json"
    root = _git_project(tmp_path, "pinned")

    enable_lsp_at_path(cfg, workspace_root=root)

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"][
        SERENA_SERVER_NAME
    ]
    assert _root_arg(entry["args"]) == str(root.resolve())
    assert AGENT_WORKING_DIR_PLACEHOLDER not in json.dumps(entry)


def test_builder_requires_a_root_unless_following_cwd():
    """Guard the builder's own contract: no accidental rootless entry."""
    with pytest.raises(ValueError):
        build_bridge_mcp_entry()


# ── ``lsp migrate`` must not reintroduce the literal ──
#
# ``lsp migrate`` accepts ``--scope user`` too, so migrating a legacy
# entry at user scope would recreate the original defect verbatim: the
# legacy ``--project <repo>`` literal becomes every agent's navigation
# target. The no-``--project`` legacy form is worse — its fallback is
# the config file's own directory (``~/.deep_worker`` at user scope).

#: The legacy shape's discriminator, as ``migrate`` itself defines it.
#: Taken from the module rather than retyped so an invented fixture
#: cannot silently fail to be recognised as legacy.
_LEGACY_SERENA = {
    "type": "stdio",
    "command": "serena",
    "args": ["start-mcp-server", "--project", "/path/to/some_other_repo"],
    "auto_approve": True,
}


def test_migrate_user_scope_writes_the_placeholder(tmp_path):
    """user scope → placeholder, NOT the legacy literal."""
    cfg = tmp_path / "dot_deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"mcpServers": {"serena": dict(_LEGACY_SERENA)}}),
        encoding="utf-8",
    )

    assert migrate_mcp_config(cfg, follow_agent_cwd=True) is True

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["serena"]
    assert _root_arg(entry["args"]) == AGENT_WORKING_DIR_PLACEHOLDER
    assert "/path/to/some_other_repo" not in json.dumps(entry)


def test_migrate_user_scope_never_roots_at_the_config_dir(tmp_path):
    """The no-``--project`` legacy form must not root at ``~/.deep_worker``.

    Distinct axis from the test above: here the legacy entry carries no
    root at all, so the pre-fix fallback was ``path.parent``.
    """
    cfg = tmp_path / "dot_deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    legacy = dict(_LEGACY_SERENA, args=["start-mcp-server"])
    cfg.write_text(
        json.dumps({"mcpServers": {"serena": legacy}}), encoding="utf-8"
    )

    assert migrate_mcp_config(cfg, follow_agent_cwd=True) is True

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["serena"]
    assert _root_arg(entry["args"]) == AGENT_WORKING_DIR_PLACEHOLDER
    assert str(cfg.parent) not in json.dumps(entry)


def test_migrate_project_scope_keeps_the_legacy_literal(tmp_path):
    """Project scope still recovers the explicit root — unchanged."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"serena": dict(_LEGACY_SERENA)}}),
        encoding="utf-8",
    )

    assert migrate_mcp_config(cfg) is True

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["serena"]
    assert _root_arg(entry["args"]) == "/path/to/some_other_repo"
    assert AGENT_WORKING_DIR_PLACEHOLDER not in json.dumps(entry)


def test_mcp_list_still_lists_with_a_placeholder_entry(tmp_path, monkeypatch):
    """``mcp list`` must keep inspecting.

    One unresolvable placeholder used to abort the whole merge, hiding
    every unrelated server too — an inspection command that stops
    inspecting. It now resolves against the invocation cwd.
    """
    from deep_worker_cli.cli import mcp_command

    servers = _user_scope_entry()
    servers["unrelated"] = {
        "type": "stdio",
        "command": "echo",
        "args": ["hi"],
        "auto_approve": False,
    }
    user_cfg = _write(tmp_path / "user.json", servers)
    wd = _git_project(tmp_path, "cwd_project")

    monkeypatch.setattr(
        mcp_command,
        "_scope_to_path",
        lambda scope: user_cfg if scope == "user" else tmp_path / "none.json",
    )
    monkeypatch.setattr(mcp_command, "invocation_cwd", lambda: wd)

    rc = mcp_command._cmd_list("all")

    assert rc == 0
