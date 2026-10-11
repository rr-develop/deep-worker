"""``deep-worker --restart <name>`` must not silently wipe persisted
launch-config fields.

Reported 2026-08-14 by an agent operator: the agent's ``mcp_config_paths``
was being reset to ``[]`` on every ``restart_self``, dragging external MCP
tools out of the graph. Reproduced on a probe agent live. Owner
constraint verbatim: "Configs must not be overwritten on restart at
all!!!! Options specified at restart may be added or removed, but
otherwise nothing may be overwritten!!!!"

Root cause (from ``DESIGN_RESTART_PRESERVES_LAUNCH_CONFIG.md``):
``restart_agent()`` step 5 copies ``effective`` back into ``args``
field-by-field via a hand-written enumeration, then calls
``run_agent(args)``. SIX persisted inputs of ``build_launch_config``
are missing from that list, so ``run_agent`` sees the raw parser
defaults and ``build_launch_config(args)`` overwrites the persisted
values:

  1. ``mcp_config_paths``    → ``[]``      (the reported symptom)
  2. ``mcp_server_inline``   → ``[]``
  3. ``turn_ending_tools``   → ``[]``
  4. ``dev_tree``            → masked: re-derived from ``_repo_root``
  5. ``web_search_policy``   → ``"auto"``
  6. ``preferred_model``     → masked when ``provider ==
     preferred_provider`` (re-derived from ``model``); observable, and
     damaging, only in the failover shape ``provider !=
     preferred_provider``

Fields 4 and 6 are the reason a "read the persisted file" assertion is
not enough on its own: ``build_launch_config`` re-derives both from
other state, so such a test stays green while the field is in fact
being dropped from the CLI dataflow. Those two are asserted at the
boundary where the loss is actually observable.

Why these tests are new and drive the REAL ``restart_agent()``: the
shipped ``merge_restart_config`` tests exercise only that helper in
isolation — and it was always correct. The defect lived one function
lower, in the copy-back. Testing at ``restart_agent`` end-to-end is
what would have caught it.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.cli import agent_command
from deep_worker_cli.cli.agent_command import _PARSER_DEFAULTS
from deep_worker_cli.tool_selection import parse_tool_selection


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _write_saved_config(
    tmp_path: Path,
    name: str,
    extra: dict,
) -> Path:
    """Write ``session_metadata.json`` for an existing agent named ``name``.

    Returns the agent's data dir. Only fields relevant to the persisted
    launch config need to be supplied via ``extra``; the required baseline
    (provider/model/working_dir) is filled in.
    """
    data_dir = tmp_path / "agents" / name
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir()
    launch_config = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": str(tmp_path),
        "skip_permissions": False,
        "task_manager": False,
        "memory_only": False,
        "history_memory": False,
        "use_sqlite": False,
        "clean": False,
        "auto_approve_send_message": False,
        "dev_tree": False,
        "web_search_policy": "auto",
        "preferred_provider": "deepseek",
        "preferred_model": None,
        "mcp_config_paths": [],
        "mcp_server_inline": [],
        "turn_ending_tools": [],
        "storage_version": 5,
        "tree_root": "/tmp",
        "started_at": "2026-08-14T00:00:00",
        "pid": 1234,
        "cli_args": "",
    }
    launch_config.update(extra)
    metadata = {
        "status": "idle",
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": "2026-08-14T00:00:00",
        "storage_version": 5,
        "launch_config": launch_config,
    }
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2))
    return data_dir


def _restart_args(name: str, **cli_overrides) -> argparse.Namespace:
    """Build the argparse namespace ``deep-worker --restart <name>`` produces.

    All fields the CLI parser sets on a bare ``--restart`` invocation are
    listed here so tests can override individual ones (e.g. pass a fresh
    ``--mcp-config`` on restart). Matches ``_PARSER_DEFAULTS`` shape.
    """
    ns = argparse.Namespace(
        name=name,
        restart=True,
        provider="deepseek",
        model=None,
        dir=None,
        skip_permissions=False,
        task_manager=False,
        memory_only=False,
        history_memory=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        dev_tree=False,
        web_search_policy="auto",
        # NOTE: the parser default for --preferred-provider is None, NOT a
        # provider name. Seeding "deepseek" here would make _is_override()
        # treat every bare --restart as an explicit -pp declaration and fire
        # the pp-block (which resets model/preferred_model), silently
        # changing what these tests exercise.
        preferred_provider=_PARSER_DEFAULTS["preferred_provider"],
        preferred_model=None,
        no_update_preferred=False,
        mcp_config_paths=[],
        mcp_server_inline=[],
        turn_ending_tools=[],
        # Tool selection: ``--tools`` is absolute, the two increments are
        # relative to whatever selection is already in force.
        tools=None,
        tools_add=None,
        tools_remove=None,
        config=None,
        summarize=False,
        fork=None,
        fork_from=None,
        context_limit=None,
    )
    for k, v in cli_overrides.items():
        setattr(ns, k, v)
    return ns


class _SpyCleanup:
    async def __call__(self, *a, **k):
        return None


async def _shutdown_event_wait(self):  # replaces AgentProcess.shutdown_event.wait
    return None


class _EventLike:
    async def wait(self):
        return None


def _install_agent_process_spy(monkeypatch, persist_root: Path | None = None) -> list[dict]:
    """Capture the ``launch_config`` passed to ``AgentProcess(...)``.

    Returns the list into which each call's kwargs are appended.

    When ``persist_root`` is given, the spy's ``start()`` also WRITES the
    captured ``launch_config`` to ``<persist_root>/<name>/session_metadata.json``,
    mirroring what the real ``AgentProcess.start()`` does via
    ``SessionManager(data_dir=…, launch_config=…)`` (session_manager.py:100-107,
    226-227). Without this, a no-op ``start()`` leaves the seed file on disk
    untouched, so any test that restarts twice merely re-reads the seed twice
    and cannot observe a restart-over-restart regression at all. A reviewer
    flagged exactly that vacuity on 2026-08-14.
    """
    calls: list[dict] = []

    def _spy_init(agent_self, *args, **kwargs):
        calls.append(kwargs)
        agent_self.name = kwargs.get("name")
        agent_self.running = False
        agent_self.shutdown_event = _EventLike()
        agent_self._spy_launch_config = kwargs.get("launch_config")

    async def _spy_start(agent_self):
        if persist_root is None:
            return None
        # Faithful stand-in for the real start()'s persistence step: the
        # launch_config this process was constructed with becomes the
        # launch_config the NEXT --restart will read back.
        meta_file = persist_root / "agents" / agent_self.name / "session_metadata.json"
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        metadata["launch_config"] = agent_self._spy_launch_config
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(metadata, indent=2, default=str))
        return None

    async def _spy_cleanup(agent_self):
        return None

    monkeypatch.setattr(agent_command.AgentProcess, "__init__", _spy_init)
    monkeypatch.setattr(agent_command.AgentProcess, "start", _spy_start)
    monkeypatch.setattr(agent_command.AgentProcess, "cleanup", _spy_cleanup)
    return calls


def _no_registered_agent(monkeypatch):
    """Make ``ProcessRegistry.lookup`` return None so restart skips the stop step."""

    class _EmptyRegistry:
        def __init__(self, *a, **k):
            pass

        def lookup(self, name):
            return None

    monkeypatch.setattr(
        "deep_worker_cli.process.registry.ProcessRegistry",
        _EmptyRegistry,
    )


@pytest.fixture
def _sandbox(tmp_path, monkeypatch):
    """Point deep_worker_home at a per-test temp dir.

    Both call sites are patched because ``agent_root`` (in
    ``deep_worker_cli.storage_paths``) uses its own module-level binding
    to ``deep_worker_home``, while ``agent_command.py`` keeps a direct
    import alias. DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.5.
    """
    monkeypatch.setattr(
        "deep_worker_cli.cli.agent_command.deep_worker_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda h=None: tmp_path,
    )
    return tmp_path


# ─── Loss-of-field tests (RED on the buggy dev) ──────────────────────────────


@pytest.mark.asyncio
async def test_restart_preserves_mcp_config_paths(_sandbox, monkeypatch):
    """First launch saved a path; --restart without flags must keep it."""
    _write_saved_config(_sandbox, "a", {"mcp_config_paths": ["/etc/mcp.json"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert calls, "AgentProcess should have been constructed"
    assert calls[0]["mcp_config_paths"] == [Path("/etc/mcp.json")], (
        "mcp_config_paths was wiped on --restart (bug reported 2026-08-14)"
    )


@pytest.mark.asyncio
async def test_restart_preserves_mcp_server_inline(_sandbox, monkeypatch):
    """Saved inline MCP server survives --restart without flags."""
    _write_saved_config(_sandbox, "a", {"mcp_server_inline": ["srv={\"cmd\":\"x\"}"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert calls[0]["mcp_server_inline"] == ["srv={\"cmd\":\"x\"}"]


@pytest.mark.asyncio
async def test_restart_preserves_turn_ending_tools(_sandbox, monkeypatch):
    """Saved --turn-ending-tool survives --restart without flags.

    Regression of the aa70703 feature. Same class of defect as
    ``mcp_config_paths``: the field is not in the step-5 copy-back list.
    """
    _write_saved_config(
        _sandbox,
        "a",
        {"turn_ending_tools": ["repo_graph__submit_fragments:document_closed"]},
    )
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert calls[0]["turn_ending_tool_specs"] == [
        "repo_graph__submit_fragments:document_closed"
    ], "turn_ending_tools was wiped on --restart (same class as mcp_config_paths)"


@pytest.mark.asyncio
async def test_restart_preserves_dev_tree(_sandbox, monkeypatch):
    """``dev_tree`` on ``effective`` survives ``restart_agent()``.

    dev_tree is subtle: even if ``restart_agent`` dropped the field on
    the way to ``run_agent``, ``build_launch_config`` re-derives it from
    ``_repo_root`` (the running code tree) — so a naive "read the
    persisted file" assertion would stay green while the field was in
    fact being wiped from the CLI dataflow. That masking is exactly the
    class of tautology test-theater that the reviewer flagged, so this
    test observes the field at the ``run_agent`` boundary directly: the
    argparse.Namespace that ``run_agent`` receives must carry
    ``dev_tree=True``, matching ``effective``, when the saved config
    said so.

    Verified mutation-worthy: replacing ``effective.dev_tree =
    saved.get("dev_tree", False)`` with a hard ``False`` in
    ``merge_restart_config`` turns THIS assertion red, whereas the
    previous "read the persisted file" form of this test stayed green
    against the same mutation (the ``_repo_root`` re-derivation masked
    it). See DESIGN_RESTART_PRESERVES_LAUNCH_CONFIG.md §8.
    """
    _write_saved_config(_sandbox, "a", {"dev_tree": True})
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        # Read what restart_agent actually handed to run_agent — the
        # single place where the CLI dataflow is observable before
        # build_launch_config re-derives dev_tree from _repo_root.
        captured["dev_tree"] = getattr(effective_ns, "dev_tree", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert captured.get("dev_tree") is True, (
        f"dev_tree was dropped between merge_restart_config and run_agent; "
        f"got {captured.get('dev_tree')!r} (expected True). This is the "
        f"same dataflow the mcp_config_paths / turn_ending_tools bug lived "
        f"in — dev_tree just happened to survive on disk because "
        f"build_launch_config re-derives it from _repo_root."
    )


@pytest.mark.asyncio
async def test_restart_preserves_web_search_policy(_sandbox, monkeypatch):
    """Saved ``web_search_policy`` survives --restart without flags.

    Fifth field of the same defect class: absent from the old step-5
    copy-back list, so ``build_launch_config(args)`` re-persisted the
    parser default ``"auto"`` and an operator's explicit ``enabled`` /
    ``disabled`` policy silently reverted on every restart. Observed on
    the persisted launch_config, which is what the next launch reads.
    """
    _write_saved_config(_sandbox, "a", {"web_search_policy": "enabled"})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert calls[0]["launch_config"]["web_search_policy"] == "enabled", (
        "web_search_policy reverted to the parser default on --restart "
        "(same copy-back omission as mcp_config_paths)"
    )


@pytest.mark.asyncio
async def test_restart_preserves_failover_chain(_sandbox, monkeypatch):
    """Saved ``failover_chain`` survives --restart without flags.

    Same defect class as ``web_search_policy``, with a sharper consequence:
    EVERY automatic failover goes through ``--restart``. A chain that reverted
    to ``"default"`` here would undo the operator's choice on the very first
    usage limit — the one moment the setting matters.
    """
    _write_saved_config(_sandbox, "a", {"failover_chain": "cheap"})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert calls[0]["launch_config"]["failover_chain"] == "cheap", (
        "failover_chain reverted to 'default' on --restart; an automatic "
        "failover would silently drop the operator's chain"
    )


@pytest.mark.asyncio
async def test_restart_flag_replaces_saved_failover_chain(_sandbox, monkeypatch):
    """An explicit ``--failover-chain`` on restart overrides the saved name."""
    _write_saved_config(_sandbox, "a", {"failover_chain": "cheap"})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    args = _restart_args("a")
    args.failover_chain = "premium"
    with patch("sys.argv", ["deep-worker", "--restart", "a", "--failover-chain", "premium"]):
        rc = await agent_command.restart_agent(args)

    assert rc == 0
    assert calls[0]["launch_config"]["failover_chain"] == "premium"


@pytest.mark.asyncio
async def test_restart_preserves_preferred_model_under_failover(
    _sandbox, monkeypatch
):
    """Saved ``preferred_model`` survives --restart in the FAILOVER shape.

    ``preferred_model`` is the sixth lost field, and it is the one whose
    loss is MASKED in the ordinary shape: when ``provider ==
    preferred_provider``, ``build_launch_config`` re-derives
    ``preferred_model`` from ``model`` (agent_command.py:735-740), so a
    test written that way stays green even while the field is being
    dropped from the dataflow — the same class of masking that made the
    first ``dev_tree`` test theater.

    The loss is only observable when ``provider != preferred_provider``,
    i.e. while a temporary TaskManager failover has moved the agent off
    its preferred provider. That is also the shape where the field
    MATTERS: ``preferred_model`` is what restore-on-idle uses to put the
    operator back on ``opus`` instead of the provider default.

    Verified against the pre-fix code path: building from the raw
    ``args`` namespace yields ``preferred_model=None`` here, versus
    ``"opus"`` from ``effective``.
    """
    _write_saved_config(
        _sandbox,
        "a",
        {
            # Temporary failover: running on deepseek, but the operator's
            # long-term preference is claude-code-sdk/opus.
            "provider": "deepseek",
            "model": "deepseek-chat",
            "model_provider": "deepseek",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "opus",
        },
    )
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    lc = calls[0]["launch_config"]
    assert lc["preferred_model"] == "opus", (
        f"preferred_model was wiped on --restart while failed over onto "
        f"{lc['provider']!r}; got {lc['preferred_model']!r}. Restore-on-idle "
        f"would then boot the preferred provider on its default model "
        f"instead of the operator's choice."
    )
    assert lc["preferred_provider"] == "claude-code-sdk"


# ─── CLI-add / CLI-remove semantics ──────────────────────────────────────────
# Owner: "Options specified at restart may be added or removed."
# The list flags already have wholesale-replace semantics; these tests
# guard that the fix does NOT accidentally make CLI values additive or
# make inheritance silently persistent when a fresh value is passed.


@pytest.mark.asyncio
async def test_restart_flag_replaces_saved_mcp_config_paths(_sandbox, monkeypatch):
    """--mcp-config on --restart REPLACES the saved list wholesale."""
    _write_saved_config(_sandbox, "a", {"mcp_config_paths": ["/old.json"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    with patch(
        "deep_worker_cli.cli.agent_command._explicit_cli_flags",
        return_value={"mcp_config_paths"},
    ):
        rc = await agent_command.restart_agent(
            _restart_args("a", mcp_config_paths=["/new.json"])
        )

    assert rc == 0
    assert calls[0]["mcp_config_paths"] == [Path("/new.json")]


@pytest.mark.asyncio
async def test_restart_flag_replaces_saved_turn_ending_tools(_sandbox, monkeypatch):
    """--turn-ending-tool on --restart REPLACES the saved list wholesale."""
    _write_saved_config(_sandbox, "a", {"turn_ending_tools": ["old_tool"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    with patch(
        "deep_worker_cli.cli.agent_command._explicit_cli_flags",
        return_value={"turn_ending_tools"},
    ):
        rc = await agent_command.restart_agent(
            _restart_args("a", turn_ending_tools=["new_tool"])
        )

    assert rc == 0
    assert calls[0]["turn_ending_tool_specs"] == ["new_tool"]


# ─── Incremental tool selection (--tools-add / --tools-remove) ───────────────


@pytest.mark.asyncio
async def test_restart_tools_add_layers_onto_the_saved_selection(
    _sandbox, monkeypatch
):
    """``--tools-add`` gains a tool WITHOUT restating the saved ones.

    This is the whole point of the flag: an agent confined to
    ``none,read_file`` must be able to gain ``grep`` without the operator
    retyping ``read_file``. Retyping is not merely inconvenient — omitting
    a name silently narrows the agent, which is the failure the
    unknown-name check exists to prevent.
    """
    _write_saved_config(_sandbox, "a", {"tools": ["none", "read_file"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a", tools_add=["grep"]))

    assert rc == 0
    spec = calls[0]["launch_config"]["tools"]
    selection = parse_tool_selection(spec)
    assert selection.permits("read_file"), (
        f"the saved allow-list was lost by --tools-add; spec={spec}"
    )
    assert selection.permits("grep"), f"--tools-add did not add; spec={spec}"
    assert not selection.permits("execute"), (
        f"--tools-add widened the agent beyond what was asked; spec={spec}"
    )


@pytest.mark.asyncio
async def test_restart_tools_remove_layers_onto_the_saved_selection(
    _sandbox, monkeypatch
):
    """``--tools-remove`` drops one tool and leaves the rest alone."""
    _write_saved_config(_sandbox, "a", {"tools": []})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(
        _restart_args("a", tools_remove=["execute"])
    )

    assert rc == 0
    selection = parse_tool_selection(calls[0]["launch_config"]["tools"])
    assert not selection.permits("execute")
    assert selection.permits("read_file")


@pytest.mark.asyncio
async def test_restart_pattern_removal_keeps_the_specific_exception(
    _sandbox, monkeypatch
):
    """``--tools-rm 'serena__*' --tools-add serena__find_symbol``.

    The owner's case: drop a family, keep one member. The specific rule
    must beat the pattern through the real CLI dataflow, not just in the
    parser unit tests.
    """
    _write_saved_config(_sandbox, "a", {"tools": []})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(
        _restart_args(
            "a",
            tools_add=["serena__find_symbol"],
            tools_remove=["serena__*"],
        )
    )

    assert rc == 0
    selection = parse_tool_selection(calls[0]["launch_config"]["tools"])
    assert selection.permits("serena__find_symbol")
    assert not selection.permits("serena__rename_symbol")
    assert selection.permits("read_file")


@pytest.mark.asyncio
async def test_restart_without_flags_preserves_an_incremental_spec(
    _sandbox, monkeypatch
):
    """A spec containing increments survives a bare ``--restart``.

    The increments live inside the stored spec, so the restart path must
    replay them rather than treat ``+name``/``~name`` as unknown tools.
    """
    _write_saved_config(
        _sandbox, "a", {"tools": ["~serena__*", "+serena__find_symbol"]}
    )
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    selection = parse_tool_selection(calls[0]["launch_config"]["tools"])
    assert selection.permits("serena__find_symbol")
    assert not selection.permits("serena__rename_symbol")


@pytest.mark.asyncio
async def test_two_restarts_with_the_same_increment_do_not_accumulate(
    _sandbox, monkeypatch
):
    """Repeating ``--tools-add grep`` must not grow the stored spec.

    A layer that re-applied itself on every start would make the spec
    grow without bound and would resurrect a rule the operator later
    reversed. The second restart writes through the spy's persistence, so
    this observes the real restart-over-restart dataflow.
    """
    _write_saved_config(_sandbox, "a", {"tools": ["none", "read_file"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch, persist_root=_sandbox)

    rc = await agent_command.restart_agent(_restart_args("a", tools_add=["grep"]))
    assert rc == 0
    first = calls[0]["launch_config"]["tools"]

    rc = await agent_command.restart_agent(_restart_args("a", tools_add=["grep"]))
    assert rc == 0
    second = calls[1]["launch_config"]["tools"]

    assert first == second, (
        f"the increment accumulated across restarts: {first} → {second}"
    )
    selection = parse_tool_selection(second)
    assert selection.permits("grep")
    assert selection.permits("read_file")


@pytest.mark.asyncio
async def test_a_later_restart_can_reverse_an_earlier_increment(
    _sandbox, monkeypatch
):
    """``--tools-rm grep`` after ``--tools-add grep`` actually removes it."""
    _write_saved_config(_sandbox, "a", {"tools": ["+grep"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch, persist_root=_sandbox)

    rc = await agent_command.restart_agent(
        _restart_args("a", tools_remove=["grep"])
    )

    assert rc == 0
    spec = calls[0]["launch_config"]["tools"]
    assert not parse_tool_selection(spec).permits("grep"), (
        f"the superseded add rule survived and kept grep alive; spec={spec}"
    )


# ─── Round-trip: two consecutive restarts (the reported loop) ──────────────


@pytest.mark.asyncio
async def test_two_consecutive_restarts_preserve_mcp_config_paths(
    _sandbox, monkeypatch
):
    """First launch → save → restart 1 (no flags) → restart 2 (no flags).

    The exact loop the reporting agent got stuck in twice. After each restart the
    persisted ``mcp_config_paths`` must equal the original — not just on
    the first restart but on every subsequent one.

    The spy is installed with ``persist_root`` so restart-1 actually
    REWRITES ``session_metadata.json`` from the launch_config it built,
    and restart-2 therefore reads restart-1's output rather than the
    original seed. With a no-op ``start()`` this test was vacuous: both
    restarts would re-read the same untouched seed file and the second
    assertion could not fail independently of the first (flagged in
    review, 2026-08-14).
    """
    _write_saved_config(_sandbox, "a", {"mcp_config_paths": ["/etc/mcp.json"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch, persist_root=_sandbox)

    rc1 = await agent_command.restart_agent(_restart_args("a"))
    # First restart wrote a new session_metadata.json via run_agent's
    # build_launch_config. Read it back and assert it survived.
    persisted_after_1 = json.loads(
        (_sandbox / "agents" / "a" / "session_metadata.json").read_text()
    )["launch_config"]["mcp_config_paths"]
    # Prove the file really was rewritten by restart 1 (otherwise restart 2
    # would just be re-reading the seed and this test would be vacuous):
    # build_launch_config stamps a fresh pid, the seed's was 1234.
    meta_after_1 = json.loads(
        (_sandbox / "agents" / "a" / "session_metadata.json").read_text()
    )["launch_config"]
    assert meta_after_1.get("pid") != 1234, (
        "restart 1 did not rewrite session_metadata.json — restart 2 would "
        "re-read the seed and this round-trip test would prove nothing"
    )

    rc2 = await agent_command.restart_agent(_restart_args("a"))
    persisted_after_2 = json.loads(
        (_sandbox / "agents" / "a" / "session_metadata.json").read_text()
    )["launch_config"]["mcp_config_paths"]

    assert rc1 == 0 and rc2 == 0
    assert persisted_after_1 == ["/etc/mcp.json"], (
        f"first restart wiped the path; got {persisted_after_1!r}"
    )
    assert persisted_after_2 == ["/etc/mcp.json"], (
        f"second restart wiped what the first preserved; got {persisted_after_2!r}"
    )
    # And AgentProcess got the right list every time.
    assert calls[0]["mcp_config_paths"] == [Path("/etc/mcp.json")]
    assert calls[1]["mcp_config_paths"] == [Path("/etc/mcp.json")]


# ─── --clean keeps its one-shot semantics under Option A ─────────────────────


@pytest.mark.asyncio
async def test_restart_clean_runs_once_and_is_not_persisted(_sandbox, monkeypatch):
    """``--restart --clean`` cleans exactly once and persists ``clean=False``.

    Under the old copy-back, ``args.clean`` was left untouched and
    ``build_launch_config(args)`` therefore saw the CLI's own ``clean``
    value. Option A passes ``effective`` — whose ``clean`` was set True by
    ``merge_restart_config`` so step 4 fires — straight to ``run_agent``,
    so without the explicit ``effective.clean = False`` reset the flag
    would be PERSISTED and every subsequent restart would wipe the
    agent's history. This pins both halves: cleanup called exactly once,
    and the rebuilt launch_config carries ``clean=False``.
    """
    _write_saved_config(_sandbox, "a", {"mcp_config_paths": ["/etc/mcp.json"]})
    _no_registered_agent(monkeypatch)
    calls = _install_agent_process_spy(monkeypatch)

    cleaned: list[str] = []

    async def _spy_clean(agent_name):
        cleaned.append(agent_name)

    monkeypatch.setattr(
        "deep_worker_cli.sessions.clean_agent_history", _spy_clean
    )

    with patch(
        "deep_worker_cli.cli.agent_command._explicit_cli_flags",
        return_value={"clean"},
    ):
        rc = await agent_command.restart_agent(_restart_args("a", clean=True))

    assert rc == 0
    assert cleaned == ["a"], (
        f"--clean must trigger exactly one history cleanup; got {cleaned!r}"
    )
    assert calls[0]["launch_config"]["clean"] is False, (
        "clean was persisted as True — the next --restart would wipe the "
        "agent's history again without the operator asking (one-shot "
        "semantics broken)"
    )
    # And the preservation guarantee still holds on the clean path.
    assert calls[0]["mcp_config_paths"] == [Path("/etc/mcp.json")]


# ─── Class-of-bug guards (the point of Option A over Option B) ───────────────


def test_no_field_by_field_copy_back_in_restart_agent():
    """``restart_agent`` must NOT contain a hand-written
    ``args.X = effective.X`` copy-back list.

    That list is the class-of-bug: every new persisted field a future
    contributor adds needs another line here, and one omission silently
    wipes the field on --restart. Option A removes the list; this test
    keeps it removed against a future "let me restore clarity" refactor.
    """
    tree = ast.parse(inspect.getsource(agent_command.restart_agent))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "args"
        ):
            continue
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "effective"
        ):
            offenders.append(node.targets[0].attr)
    assert not offenders, (
        f"restart_agent still copies fields from `effective` back into "
        f"`args` — the class-of-bug the fix removed. Offenders: "
        f"{offenders!r}. Delete them and call run_agent(effective) instead."
    )


def _args_attrs_read_by_build_launch_config() -> set[str]:
    """Every attribute of ``args`` that ``build_launch_config`` reads.

    Covers both the ``args.X`` and ``getattr(args, "X", …)`` forms.
    """
    tree = ast.parse(inspect.getsource(agent_command.build_launch_config))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            names.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def _effective_attrs_assigned_by_merge_restart_config() -> set[str]:
    """Every attribute of ``effective`` that ``merge_restart_config`` assigns."""
    tree = ast.parse(inspect.getsource(agent_command.merge_restart_config))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "effective"
            ):
                names.add(target.attr)
    return names


def test_every_build_launch_config_input_is_handled_by_merge_restart_config():
    """Every ``args`` attribute ``build_launch_config`` reads must be
    explicitly assigned on ``effective`` by ``merge_restart_config``.

    SCOPE — what this guard does and does not prove:

    It proves a STRUCTURAL property: no field can be added to
    ``build_launch_config`` and silently inherit whatever the bare CLI
    namespace happened to hold, because ``merge_restart_config`` is
    forced to make a deliberate decision about it (restore from saved,
    or explicitly reset — ``clean`` is assigned ``False`` on purpose and
    so is covered). It is NOT a hand-written list of known fields, so
    unlike the earlier version of this test it does not silently pass
    over a field its author forgot: a new ``args`` read in
    ``build_launch_config`` with no corresponding ``effective``
    assignment fails this test immediately.

    It does NOT prove the assignment is SEMANTICALLY right (that the
    value restored equals the saved one). That is what the per-field
    behavioral ``restart_agent`` tests above do — one per currently
    persisted field: mcp_config_paths, mcp_server_inline,
    turn_ending_tools, dev_tree, web_search_policy, preferred_model.
    The two layers are complementary; neither is claimed to subsume the
    other.

    Mutation-verified: adding a hypothetical ``"new_flag":
    getattr(args, "new_flag", None)`` entry to ``build_launch_config``
    without touching ``merge_restart_config`` turns this red with
    ``uncovered == ['new_flag']``.
    """
    read_names = _args_attrs_read_by_build_launch_config()
    assigned = _effective_attrs_assigned_by_merge_restart_config()

    assert read_names, "AST scan found no args reads — the scan itself broke"

    uncovered = sorted(read_names - assigned)
    assert not uncovered, (
        f"build_launch_config reads these args attributes that "
        f"merge_restart_config never assigns on `effective`: {uncovered}. "
        f"On --restart they take whatever the bare CLI namespace holds — "
        f"i.e. the parser default — which is exactly how mcp_config_paths, "
        f"turn_ending_tools, web_search_policy and preferred_model got "
        f"wiped for a reported agent (2026-08-14). Restore each from the saved "
        f"launch_config in merge_restart_config, or assign it an explicit "
        f"deliberate value (as `clean` does), and add a behavioral "
        f"restart_agent test for it."
    )
