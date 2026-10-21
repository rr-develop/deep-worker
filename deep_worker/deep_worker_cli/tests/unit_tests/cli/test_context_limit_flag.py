"""CLI + persist/restore tests for ``--context_limit`` / ``-l`` and ``-r``.

Design: docs/DESIGN_CLI_CONTEXT_LIMIT_FLAG.md.

These tests are the RED-first pins for hops 1-3 (parse, persist, restore)
and the alias/reset/failover mechanism pins required by the design
sign-off. Live-path trigger/keep semantics live in deepagents
(``test_context_limit_summarization.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.cli.agent_command import (
    _PARSER_DEFAULTS,
    _explicit_cli_flags,
    build_launch_config,
    create_parser,
    merge_restart_config,
)


def _parser() -> argparse.ArgumentParser:
    return create_parser()


def _saved(*, context_limit: int | None = 400_000) -> dict:
    return {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": "/tmp",
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
        "context_limit": context_limit,
    }


def _bare_restart_args(name: str = "my_agent") -> argparse.Namespace:
    """Namespace of a bare ``deep-worker --restart NAME`` (no --context_limit)."""
    return _parser().parse_args([name, "--restart"])


# ---------------------------------------------------------------------------
# 1 / 2 — grammar
# ---------------------------------------------------------------------------


class TestParseContextLimitGrammar:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("200000", 200_000),
            ("200k", 200_000),
            ("200K", 200_000),
            ("1M", 1_000_000),
            ("1m", 1_000_000),
            # ``default`` returns the clear sentinel — distinct from
            # the bare-restart case where ``args.context_limit is None``
            # simply because no ``-l`` was passed. Merged by
            # ``merge_restart_config``. See design §C2 + the
            # agent_a / 2026-08-18 / sentinel discussion in
            # ``test_restart_preserves_context_limit.py``.
            ("default", "_SENTINEL"),
            ("DEFAULT", "_SENTINEL"),
        ],
    )
    def test_accepted_values(self, raw: str, expected) -> None:
        from deep_worker_cli.cli.agent_command import (
            _CONTEXT_LIMIT_CLEAR_SENTINEL,
            parse_context_limit,
        )

        actual = parse_context_limit(raw)
        if expected == "_SENTINEL":
            assert actual == _CONTEXT_LIMIT_CLEAR_SENTINEL
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        "raw",
        ["0", "-5", "1.5k", "200g", "abc", "", "1.5", "200.0", "k200", "200 k"],
    )
    def test_rejected_values(self, raw: str) -> None:
        from deep_worker_cli.cli.agent_command import parse_context_limit

        with pytest.raises(argparse.ArgumentTypeError):
            parse_context_limit(raw)

    def test_parser_accepts_suffix_and_default(self) -> None:
        from deep_worker_cli.cli.agent_command import _CONTEXT_LIMIT_CLEAR_SENTINEL
        parser = _parser()
        assert parser.parse_args(["a", "--context_limit", "200k"]).context_limit == 200_000
        assert parser.parse_args(["a", "--context_limit", "1M"]).context_limit == 1_000_000
        assert parser.parse_args(["a", "--context_limit", "200000"]).context_limit == 200_000
        assert (
            parser.parse_args(["a", "--context_limit", "default"]).context_limit
            == _CONTEXT_LIMIT_CLEAR_SENTINEL
        )

    def test_parser_rejects_bad_values(self) -> None:
        parser = _parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["a", "--context_limit", "1.5k"])
        with pytest.raises(SystemExit):
            parser.parse_args(["a", "--context_limit", "200g"])
        with pytest.raises(SystemExit):
            parser.parse_args(["a", "--context_limit", "0"])


# ---------------------------------------------------------------------------
# 10 — aliases on BOTH parsers (the -p / --preferred-provider class of bug)
# ---------------------------------------------------------------------------


class TestAliasSafety:
    def test_short_l_parses_to_context_limit(self) -> None:
        args = _parser().parse_args(["-l", "200k"])
        assert args.context_limit == 200_000

    def test_ls_is_not_context_limit(self) -> None:
        args = _parser().parse_args(["-ls", "x"])
        assert args.context_limit is None
        assert args.list_threads is True

    def test_short_r_is_restart(self) -> None:
        args = _parser().parse_args(["my_agent", "-r"])
        assert args.restart is True

    def test_rm_is_not_restart(self) -> None:
        args = _parser().parse_args(["-rm", "thread-1"])
        assert args.restart is False
        assert args.remove_thread == "thread-1"

    def test_detector_l_marks_context_limit_explicit(self) -> None:
        assert "context_limit" in _explicit_cli_flags(["my_agent", "-l", "200k"])

    def test_detector_long_form_marks_explicit(self) -> None:
        assert "context_limit" in _explicit_cli_flags(
            ["my_agent", "--context_limit", "400k"]
        )

    def test_detector_ls_is_not_context_limit(self) -> None:
        explicit = _explicit_cli_flags(["-ls", "x"])
        assert "context_limit" not in explicit

    def test_detector_absent_when_not_passed(self) -> None:
        assert "context_limit" not in _explicit_cli_flags(["my_agent", "--restart"])


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------


class TestBuildLaunchConfigPersistsContextLimit:
    def test_persists_int_tokens(self) -> None:
        args = _parser().parse_args(["a", "--context_limit", "400k"])
        with patch.dict("os.environ", {}, clear=False):
            cfg = build_launch_config(args)
        assert cfg["context_limit"] == 400_000

    def test_persists_none_for_default(self) -> None:
        args = _parser().parse_args(["a", "--context_limit", "default"])
        with patch.dict("os.environ", {}, clear=False):
            cfg = build_launch_config(args)
        assert cfg["context_limit"] is None

    def test_omitted_flag_persists_none(self) -> None:
        args = _parser().parse_args(["a"])
        with patch.dict("os.environ", {}, clear=False):
            cfg = build_launch_config(args)
        assert cfg["context_limit"] is None
        assert "context_limit" in cfg


# ---------------------------------------------------------------------------
# 7 — merge_restart_config mechanism pin (not just the restart effect)
# ---------------------------------------------------------------------------


class TestMergeRestartConfigContextLimit:
    def test_explicit_override_wins_even_when_value_is_none(self) -> None:
        """``explicit={"context_limit"}`` + no-flag args (None) → override applies → None."""
        saved = _saved(context_limit=400_000)
        args = _bare_restart_args()
        assert args.context_limit is None
        merged = merge_restart_config(saved, args, explicit={"context_limit"})
        assert merged.context_limit is None

    def test_empty_explicit_restores_saved(self) -> None:
        """``explicit=set()`` → saved 400000 is applied."""
        saved = _saved(context_limit=400_000)
        args = _bare_restart_args()
        merged = merge_restart_config(saved, args, explicit=set())
        assert merged.context_limit == 400_000

    def test_assigns_effective_from_saved_get(self) -> None:
        """The restore path must actually read saved['context_limit'].

        The current implementation reads the field into a local
        (``saved_cl = saved.get("context_limit")``) and then either
        assigns it to ``effective.context_limit`` (bare restart) or
        ignores it (explicit override / explicit clear). Either way,
        the saved value MUST be reachable through ``saved.get(...)``
        on the restore code path — that's the regression net for
        "saved value silently dropped after refactor".
        """
        here = Path(__file__).resolve()
        agent_command_path = here.parents[3] / "deep_worker_cli" / "cli" / "agent_command.py"
        text = agent_command_path.read_text()
        assert (
            'saved.get("context_limit")' in text
        ), (
            "merge_restart_config must read saved['context_limit'] "
            "to preserve the operator's last setting across restarts. "
            "The persist/restore guard checks that the read is wired; "
            "see test_restart_preserves_context_limit.py for the "
            "behavioural pin."
        )

    def test_parser_defaults_include_context_limit(self) -> None:
        assert "context_limit" in _PARSER_DEFAULTS
        assert _PARSER_DEFAULTS["context_limit"] is None


# ---------------------------------------------------------------------------
# 8 — failover shape: --restart NAME --provider X does not pass the flag
# ---------------------------------------------------------------------------


class TestFailoverPreservesContextLimit:
    def test_restart_with_provider_only_restores_saved_limit(self) -> None:
        argv = ["my_agent", "--restart", "--provider", "grok", "--no-update-preferred"]
        args = _parser().parse_args(argv)
        explicit = _explicit_cli_flags(argv)
        assert "context_limit" not in explicit
        merged = merge_restart_config(
            _saved(context_limit=400_000), args, explicit=explicit
        )
        assert merged.context_limit == 400_000


# ---------------------------------------------------------------------------
# 9 — --context_limit default / -l default clears a saved value
# ---------------------------------------------------------------------------


class TestResetClearsSavedValue:
    def test_long_form_default_clears_saved(self) -> None:
        argv = ["my_agent", "--restart", "--context_limit", "default"]
        args = _parser().parse_args(argv)
        explicit = _explicit_cli_flags(argv)
        assert "context_limit" in explicit
        merged = merge_restart_config(
            _saved(context_limit=400_000), args, explicit=explicit
        )
        assert merged.context_limit is None

    def test_short_l_default_clears_saved(self) -> None:
        argv = ["my_agent", "--restart", "-l", "default"]
        args = _parser().parse_args(argv)
        explicit = _explicit_cli_flags(argv)
        assert "context_limit" in explicit
        merged = merge_restart_config(
            _saved(context_limit=400_000), args, explicit=explicit
        )
        assert merged.context_limit is None


# ---------------------------------------------------------------------------
# wiring: create_cli_agent + AgentProcess start()
# ---------------------------------------------------------------------------


class TestCreateCliAgentForwardsContextLimit:
    def test_kwarg_exists_and_defaults_to_none(self) -> None:
        import inspect

        from deep_worker_cli.agent import create_cli_agent

        sig = inspect.signature(create_cli_agent)
        assert "context_limit" in sig.parameters
        assert sig.parameters["context_limit"].default is None

    def test_forwards_to_create_deep_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import deepagents as da_mod
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from deep_worker_cli.agent import create_cli_agent

        captured: dict = {}

        def _fake_create_deep_agent(**kwargs: object):
            captured.update(kwargs)

            class _Fake:
                def with_config(self, _cfg: object):
                    return self

            return _Fake()

        monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)
        create_cli_agent(
            model=FakeListChatModel(responses=["ok"]),
            assistant_id="ctx_limit_fwd",
            tools=[],
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
            context_limit=400_000,
        )
        assert captured.get("context_limit") == 400_000


class TestAgentProcessForwardsLaunchConfigContextLimit:
    @pytest.mark.asyncio
    async def test_start_passes_launch_config_context_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration check that the launch_config ``context_limit``
        value flows through to ``create_cli_agent``.

        For the auto-clamp behavior itself, see the unit tests in
        ``tests/unit_tests/context_limit/test_provider_max.py`` —
        those exercise the SDK-sourced lookup table directly
        without spawning the full agent process and are the
        authoritative pin for the clamp logic.

        The model + provider pair used here is one that doesn't
        clamp (deepseek-chat is silent passthrough because the
        LangChain ``langchain_deepseek`` SDK does not ship a
        ``data._profiles`` module as of 2026-08-19). The test
        verifies the end-to-end flow:

          -l 400_000 → agent_process.start() → warn + clamp →
          create_cli_agent(..., context_limit=400_000)

        The 400_000 value passes through unchanged because
        provider_max_for("deepseek", "deepseek-chat") returns
        None (silent passthrough). The point is that the value
        reaches ``create_cli_agent`` intact — proving the flow
        works. The actual clamp is tested in test_provider_max.py.

        CI note: this test uses the ``deepseek`` provider because
        DEEPSEEK_API_KEY is the only API key available in CI; an
        earlier revision used ``openai``/``gpt-4`` but that
        requires OPENAI_API_KEY, which is not set in
        ``unittest.yml``. Using ``claude-code-sdk`` was also
        rejected because it tries to spawn the actual Claude
        Code subprocess during model creation, which blocks
        before ``create_cli_agent`` is reached.
        """
        from unittest.mock import MagicMock

        from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
        from deep_worker_cli.process.agent_process import AgentProcess
        from tests.unit_tests.test_agent_checkpointer import (
            _run_start_until_create_cli_agent,
        )

        working_dir = tmp_path / "wd"
        working_dir.mkdir()
        process = AgentProcess(
            name="ctx_limit_ap",
            config=AgentConfig(model="deepseek-chat", working_dir=working_dir),
            rpyc_port=False,
            data_dir=tmp_path / "data",
            launch_config={
                "provider": "deepseek",
                "model": "deepseek-chat",
                "context_limit": 400_000,
            },
        )
        with patch(
            "deep_worker_cli.agent.create_cli_agent"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            await _run_start_until_create_cli_agent(process, mock_create)
        # The 400_000 value passes through unchanged because
        # deepseek-chat is silent passthrough (no SDK profile data).
        # The provider-max clamp is exhaustively exercised in
        # test_provider_max.py for the SDK-backed case (gpt-4 → 8_192,
        # claude-haiku-4-5 → 200_000, etc.).
        assert mock_create.call_args.kwargs.get("context_limit") == 400_000

    @pytest.mark.asyncio
    async def test_start_family_alias_clamps_via_picker_resolver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Family alias on ``claude-code`` resolves via the picker
        and the resolved SDK value drives the boot-time clamp.

        The picker subprocess is NOT spawned in tests — we mock
        ``resolve_provider_max`` directly. This test is the
        end-to-end check that ``agent_process.start()`` calls the
        async resolver (not the sync helper) and applies the
        resolved cap.
        """
        from unittest.mock import AsyncMock, MagicMock

        from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
        from deep_worker_cli.process.agent_process import AgentProcess
        from tests.unit_tests.test_agent_checkpointer import (
            _run_start_until_create_cli_agent,
        )

        # Mock resolve_provider_max to return 200_000 for fable.
        # The test passes ``model="fable"`` through ``create_model``
        # which auto-detects the provider (deepseek in CI because
        # only DEEPSEEK_API_KEY is set). The fake resolver returns
        # the SDK value regardless of provider so the clamp
        # triggers — what we're testing here is that the boot-time
        # flow uses the async resolver and clamps correctly.
        async def fake_resolve(provider: str, model: str) -> int | None:
            if model == "fable":
                return 200_000
            return None

        # Patch the source-module attribute. ``agent_process.start()``
        # imports ``resolve_provider_max`` INSIDE the function via
        # ``from ... import ...``, so the import re-reads the source
        # module attribute on each call and picks up the patched
        # value here.
        monkeypatch.setattr(
            "deep_worker_cli.context_limit.provider_max.resolve_provider_max",
            fake_resolve,
        )

        working_dir = tmp_path / "wd"
        working_dir.mkdir()
        process = AgentProcess(
            name="ctx_limit_fable",
            config=AgentConfig(model="fable", working_dir=working_dir),
            rpyc_port=False,
            data_dir=tmp_path / "data",
            launch_config={
                "provider": "claude-code-sdk",
                "model": "fable",
                "context_limit": 400_000,
            },
        )
        with patch(
            "deep_worker_cli.agent.create_cli_agent"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            await _run_start_until_create_cli_agent(process, mock_create)

        # The 400_000 value gets clamped to 200_000 (resolved cap).
        assert mock_create.call_args.kwargs.get("context_limit") == 200_000

    @pytest.mark.asyncio
    async def test_start_family_alias_picker_fails_passes_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the picker resolver returns ``None`` (CLI missing,
        timeout, no ``REPLACES`` marker), the operator's ``-l``
        value passes through unchanged — per the operator
        contract: "If you don't know and can't get it via the API,
        better not do any cap at all besides the operator's -l."
        """
        from unittest.mock import MagicMock

        from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
        from deep_worker_cli.process.agent_process import AgentProcess
        from tests.unit_tests.test_agent_checkpointer import (
            _run_start_until_create_cli_agent,
        )

        async def fake_resolve(provider: str, model: str) -> int | None:
            return None  # picker unavailable

        # Patch the source-module attribute. ``agent_process.start()``
        # imports ``resolve_provider_max`` INSIDE the function via
        # ``from ... import ...``, so the import re-reads the source
        # module attribute on each call and picks up the patched
        # value here.
        monkeypatch.setattr(
            "deep_worker_cli.context_limit.provider_max.resolve_provider_max",
            fake_resolve,
        )

        working_dir = tmp_path / "wd"
        working_dir.mkdir()
        process = AgentProcess(
            name="ctx_limit_fable_passthrough",
            config=AgentConfig(model="fable", working_dir=working_dir),
            rpyc_port=False,
            data_dir=tmp_path / "data",
            launch_config={
                "provider": "claude-code-sdk",
                "model": "fable",
                "context_limit": 400_000,
            },
        )
        with patch(
            "deep_worker_cli.agent.create_cli_agent"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())
            await _run_start_until_create_cli_agent(process, mock_create)

        # 400_000 passes through unchanged (silent passthrough).
        assert mock_create.call_args.kwargs.get("context_limit") == 400_000
