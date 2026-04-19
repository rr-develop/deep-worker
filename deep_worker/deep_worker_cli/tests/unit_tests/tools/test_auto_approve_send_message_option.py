"""Tests for the --auto-approve-send-message option.

The option allows running an agent WITHOUT `-sp` (i.e. approvals are
still required for other side-effect tools like `execute`, `edit_file`,
`write_file`, etc.) while specifically auto-approving
`send_message_to_agent`.

This is useful for agents that are expected to cooperate with other
agents heavily but whose file/shell side effects still require human
confirmation.

Behaviour matrix:

    auto_approve (-sp) | auto_approve_send_message | send_message_to_agent HITL?
    -------------------|---------------------------|-------------------------
    False              | False                     | YES (gated — default)
    False              | True                      | NO  (auto-approved)
    True               | False                     | NO  (everything auto)
    True               | True                      | NO  (everything auto)

Other tools (execute, edit_file, ...) are NOT affected by
auto_approve_send_message — only `-sp` auto-approves them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


# ===========================================================================
# _add_interrupt_on() honours the flag
# ===========================================================================


class TestAddInterruptOn:
    """Unit-level tests for _add_interrupt_on(auto_approve_send_message=...)."""

    def test_default_includes_send_message_to_agent(self) -> None:
        """Without the flag, the tool is HITL-gated (current behaviour)."""
        from deep_worker_cli.agent import _add_interrupt_on

        cfg = _add_interrupt_on()
        assert "send_message_to_agent" in cfg

    def test_flag_false_still_includes_send_message_to_agent(self) -> None:
        """Explicit False is equivalent to default."""
        from deep_worker_cli.agent import _add_interrupt_on

        cfg = _add_interrupt_on(auto_approve_send_message=False)
        assert "send_message_to_agent" in cfg

    def test_flag_true_omits_send_message_to_agent(self) -> None:
        """When the flag is True, the tool is NOT in interrupt_on,
        which means LangGraph will run it without asking for approval.
        """
        from deep_worker_cli.agent import _add_interrupt_on

        cfg = _add_interrupt_on(auto_approve_send_message=True)
        assert "send_message_to_agent" not in cfg

    def test_flag_true_does_not_affect_other_tools(self) -> None:
        """Other side-effect tools remain gated — only send_message_to_agent
        is opted out."""
        from deep_worker_cli.agent import _add_interrupt_on

        cfg = _add_interrupt_on(auto_approve_send_message=True)
        # All the other gated tools remain
        for expected in (
            "execute",
            "write_file",
            "edit_file",
            "web_search",
            "fetch_url",
            "task",
        ):
            assert expected in cfg, (
                f"Tool '{expected}' must still be gated when only "
                "auto_approve_send_message=True"
            )


# ===========================================================================
# create_cli_agent() plumbing
# ===========================================================================


class TestCreateCliAgentPlumbing:
    """create_cli_agent() must accept and honour auto_approve_send_message."""

    def test_create_cli_agent_accepts_auto_approve_send_message_kwarg(
        self,
    ) -> None:
        """The kwarg must exist in the signature. We don't actually
        invoke the agent here — just confirm the parameter is exposed."""
        import inspect

        from deep_worker_cli.agent import create_cli_agent

        sig = inspect.signature(create_cli_agent)
        assert "auto_approve_send_message" in sig.parameters, (
            "create_cli_agent() must accept auto_approve_send_message kwarg"
        )
        # Default must be False to preserve existing behaviour.
        assert sig.parameters["auto_approve_send_message"].default is False


# ===========================================================================
# AgentProcess plumbing
# ===========================================================================


class TestAgentProcessPlumbing:
    """AgentProcess stores the flag and forwards it to create_cli_agent."""

    def test_agent_process_accepts_auto_approve_send_message(self) -> None:
        """AgentProcess.__init__ must accept auto_approve_send_message."""
        import inspect

        from deep_worker_cli.process.agent_process import AgentProcess

        sig = inspect.signature(AgentProcess.__init__)
        assert "auto_approve_send_message" in sig.parameters, (
            "AgentProcess.__init__ must accept auto_approve_send_message kwarg"
        )
        assert sig.parameters["auto_approve_send_message"].default is False

    def test_agent_process_stores_flag(self, tmp_path: Path) -> None:
        from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
        from deep_worker_cli.process.agent_process import AgentProcess

        working_dir = tmp_path / "wd"
        working_dir.mkdir()
        config = AgentConfig(
            model="claude-sonnet-4-5-20250929",
            working_dir=working_dir,
        )

        p = AgentProcess(
            name="test_asm_storage",
            config=config,
            rpyc_port=False,
            auto_approve_send_message=True,
        )
        assert p.auto_approve_send_message is True

        p2 = AgentProcess(
            name="test_asm_storage_default",
            config=config,
            rpyc_port=False,
        )
        assert p2.auto_approve_send_message is False


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCLIParser:
    """The --auto-approve-send-message / -asm flag is accepted by the
    deep-worker CLI parser and defaults to False."""

    def test_parser_defines_auto_approve_send_message(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["myagent"])
        assert hasattr(args, "auto_approve_send_message")
        assert args.auto_approve_send_message is False

    def test_parser_accepts_long_form(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["myagent", "--auto-approve-send-message"])
        assert args.auto_approve_send_message is True

    def test_parser_accepts_short_form(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["myagent", "-asm"])
        assert args.auto_approve_send_message is True

    def test_parser_default_still_false_with_other_flags(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["myagent", "-sp", "-tm"])
        assert args.auto_approve_send_message is False


# ===========================================================================
# launch_config persistence (for --restart)
# ===========================================================================


class TestLaunchConfigPersistence:
    """--auto-approve-send-message must be persisted in launch_config
    so that --restart preserves it."""

    def test_build_launch_config_includes_flag(self) -> None:
        from deep_worker_cli.cli.agent_command import build_launch_config

        args = argparse.Namespace(
            provider="deepseek",
            model="deepseek-chat",
            dir=Path("/tmp"),
            skip_permissions=False,
            task_manager=False,
            memory_only=False,
            use_sqlite=False,
            clean=False,
            auto_approve_send_message=True,
        )
        cfg = build_launch_config(args)
        assert cfg["auto_approve_send_message"] is True

    def test_build_launch_config_flag_defaults_false(self) -> None:
        from deep_worker_cli.cli.agent_command import build_launch_config

        args = argparse.Namespace(
            provider="deepseek",
            model="deepseek-chat",
            dir=Path("/tmp"),
            skip_permissions=False,
            task_manager=False,
            memory_only=False,
            use_sqlite=False,
            clean=False,
            auto_approve_send_message=False,
        )
        cfg = build_launch_config(args)
        assert cfg["auto_approve_send_message"] is False

    def test_merge_restart_config_inherits_flag_from_saved(self) -> None:
        """If saved launch_config had auto_approve_send_message=True and
        CLI did not override it, effective.auto_approve_send_message
        must be True."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        saved = {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "working_dir": "/tmp",
            "skip_permissions": False,
            "task_manager": False,
            "memory_only": False,
            "use_sqlite": False,
            "auto_approve_send_message": True,
        }
        args = argparse.Namespace(
            provider="deepseek",
            model=None,
            dir=None,
            skip_permissions=False,
            task_manager=False,
            clean=False,
            memory_only=False,
            use_sqlite=False,
            auto_approve_send_message=False,  # parser default
        )
        effective = merge_restart_config(saved, args)
        assert effective.auto_approve_send_message is True

    def test_merge_restart_config_cli_overrides_saved(self) -> None:
        """Explicit --auto-approve-send-message on the CLI must override
        the saved value (even if the saved value is False)."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        saved = {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "working_dir": "/tmp",
            "skip_permissions": False,
            "task_manager": False,
            "memory_only": False,
            "use_sqlite": False,
            "auto_approve_send_message": False,
        }
        args = argparse.Namespace(
            provider="deepseek",
            model=None,
            dir=None,
            skip_permissions=False,
            task_manager=False,
            clean=False,
            memory_only=False,
            use_sqlite=False,
            auto_approve_send_message=True,  # user explicitly asked for it
        )
        effective = merge_restart_config(saved, args)
        assert effective.auto_approve_send_message is True


# ===========================================================================
# End-to-end behavioural semantics (via create_cli_agent wiring)
# ===========================================================================


class TestEndToEndSemantics:
    """Verify the dispatcher logic in create_cli_agent correctly combines
    `auto_approve` and `auto_approve_send_message`.

    We don't instantiate the LangGraph agent here — we just monkeypatch
    `create_deep_agent` and inspect the `interrupt_on` argument it
    receives, same pattern as other unit tests in this codebase.
    """

    def _run_create(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        auto_approve: bool,
        auto_approve_send_message: bool,
    ) -> dict | None:
        from deep_worker_cli import agent as agent_mod

        captured: dict[str, object] = {}

        def _fake_create_deep_agent(**kwargs: object) -> object:
            captured.update(kwargs)

            class _Fake:
                def with_config(self, _cfg: object) -> "_Fake":
                    return self

            return _Fake()

        # Patch create_deep_agent inside deepagents (imported lazily
        # inside create_cli_agent). The import path is:
        #     from deepagents import create_deep_agent
        # inside the function body, so we patch the source module.
        import deepagents as da_mod

        monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)

        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fake_model = FakeListChatModel(responses=["ok"])

        agent_mod.create_cli_agent(
            model=fake_model,
            assistant_id="test_asm_e2e",
            tools=[],
            auto_approve=auto_approve,
            auto_approve_send_message=auto_approve_send_message,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
        )
        return captured.get("interrupt_on")  # type: ignore[return-value]

    def test_no_flags_sends_message_is_gated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        interrupt_on = self._run_create(
            monkeypatch, tmp_path,
            auto_approve=False,
            auto_approve_send_message=False,
        )
        assert isinstance(interrupt_on, dict)
        assert "send_message_to_agent" in interrupt_on
        assert "execute" in interrupt_on

    def test_asm_only_ungates_only_send_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        interrupt_on = self._run_create(
            monkeypatch, tmp_path,
            auto_approve=False,
            auto_approve_send_message=True,
        )
        assert isinstance(interrupt_on, dict)
        assert "send_message_to_agent" not in interrupt_on
        # Others stay gated
        assert "execute" in interrupt_on
        assert "edit_file" in interrupt_on

    def test_sp_ungates_everything(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """-sp (auto_approve=True) empties interrupt_on regardless of
        auto_approve_send_message."""
        interrupt_on = self._run_create(
            monkeypatch, tmp_path,
            auto_approve=True,
            auto_approve_send_message=False,
        )
        assert interrupt_on == {}

    def test_sp_and_asm_together(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        interrupt_on = self._run_create(
            monkeypatch, tmp_path,
            auto_approve=True,
            auto_approve_send_message=True,
        )
        assert interrupt_on == {}
