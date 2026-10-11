"""An in-flight ``session/prompt`` must have a liveness bound (round 8 #4).

``_consume_turn`` and ``_drain_pump`` waited on {notification, response}
with NO timeout, so only a CRASH (EOF on stdout) was ever handled. The
"alive but silent" case — a wedged CLI, or a reader task that died before
round 8's guard existed — hung the agent FOREVER, with nothing logged.

Contract:

* the bound is an IDLE timer, not a total turn budget: it RESETS on every
  notification, so a legitimately long turn never trips it as long as the
  CLI keeps streaming;
* total silence for the window fails the turn LOUDLY with a distinct error
  (not the crash error — the process is alive, which the operator needs to
  know);
* the pump has no caller to raise at, so it fails the response future
  instead — that is what the next ``_consume_turn`` awaits;
* the window is operator-tunable, and a bad value must NOT disable the
  bound (an unbounded wait is the defect being fixed).
"""
from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.acp.chat_model import (
    AcpPromptIdleTimeoutError,
    _prompt_idle_timeout,
)
from deep_worker.providers.acp.custom_types import (
    PROMPT_IDLE_TIMEOUT_DEFAULT,
    PROMPT_IDLE_TIMEOUT_ENV,
)


class TestIdleTimeoutSetting:
    def test_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(PROMPT_IDLE_TIMEOUT_ENV, raising=False)
        assert _prompt_idle_timeout() == PROMPT_IDLE_TIMEOUT_DEFAULT

    def test_operator_can_tune(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROMPT_IDLE_TIMEOUT_ENV, "42.5")
        assert _prompt_idle_timeout() == 42.5

    def test_read_live_not_cached_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PROMPT_IDLE_TIMEOUT_ENV, "11")
        assert _prompt_idle_timeout() == 11
        monkeypatch.setenv(PROMPT_IDLE_TIMEOUT_ENV, "12")
        assert _prompt_idle_timeout() == 12

    @pytest.mark.parametrize("bad", ["", "abc", "None", "1e", "0", "-5"])
    def test_bad_value_never_disables_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        """A typo must not restore the unbounded wait this fix removes."""
        monkeypatch.setenv(PROMPT_IDLE_TIMEOUT_ENV, bad)
        value = _prompt_idle_timeout()
        assert value == PROMPT_IDLE_TIMEOUT_DEFAULT
        assert value > 0


class TestConsumeTurnIdleBound:
    """Drive the real ``asyncio.wait`` shape used by ``_consume_turn``."""

    @staticmethod
    async def _wait_once(timeout: float, feed: bool):
        """Mirror of the production wait: a queue getter + a response future."""
        q: asyncio.Queue = asyncio.Queue()
        response_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        get_task = asyncio.ensure_future(q.get())
        if feed:
            q.put_nowait({"some": "update"})
        done, _pending = await asyncio.wait(
            {get_task, response_fut},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )
        get_task.cancel()
        return done

    def test_silence_produces_an_empty_done_set(self) -> None:
        """``not done`` is the signal the production code branches on."""
        done = asyncio.run(self._wait_once(0.05, feed=False))
        assert done == set(), "silent wait did not time out"

    def test_activity_prevents_the_timeout(self) -> None:
        done = asyncio.run(self._wait_once(0.05, feed=True))
        assert done, "a notification failed to satisfy the wait"

    def test_error_names_the_cause_and_is_not_a_crash(self) -> None:
        """Operators must be able to tell 'silent' from 'crashed'."""
        from deep_worker.providers.acp.acp_client import AcpClientError

        err = AcpPromptIdleTimeoutError("ACP turn produced no output for 900s")
        assert not isinstance(err, AcpClientError), (
            "idle timeout must not masquerade as a CLI crash"
        )
        assert "no output" in str(err)


class TestIdleTimerResets:
    """The bound must be IDLE-based: a long-but-chatty turn must survive."""

    def test_many_short_gaps_do_not_trip_the_bound(self) -> None:
        async def scenario() -> int:
            q: asyncio.Queue = asyncio.Queue()
            response_fut: asyncio.Future = (
                asyncio.get_running_loop().create_future()
            )
            rounds = 0
            for _ in range(5):
                get_task = asyncio.ensure_future(q.get())
                q.put_nowait({"n": rounds})
                done, _p = await asyncio.wait(
                    {get_task, response_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.05,
                )
                if not done:
                    get_task.cancel()
                    break
                get_task.cancel()
                rounds += 1
                await asyncio.sleep(0.02)  # gap shorter than the window
            return rounds

        assert asyncio.run(scenario()) == 5, (
            "a steadily-streaming turn tripped the idle bound — the timer "
            "is behaving as a TOTAL budget instead of resetting"
        )


class TestProductionCallSitesAreBounded:
    """Guard against a regression that the behavioural tests above cannot
    see: they exercise the SHAPE of ``asyncio.wait``, not the actual call
    sites. If someone drops the ``timeout=`` argument in chat_model.py the
    unbounded wait is back and every test above still passes.

    Parse the real module and assert that EVERY ``asyncio.wait`` in it
    passes a timeout. This is a structural claim about production code, so
    it is checked structurally — no import-time monkeypatching, no source
    substring matching.
    """

    @staticmethod
    def _wait_calls() -> list:
        import ast
        import inspect

        from deep_worker.providers.acp import chat_model

        tree = ast.parse(inspect.getsource(chat_model))
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "wait"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            ):
                out.append(node)
        return out

    def test_every_asyncio_wait_passes_a_timeout(self) -> None:
        calls = self._wait_calls()
        assert calls, "no asyncio.wait found — did the module move?"
        unbounded = [
            node.lineno
            for node in calls
            if not any(kw.arg == "timeout" for kw in node.keywords)
        ]
        assert not unbounded, (
            f"asyncio.wait without a timeout at line(s) {unbounded} — "
            "this reintroduces the unbounded wait of round 8 defect #4"
        )

    def test_both_known_call_sites_are_still_present(self) -> None:
        """_consume_turn and _drain_pump — if one disappears, this test
        must fail loudly rather than silently guard only the survivor."""
        assert len(self._wait_calls()) >= 2, (
            "expected the consume-turn and pump waits; found "
            f"{len(self._wait_calls())}"
        )
