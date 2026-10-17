"""Regression: web-UI restart MUST honor the agent's persisted dev_tree.

Bug (2026-05-23): ``_spawn_subprocess`` built the relaunch PYTHONPATH from
the WEB SERVER's OWN checkout (``_tree_root`` derived from ``__file__``),
ignoring the agent's saved ``launch_config['dev_tree']``. So an agent
originally launched with ``--dev`` (dev codebase, /path/to/dev-tree),
when restarted from the PROD web UI (:7887), was respawned on the PROD
codebase — silently losing ``--dev``. The CLI ``deep-worker --restart``
launcher already auto-detects this; the web path bypassed it.

Contract under test:
  * config['dev_tree'] truthy  -> spawn from the DEV tree: PYTHONPATH points
    at /path/to/dev-tree/... AND env DEEP_WORKER_DEV_TREE=1 (so the
    relaunched agent re-persists dev_tree=True).
  * config['dev_tree'] falsy/absent -> unchanged: spawn from the web server's
    own tree (prod when run from the production tree).
  * DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE is honored (so tests don't depend on a
    real dev-tree checkout).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.routers.agents import _spawn_subprocess


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the Popen cmd + env that _spawn_subprocess builds."""
    calls: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        calls.append({"cmd": cmd, "env": kwargs.get("env", {})})
        return MagicMock()

    monkeypatch.setattr("app.routers.agents.subprocess.Popen", fake_popen)
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return calls


BASE = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "working_dir": "/path/to/dev-tree",
    "skip_permissions": True,
}


def test_dev_tree_true_spawns_from_dev_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": True}

    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    pythonpath = env.get("PYTHONPATH", "")
    assert "/tmp/fake_dev_tree/deep_worker/deep_worker_cli" in pythonpath, (
        f"a dev_tree agent must be relaunched on the DEV tree, not the web "
        f"server's own tree. PYTHONPATH was {pythonpath!r}"
    )
    # The relaunched agent must re-persist dev_tree=True via this env marker.
    assert env.get("DEEP_WORKER_DEV_TREE") == "1"


def test_dev_tree_false_spawns_from_web_server_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": False}

    _spawn_subprocess("prod_agent", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    pythonpath = env.get("PYTHONPATH", "")
    assert "/tmp/fake_dev_tree" not in pythonpath, (
        f"a non-dev agent must NOT be forced onto the dev tree. PYTHONPATH "
        f"was {pythonpath!r}"
    )


def test_dev_tree_absent_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No dev_tree key at all (legacy config) -> default (web server tree)."""
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = dict(BASE)  # no dev_tree key

    _spawn_subprocess("legacy_agent", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    assert "/tmp/fake_dev_tree" not in env.get("PYTHONPATH", "")
    # No spurious dev marker for a non-dev agent.
    assert env.get("DEEP_WORKER_DEV_TREE") != "1"


def test_dev_tree_true_without_override_uses_canonical_dev_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the test override, dev_tree=True must target the canonical
    /path/to/dev-tree."""
    monkeypatch.delenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", raising=False)
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": True}

    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)

    pythonpath = calls[0]["env"].get("PYTHONPATH", "")
    assert "/path/to/dev-tree/deep_worker/deep_worker_cli" in pythonpath


class TestRpycAsyncPathIsResolvedNotPasted:
    """The spawned agent's rpyc_async path must come from the resolver.

    ``_spawn_subprocess`` used to append a LITERAL ``/path/to/rpyc_async``
    to the child's PYTHONPATH:

        env["PYTHONPATH"] = ":".join([*_web_tree_paths, "/path/to/rpyc_async"])

    ``deep_worker_cli.rpyc_async_source`` exists precisely to answer that
    question, and it does one thing this literal did not: ``_is_checkout()``
    verifies that ``rpyc_async/__init__.py`` is actually present. Its own
    docstring states why — "so that a stray empty directory cannot shadow a
    working installed distribution; that would turn a cosmetic filesystem
    leftover into an ImportError at startup".

    The literal bypassed that guard for every agent the web UI spawns, and
    on an installed host (no checkout at all) it pointed the child at a
    directory that does not exist.
    """

    @staticmethod
    def _stub_boolean_flags(monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the lazy ``agent_command`` import out of this test.

        ``_spawn_subprocess`` imports ``_BOOLEAN_FLAGS`` from
        ``deep_worker_cli.cli.agent_command`` partway through, and that
        module's import chain reaches ``transformers``, which raises
        ``KeyError: frozenset()`` in some environments. The four dev-tree tests
        above fail for that reason at HEAD too — it is pre-existing and
        unrelated to PYTHONPATH construction, which is what is under test
        here.

        The stub's VALUE is copied from the real
        ``agent_command._BOOLEAN_FLAGS`` rather than invented, so the
        flags appended to ``cmd`` stay the ones production appends.
        """
        import sys
        import types

        if "deep_worker_cli.cli.agent_command" in sys.modules:
            return
        stub = types.ModuleType("deep_worker_cli.cli.agent_command")
        stub._BOOLEAN_FLAGS = (  # type: ignore[attr-defined]
            ("skip_permissions", "-sp", "--no-skip-permissions"),
            ("task_manager", "-tm", "--no-tm"),
            ("auto_approve_send_message", "-asm", "--no-auto-approve-send-message"),
            ("memory_only", "--memory-only", "--no-memory-only"),
            ("history_memory", "-m", "--no-m"),
            ("use_sqlite", "--use-sqlite", "--no-use-sqlite"),
            ("dev_tree", "--dev", "--no-dev"),
        )
        monkeypatch.setitem(sys.modules, "deep_worker_cli.cli.agent_command", stub)

    def test_absent_checkout_is_not_pasted_onto_the_child_pythonpath(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """With no checkout present, the path must not appear at all.

        This is the test that FAILS on the previous code: the literal was
        unconditional, so ``/path/to/rpyc_async`` appeared in the child's
        PYTHONPATH whether or not anything was there. The assertion names
        the offending path, so the failure message identifies the defect
        rather than merely reporting a type or a count.
        """
        monkeypatch.setattr(
            "app.routers.agents.ACTIVE_LOCAL_CHECKOUT", None
        )
        self._stub_boolean_flags(monkeypatch)
        calls = _capture(monkeypatch)

        _spawn_subprocess("agent_x", dict(BASE), model=None, provider=None, clean=False)

        pythonpath = calls[0]["env"].get("PYTHONPATH", "")
        assert "/path/to/rpyc_async" not in pythonpath, (
            "the spawned agent's PYTHONPATH still carries the literal "
            "/path/to/rpyc_async even though no checkout is present; an empty "
            "or missing directory there shadows the installed rpyc_async "
            "distribution and the agent dies on ImportError at startup. "
            f"PYTHONPATH was {pythonpath!r}"
        )

    def test_present_checkout_is_handed_to_the_child(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Positive control: a real checkout MUST still be forwarded.

        Without this, the test above would pass just as well on code that
        dropped rpyc_async support altogether — which would break every
        agent where the checkout is the source of truth
        (see rpyc_async_source's module docstring).
        """
        checkout = tmp_path / "rpyc_async_checkout"
        (checkout / "rpyc_async").mkdir(parents=True)
        (checkout / "rpyc_async" / "__init__.py").write_text("")
        monkeypatch.setattr(
            "app.routers.agents.ACTIVE_LOCAL_CHECKOUT", checkout
        )
        self._stub_boolean_flags(monkeypatch)
        calls = _capture(monkeypatch)

        _spawn_subprocess("agent_y", dict(BASE), model=None, provider=None, clean=False)

        pythonpath = calls[0]["env"].get("PYTHONPATH", "")
        assert str(checkout) in pythonpath, (
            "a real rpyc_async checkout was NOT forwarded to the spawned "
            f"agent; PYTHONPATH was {pythonpath!r}"
        )

    def test_call_site_holds_no_rpyc_path_literal(self) -> None:
        """The literal must be gone from the source, not merely bypassed.

        Asserted on the executable source because the defect was a hard-coded
        string: a behavioural test alone would keep passing if someone
        reintroduced the literal behind the same condition. Docstrings and
        comments are excluded — the comment at the call site necessarily
        quotes the path while explaining why it was removed, and matching
        that would be a test that can only be satisfied by deleting its own
        explanation.
        """
        import ast
        import inspect

        from app.routers import agents as agents_module

        tree = ast.parse(inspect.getsource(agents_module))
        # Strip docstrings so prose that mentions the path is not matched.
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:]

        offenders = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "/path/to/rpyc_async" in n.value
        ]
        assert offenders == [], (
            "executable code still contains a hard-coded /path/to/rpyc_async "
            "string literal; the path must come from "
            "deep_worker_cli.rpyc_async_source, which verifies the checkout "
            f"actually exists. Found: {offenders!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
