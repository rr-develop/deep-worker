"""The agent-side import boundary of the semantic-memory package.

Design: docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md §9 / §9.1.

The supervisor lives inside ``service/`` (policy §8: only ``*.service*``
may spawn a backend) and the agent-side client imports it. So the ONLY
thing keeping the heavy stack out of every agent process is that
``service/__init__.py`` executes nothing.

``the code-graph server`` has exactly this defect: its design doc §3.3 claims the
client never imports ``the code-graph server.service``, while ``stdio_server.py``
does — safe purely by accident. Pin it here so it cannot rot.

The full transitive import lock over the tools lands with the client
commit; this file guards the package layout the lock depends on.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import deep_worker_cli.semantic_memory as pkg
import deep_worker_cli.semantic_memory.service as service_pkg

#: Importing any of these into an agent is the regression this whole
#: service exists to prevent.
_HEAVY = frozenset(
    {
        "torch",
        "transformers",
        "sentence_transformers",
        "llama_index",
        "lancedb",
        "numpy",
        "pandas",
    }
)


def _module_dir(module: object) -> Path:
    return Path(module.__file__).parent  # type: ignore[union-attr,arg-type]


def _imported_names(source: str) -> set[str]:
    """Top-level module names imported by ``source``, at any nesting."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — stays inside the package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


class TestServiceInitIsEmpty:
    def test_contains_no_executable_statements(self) -> None:
        """``service/__init__.py`` must not execute anything.

        A single import here would be inherited by every agent that
        touches the supervisor — which is every agent that uses either
        memory tool.
        """
        path = _module_dir(service_pkg) / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = [
            node
            for node in tree.body
            if not isinstance(node, ast.Expr)
            or not isinstance(node.value, ast.Constant)
        ]
        assert offenders == [], (
            "service/__init__.py must contain nothing but an optional "
            f"docstring; found {[type(n).__name__ for n in offenders]}"
        )

    def test_package_init_is_also_inert(self) -> None:
        path = _module_dir(pkg) / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = [
            node
            for node in tree.body
            if not isinstance(node, ast.Expr)
            or not isinstance(node.value, ast.Constant)
        ]
        assert offenders == []


class TestNoHeavyImports:
    def test_registry_and_supervisor_are_light(self) -> None:
        """Both modules run on every agent start — keep them cheap."""
        for name in ("registry.py", "service/supervisor.py"):
            path = _module_dir(pkg) / name
            imported = _imported_names(path.read_text(encoding="utf-8"))
            leaked = imported & _HEAVY
            assert not leaked, f"{name} imports heavy deps: {sorted(leaked)}"

    def test_supervisor_does_not_import_grpc(self) -> None:
        """The supervisor runs on every agent start, before any RPC.

        Liveness is deliberately pid + start_ticks + is_socket rather than
        a Health RPC (design §7), precisely so this import never happens.
        """
        path = _module_dir(pkg) / "service" / "supervisor.py"
        assert "grpc" not in _imported_names(path.read_text(encoding="utf-8"))

    def test_client_transitive_imports_do_not_load_heavy_stack(self) -> None:
        forbidden = sorted(
            _HEAVY
            | {
                "deep_worker.providers.semantic_search",
            }
        )
        script = (
            "import json, sys\n"
            "import deep_worker_cli.semantic_memory.client\n"
            f"forbidden = {forbidden!r}\n"
            "print(json.dumps([name for name in forbidden "
            "if name in sys.modules or any(m.startswith(name + '.') "
            "for m in sys.modules)]))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=_module_dir(pkg).parents[1],
            text=True,
        )
        assert proc.stdout.strip() == "[]"

    def test_tool_entrypoints_do_not_load_heavy_stack(self) -> None:
        forbidden = sorted(_HEAVY | {"deep_worker.providers.semantic_search"})
        script = (
            "import json, sys\n"
            "import deepagents, deep_worker_cli\n"
            f"forbidden = {forbidden!r}\n"
            "before = {name for name in forbidden "
            "if name in sys.modules or any(m.startswith(name + '.') "
            "for m in sys.modules)}\n"
            "import deepagents.tools.project_memory\n"
            "import deep_worker_cli.history_memory_tool\n"
            "print(json.dumps([name for name in forbidden "
            "if name not in before and (name in sys.modules or "
            "any(m.startswith(name + '.') for m in sys.modules))]))\n"
        )
        package_root = _module_dir(pkg).parents[1]
        deepagents_root = package_root.parent / "deepagents"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(package_root), str(deepagents_root)]
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=package_root,
            env=env,
            text=True,
        )
        assert proc.stdout.strip() == "[]"

    def test_agent_startup_does_not_import_history_indexer(self) -> None:
        from deep_worker_cli import agent

        source = Path(agent.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        create_agent = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "create_cli_agent"
        )
        imported = {
            node.module
            for node in ast.walk(create_agent)
            if isinstance(node, ast.ImportFrom)
        }
        assert "deep_worker.providers.semantic_search.history_indexer" not in imported


class TestSpawnerAllowlist:
    """Policy §8: only a ``service/`` package may spawn a backend.

    Our supervisor sits in ``semantic_memory/service/`` precisely so it
    satisfies that rule, but the guard enforces it against a closed list
    — and a list that does not name us would fail the build the moment
    the guard is run over this package. Pin both directions: we are
    allowed, and modules outside ``service/`` still are not.
    """

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def test_semantic_memory_service_may_spawn(self) -> None:
        from deep_worker_cli.lsp.policy_guard import _in_allowlisted_package

        root = self._repo_root()
        assert _in_allowlisted_package(
            root / "deep_worker_cli/semantic_memory/service/supervisor.py", root
        )

    def test_lsp_service_is_still_allowed(self) -> None:
        """The list grew; it must not have been replaced."""
        from deep_worker_cli.lsp.policy_guard import _in_allowlisted_package

        root = self._repo_root()
        assert _in_allowlisted_package(
            root / "deep_worker_cli/lsp/service/supervisor.py", root
        )

    def test_modules_outside_service_may_not_spawn(self) -> None:
        """The allowlist stays closed — widening it to the whole package
        would let any agent-side module spawn a heavy backend."""
        from deep_worker_cli.lsp.policy_guard import _in_allowlisted_package

        root = self._repo_root()
        assert not _in_allowlisted_package(
            root / "deep_worker_cli/semantic_memory/registry.py", root
        )
        assert not _in_allowlisted_package(root / "deep_worker_cli/agent.py", root)
