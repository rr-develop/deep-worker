"""B7: lazy provider root packages + root shim contract (DESIGN §3.2/§3.2b/§3.6).

DESIGN_PROVIDERS_FROM_WHEEL rev 8 requires:

* §3.2 — canonical ``deep_worker.providers.<pkg>`` roots MUST be lazy
  (PEP 562 ``__getattr__``): importing the root must NOT load heavy
  submodules (``claude_agent_sdk`` for claude_code_provider; indexer /
  retriever / history_* / embedding for semantic_search; llama-index).
* §3.2b — the five root shims (``claude_code_provider/`` etc. at the repo
  root) must be single-``__init__.py`` dirs that warn once
  (DeprecationWarning) and delegate lazily to the canonical root with
  root-identity preserved; old submodule paths are NOT supported.
* §3.6 — semantic_search root import (canonical AND via shim) must not
  load indexer/embedding/store and must not pull llama-index.
* §3.2 mutation — no production deep_worker_cli module imports
  ``claude_agent_sdk`` at top level (only lazily inside functions after
  capability resolution).

Heavy-import assertions run in a FRESH interpreter (subprocess) so the
test session's own imports cannot pollute ``sys.modules``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]  # repository root
# OUTER deep_worker_cli dir (contains the deep_worker_cli/ package). The inner
# package dir must NOT be put on PYTHONPATH: its types.py would shadow the
# stdlib `types` module and break site-init (.pth processing) in subprocesses.
_DWC_SRC = _REPO_ROOT / "deep_worker" / "deep_worker_cli"
_DEEPAGENTS = _REPO_ROOT / "deep_worker" / "deepagents"
_DW_ROOT = _REPO_ROOT / "deep_worker"

_PROVIDERS = (
    "claude_code_provider",
    "codex_provider",
    "grok_provider",
    "semantic_search",
    "deepworker_a2a",
)


def _subprocess_env() -> dict[str, str]:
    """Fresh-interpreter env: same import surface as the dev test run."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(
        [
            str(_DWC_SRC),
            str(_DEEPAGENTS),
            str(_DW_ROOT),
            str(_REPO_ROOT),
        ]
    )
    env["PYTHONPATH"] = prefix + (os.pathsep + existing if existing else "")
    return env


def _run_code(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# Canonical roots are lazy (PEP 562)
# ---------------------------------------------------------------------------


def test_all_canonical_roots_have_lazy_getattr() -> None:
    """Every canonical ``deep_worker/providers/<pkg>/__init__.py`` must
    expose a module-level ``__getattr__`` (PEP 562) and must not contain
    eager ``from .<submodule> import`` at top level."""
    for pkg in _PROVIDERS:
        init = _DW_ROOT / "providers" / pkg / "__init__.py"
        assert init.exists(), f"missing canonical root: {init}"
        src = init.read_text(encoding="utf-8")
        assert "def __getattr__" in src, (
            f"{pkg} root must be lazy (PEP 562 __getattr__), got: {init}"
        )
        # No eager submodule import at module top level (only TYPE_CHECKING
        # and inside __getattr__ / importlib.import_module).
        assert "from ." not in src.split('if TYPE_CHECKING:')[0], (
            f"{pkg} root must not eager-import submodules at top level"
        )


def test_fresh_interpreter_canonical_roots_do_not_load_heavy_stack() -> None:
    """Importing the canonical roots in a fresh interpreter must not load
    heavy submodules, claude_agent_sdk, or llama-index."""
    code = """
import sys
import deep_worker.providers.claude_code_provider
import deep_worker.providers.codex_provider
import deep_worker.providers.grok_provider
import deep_worker.providers.semantic_search
import deep_worker.providers.deepworker_a2a

heavy = [
    m for m in sys.modules
    if m.startswith("deep_worker.providers.claude_code_provider.")
    or m.startswith("deep_worker.providers.semantic_search.")
    or m.startswith("deep_worker.providers.deepworker_a2a.")
]
sdk = "claude_agent_sdk" in sys.modules
llama = any(m.startswith("llama_index") for m in sys.modules)
assert not heavy, f"heavy submodules loaded: {heavy}"
assert not sdk, "claude_agent_sdk loaded by root import"
assert not llama, "llama_index loaded by root import"
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"root import pulled heavy stack:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_fresh_interpreter_lazy_resolution_returns_public_api() -> None:
    """Attribute access on the canonical roots resolves the public API
    from the right submodule."""
    code = """
from deep_worker.providers.claude_code_provider import (
    ClaudeCodeChatModel, ClientManager, ProxyMCPServer, WebSearchDelegationTool,
)
from deep_worker.providers.semantic_search import (
    search, IndexConfig, search_messages, index_repository, search_history,
)
from deep_worker.providers.deepworker_a2a import A2AConfig
from deep_worker.providers.codex_provider import CodexChatModel
from deep_worker.providers.grok_provider import GrokChatModel

assert ClaudeCodeChatModel.__module__ == "deep_worker.providers.claude_code_provider.chat_model"
assert search.__module__ == "deep_worker.providers.semantic_search.retriever"
assert search_messages.__module__ == "deep_worker.providers.semantic_search.message_search"
assert index_repository.__module__ == "deep_worker.providers.semantic_search.indexer"
assert search_history.__module__ == "deep_worker.providers.semantic_search.history_retriever"
assert A2AConfig.__module__ == "deep_worker.providers.deepworker_a2a.config"
assert CodexChatModel.__module__ == "deep_worker.providers.codex_provider.chat_model"
assert GrokChatModel.__module__ == "deep_worker.providers.grok_provider.chat_model"
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"lazy resolution failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_unknown_attribute_raises_attribute_error() -> None:
    """PEP 562 __getattr__ must raise AttributeError for unknown names."""
    code = """
import deep_worker.providers.semantic_search as ss
import deep_worker.providers.claude_code_provider as cc
for mod in (ss, cc):
    try:
        mod.does_not_exist_xyz
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"unknown attribute should raise AttributeError:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# Root shims (§3.2b)
# ---------------------------------------------------------------------------


def test_shim_dirs_contain_only_init_py() -> None:
    """Each root shim dir must contain exactly ``__init__.py`` (and no
    source files — old submodule paths are NOT supported)."""
    for pkg in _PROVIDERS:
        shim_dir = _REPO_ROOT / pkg
        assert shim_dir.is_dir(), f"missing root shim dir: {shim_dir}"
        entries = sorted(
            e for e in os.listdir(shim_dir) if e != "__pycache__"
        )
        assert entries == ["__init__.py"], (
            f"shim {pkg}/ must contain ONLY __init__.py, got {entries}"
        )


def test_shim_import_is_lazy_and_warns_once() -> None:
    """``import claude_code_provider`` (no attribute access) must NOT load
    the canonical root; the first attribute access emits exactly ONE
    DeprecationWarning and delegates."""
    code = """
import sys
import warnings

import claude_code_provider
assert "deep_worker.providers.claude_code_provider" not in sys.modules, (
    "shim import must not load the canonical root"
)
assert not any(m.startswith("deep_worker.providers.") for m in sys.modules), (
    "shim import must not load any canonical provider"
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from claude_code_provider import ClaudeCodeChatModel
    dwarns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dwarns) == 1, f"expected exactly 1 DeprecationWarning, got {len(dwarns)}"
    # Second access: still one warning total (warn-once guard).
    _ = claude_code_provider.ClaudeCodeChatModel
    dwarns2 = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dwarns2) == 1, "warn-once violated"

import deep_worker.providers.claude_code_provider as canonical
assert claude_code_provider.ClaudeCodeChatModel is canonical.ClaudeCodeChatModel, (
    "root identity must be preserved (no double-loading)"
)
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"shim contract violated:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_semantic_search_shim_is_lazy() -> None:
    """``import semantic_search`` (shim) must not load any canonical
    semantic_search submodule and must not pull llama-index."""
    code = """
import sys
import semantic_search
assert not any(m.startswith("deep_worker.providers.semantic_search") for m in sys.modules), (
    "semantic_search shim import must not load canonical submodules"
)
assert not any(m.startswith("llama_index") for m in sys.modules), (
    "semantic_search shim import must not pull llama-index"
)
import deep_worker.providers.semantic_search as canonical
assert semantic_search.search is canonical.search, "root identity must hold"
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"semantic_search shim violated:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_old_submodule_paths_not_supported() -> None:
    """Old top-level submodule imports must raise ModuleNotFoundError and
    must NOT create the module under the old name."""
    code = """
import sys
for old in ("claude_code_provider.web_search", "codex_provider.native_tools",
            "semantic_search.history_indexer"):
    try:
        __import__(old)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"{old} must NOT be importable (shim has no submodules)")
    assert old not in sys.modules, f"{old} must not be created in sys.modules"
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"old submodule path should fail:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_shim_unknown_attribute_raises_attribute_error() -> None:
    code = """
import claude_code_provider
try:
    claude_code_provider.nope
except AttributeError:
    pass
else:
    raise AssertionError("expected AttributeError from shim")
print("OK")
"""
    result = _run_code(code)
    assert result.returncode == 0, (
        f"shim unknown attribute should raise AttributeError:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# §3.2 mutation: no top-level claude_agent_sdk import in production code
# ---------------------------------------------------------------------------


def test_no_top_level_claude_agent_sdk_import_in_production() -> None:
    """No production deep_worker_cli module may import ``claude_agent_sdk``
    at top level — only lazily inside functions after capability
    resolution (mutation test §3.2)."""
    prod_root = _DWC_SRC
    offenders: list[str] = []
    for path in sorted(prod_root.rglob("*.py")):
        if "tests" in path.parts or "/docs/" in str(path):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped.startswith(
                ("import claude_agent_sdk", "from claude_agent_sdk")
            ):
                continue
            # Module top level = zero indentation. Any indentation means the
            # import is function-local (lazy) — allowed.
            if line[: len(line) - len(line.lstrip())] == "":
                offenders.append(
                    f"{path.relative_to(prod_root)}:{lineno}: {stripped}"
                )
    assert not offenders, (
        "top-level claude_agent_sdk import found in production code:\n"
        + "\n".join(offenders)
    )
