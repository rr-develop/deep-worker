"""Unit tests must not require a dependency CI does not install.

CI installs the package with a plain ``pip install -e .`` (see
``.github/workflows/unittest.yml``), so the optional extras and the private
``claude_agent_sdk`` fork are absent there. A test that imports one of them
without a skip does not fail loudly as "dependency missing" — it fails as a
collection error, and the whole Unit tests job goes red. That is exactly how
sixteen tests sat red across every run for weeks: green locally, where the
extras happen to be installed, red in CI where they are not.

This guard is static on purpose. It reads the test sources rather than
importing them, so it reports the same answer whether or not the optional
packages are present in the environment running it — which is the property
the failure mode needs. A guard that only fired when the dependency was
missing would be silent on every developer machine that has it.

Scope is module-level imports only, because those run during collection and
take the entire job down with them. An import nested inside a test function
is left alone: the monorepo provider packages do resolve in CI, so several
such tests pass there, and flagging them was a false positive.

Adding a test that needs one of these dependencies is fine. Mark it:

    pytest.importorskip("llama_index")  # semantic-search dep; skip when absent

or, for a single test or class, and to keep the other parameters running::

    @pytest.mark.skipif(
        importlib.util.find_spec("claude_agent_sdk") is None, reason=...
    )
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UNIT_TESTS_ROOT = Path(__file__).parent

#: Every tree whose module-level imports run during a CI collection, not just
#: this package's. The repo-root ``tests/`` directory was outside the sample
#: until 2026-10-20, so six unguarded ``llama_index`` importers there were
#: invisible to this guard while being exactly the failure it exists to catch.
#: Each root is scanned independently; a root that does not exist is skipped so
#: the guard still runs from a partial checkout.
SCANNED_ROOTS = (
    UNIT_TESTS_ROOT,
    Path(__file__).resolve().parents[4] / "tests",
)

#: Modules absent from the CI environment. ``claude_agent_sdk`` is a private
#: source fork (CLAUDE_AGENT_SDK_SRC); ``llama_index`` and ``lancedb`` come
#: with the ``semantic-search`` extra; ``a2a`` with the ``a2a`` extra. Keyed by
#: top-level module name because that is what an ``import`` statement carries.
OPTIONAL_MODULES = frozenset(
    {
        "claude_agent_sdk",
        "llama_index",
        "lancedb",
        "a2a",
    }
)

#: Provider modules that pull an optional package in at their own module
#: scope. The provider *packages* themselves import fine in CI — they live in
#: the monorepo, not behind an extra — so only the submodules that reach for
#: an absent third-party package belong here. Verified by importing each in a
#: venv built the way the workflow builds it; anything importable there is
#: deliberately absent from this tuple.
OPTIONAL_PROVIDER_PREFIXES = (
    "deep_worker.providers.claude_code_sdk_provider.chat_model",
    "deep_worker.providers.claude_code_sdk_provider.client_manager",
    "deep_worker.providers.claude_code_sdk_provider.proxy_mcp_server",
    "deep_worker.providers.claude_code_sdk_provider.web_search",
    # semantic_search submodules that import llama_index at their own module
    # scope. Established the same way as the entries above — importing each
    # submodule with the extra absent: these five raise ModuleNotFoundError,
    # while _embed_cache, _tokenizer, config, exceptions,
    # long_text_embedding, message_embedding_store, message_search and
    # scanner all import fine and are deliberately omitted, as is the
    # ``semantic_search`` package itself.
    "deep_worker.providers.semantic_search.history_indexer",
    "deep_worker.providers.semantic_search.history_retriever",
    "deep_worker.providers.semantic_search.indexer",
    "deep_worker.providers.semantic_search.retriever",
    "deep_worker.providers.semantic_search.search_cli",
)


def _iter_test_files() -> list[Path]:
    seen: set[Path] = set()
    for root in SCANNED_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("test_*.py"):
            if "__pycache__" in p.parts or p.resolve() == Path(__file__).resolve():
                continue
            seen.add(p.resolve())
    return sorted(seen)


def _case_id(path: Path) -> str:
    """Stable id that cannot collide between roots.

    Files in ``UNIT_TESTS_ROOT`` keep the bare relative id they have always
    had, so widening the sample does not rename 352 existing cases. Files
    from any other root carry that root's name as a prefix, which both marks
    where they came from and keeps a same-named file in two trees distinct.
    ``relative_to(UNIT_TESTS_ROOT)`` alone would raise for the new roots.
    """
    resolved = path.resolve()
    unit_root = UNIT_TESTS_ROOT.resolve()
    try:
        return str(resolved.relative_to(unit_root))
    except ValueError:
        pass
    for root in SCANNED_ROOTS:
        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{root.name}/{rel}"
    return str(resolved)


def _module_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    """Top-level module names an import statement pulls in."""
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    # Relative imports (level > 0) never reach a third-party package.
    if node.level or not node.module:
        return set()
    return {node.module.split(".")[0]}


def _full_module_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if node.level or not node.module:
        return set()
    return {node.module}


def _module_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Imports executed at import time — i.e. during collection.

    Only these are checked. An import nested inside a test function runs when
    that test runs, and several such tests pass in CI precisely because the
    monorepo provider packages resolve there; flagging them produced a false
    positive on cli/test_import_claude_cli.py, which CI has never failed.
    Collection-time imports are the ones that take the whole job down.
    """
    return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def _has_guard(source: str, tree: ast.Module) -> bool:
    """True when the file carries any optional-dependency guard.

    Deliberately coarse: this guard's job is to catch the author who wrote no
    guard at all. Verifying that a specific skip covers a specific import is
    the kind of precision that produces false positives on legitimate layouts
    (a parametrized case marked at the parameter, a class-level skipif, a
    fixture that skips), and a guard that cries wolf gets deleted.
    """
    if "importorskip" in source:
        return True
    # skipif / skip markers, however they are spelled.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in {"skipif", "skip"}:
            return True
    return False


def _offending_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in _module_level_imports(tree):
        for root in _module_roots(node):
            if root in OPTIONAL_MODULES:
                found.add(root)
        for full in _full_module_names(node):
            for prefix in OPTIONAL_PROVIDER_PREFIXES:
                if full == prefix or full.startswith(prefix + "."):
                    found.add(prefix)
    return found


@pytest.mark.parametrize("test_file", _iter_test_files(), ids=_case_id)
def test_optional_dependency_is_guarded_by_a_skip(test_file: Path) -> None:
    source = test_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(test_file))

    offenders = _offending_imports(tree)
    if not offenders:
        return

    assert _has_guard(source, tree), (
        f"{_case_id(test_file)} imports {sorted(offenders)}, "
        "which CI does not install (`pip install -e .` pulls no extras and no "
        "claude_agent_sdk fork), and carries no skip. Without one this file "
        "turns the whole Unit tests job red while passing locally. Add "
        'pytest.importorskip("<module>") at module scope, or '
        "@pytest.mark.skipif(importlib.util.find_spec(...) is None, ...) on the "
        "specific test, class or parameter."
    )


def test_guard_actually_scans_the_tree() -> None:
    """The parametrization must not silently degenerate to nothing.

    If ``_iter_test_files`` were ever to return an empty list — a bad root, a
    renamed layout — every case above would vanish and the guard would report
    success while checking nothing.
    """
    files = _iter_test_files()
    assert len(files) > 100, f"expected the unit test tree, found {len(files)} files"

    # Every configured root must contribute, or a path that silently stopped
    # resolving would shrink the sample back without any case turning red —
    # which is precisely how the repo-root tree went unscanned until
    # 2026-10-20.
    for root in SCANNED_ROOTS:
        if not root.is_dir():
            continue
        assert any(f.is_relative_to(root.resolve()) for f in files), (
            f"root {root} contributed no files; the sample silently shrank"
        )


def test_known_offenders_would_be_caught() -> None:
    """The guard must fail on the shape it exists to prevent.

    Pinning this keeps the detector honest: a refactor that broke
    ``_offending_imports`` would otherwise leave every file trivially passing.
    """
    bad = ast.parse("import llama_index\n\ndef test_x():\n    assert True\n")
    assert _offending_imports(bad) == {"llama_index"}
    assert not _has_guard("import llama_index", bad)

    bad_from = ast.parse(
        "from deep_worker.providers.claude_code_sdk_provider.chat_model import X\n"
    )
    assert _offending_imports(bad_from) == {
        "deep_worker.providers.claude_code_sdk_provider.chat_model"
    }

    # The provider package itself resolves in CI, so importing it is fine.
    ok_pkg = ast.parse(
        "from deep_worker.providers.claude_code_sdk_provider import session_state\n"
    )
    assert _offending_imports(ok_pkg) == set()

    # An import inside a test body is out of scope: it does not run at
    # collection time, and such tests pass in CI today.
    nested = ast.parse("def test_x():\n    import llama_index\n")
    assert _offending_imports(nested) == set()

    good_src = 'import pytest\npytest.importorskip("llama_index")\nimport llama_index\n'
    good = ast.parse(good_src)
    assert _offending_imports(good) == {"llama_index"}
    assert _has_guard(good_src, good)
