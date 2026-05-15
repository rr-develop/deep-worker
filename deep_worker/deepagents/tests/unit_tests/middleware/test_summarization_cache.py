"""Tests for SummarizationMiddleware token-count cache
(DESIGN_CONTEXT_DELTA_CACHE.md §4.6).

The middleware caches ``token_counter(messages)`` results keyed by the
id-sequence of the messages list. Repeat counts on an unchanged history
become a dict lookup instead of an O(messages × tokens) pass.

The full middleware cannot be constructed here without llama_index
(``deepagents.backends.protocol`` import transitively pulls it in via
the package's ``__init__``). Instead, these tests exercise the cache
logic directly by creating a bare ``SummarizationMiddleware``-shaped
object via ``__new__`` and manually attaching the cache attributes the
helper relies on. This mirrors exactly what production code does after
``__init__`` finishes; we just skip the parts of ``__init__`` that have
unavailable external dependencies.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_summarization_class() -> type[Any]:
    """Import ``SummarizationMiddleware`` from the source file without
    running ``deepagents.middleware.__init__`` (which imports
    ProjectMemoryMiddleware → semantic_search → llama_index). The full
    module still imports its own transitive deps, but those are
    lighter-weight langchain pieces that ARE available in our test env.
    """
    module_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "deepagents" / "middleware" / "summarization.py"
    )

    # Pre-load sentinel packages as empty module placeholders so the
    # file-level ``from deepagents.backends.protocol import
    # BackendProtocol`` can succeed without touching deepagents/__init__.
    import types
    if "deepagents" not in sys.modules:
        pkg = types.ModuleType("deepagents")
        pkg.__path__ = [str(module_path.parent.parent)]  # type: ignore[attr-defined]
        sys.modules["deepagents"] = pkg
    if "deepagents.backends" not in sys.modules:
        backends_pkg = types.ModuleType("deepagents.backends")
        backends_pkg.__path__ = [  # type: ignore[attr-defined]
            str(module_path.parent.parent / "backends")
        ]
        sys.modules["deepagents.backends"] = backends_pkg

    # Import protocol module directly.
    proto_path = module_path.parent.parent / "backends" / "protocol.py"
    proto_spec = importlib.util.spec_from_file_location(
        "deepagents.backends.protocol", str(proto_path),
    )
    if proto_spec is None or proto_spec.loader is None:
        raise RuntimeError(f"cannot load {proto_path}")
    proto_mod = importlib.util.module_from_spec(proto_spec)
    sys.modules["deepagents.backends.protocol"] = proto_mod
    proto_spec.loader.exec_module(proto_mod)

    # Now load summarization.
    spec = importlib.util.spec_from_file_location(
        "_isolated_summarization", str(module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_isolated_summarization"] = mod
    spec.loader.exec_module(mod)
    return mod.SummarizationMiddleware


try:
    SummarizationMiddleware = _load_summarization_class()
except Exception as e:  # noqa: BLE001
    import pytest
    pytest.skip(
        f"summarization middleware deps unavailable: {e}",
        allow_module_level=True,
    )


class _CountingCounter:
    """Fake token counter: one token per message; tallies call count."""

    def __init__(self) -> None:
        self.calls: int = 0

    def __call__(self, messages: list[Any]) -> int:
        self.calls += 1
        return len(messages)


class _StubMessage:
    """Minimal message shim — ``_cached_total_tokens`` only needs
    ``getattr(m, 'id', ...)``."""

    def __init__(self, mid: str) -> None:
        self.id = mid


def _bare_middleware(counter: _CountingCounter) -> Any:
    """Create a SummarizationMiddleware instance without running
    ``__init__`` so we avoid the real chat-model argument. Attach
    only the fields the cache helper needs."""
    mw = SummarizationMiddleware.__new__(SummarizationMiddleware)
    mw.token_counter = counter
    mw._cached_token_signature = None
    mw._cached_token_total = 0
    return mw


class TestTokenCountCache:
    def test_first_call_invokes_counter(self) -> None:
        counter = _CountingCounter()
        mw = _bare_middleware(counter)
        msgs = [_StubMessage("h1"), _StubMessage("a1")]
        assert mw._cached_total_tokens(msgs) == 2
        assert counter.calls == 1

    def test_same_list_is_cache_hit(self) -> None:
        counter = _CountingCounter()
        mw = _bare_middleware(counter)
        msgs = [_StubMessage("h1"), _StubMessage("a1")]
        mw._cached_total_tokens(msgs)
        mw._cached_total_tokens(msgs)
        mw._cached_total_tokens(msgs)
        assert counter.calls == 1

    def test_appended_message_triggers_recount(self) -> None:
        counter = _CountingCounter()
        mw = _bare_middleware(counter)
        msgs = [_StubMessage("h1")]
        mw._cached_total_tokens(msgs)
        msgs2 = [*msgs, _StubMessage("a1")]
        total = mw._cached_total_tokens(msgs2)
        assert total == 2
        assert counter.calls == 2

    def test_id_change_triggers_recount(self) -> None:
        counter = _CountingCounter()
        mw = _bare_middleware(counter)
        a = [_StubMessage("h1")]
        b = [_StubMessage("h2")]
        mw._cached_total_tokens(a)
        mw._cached_total_tokens(b)
        assert counter.calls == 2

    def test_reorder_triggers_recount(self) -> None:
        """Sequence order matters: swapping two messages is a real
        change that must trigger a recount."""
        counter = _CountingCounter()
        mw = _bare_middleware(counter)
        a = [_StubMessage("h1"), _StubMessage("a1")]
        b = [_StubMessage("a1"), _StubMessage("h1")]
        mw._cached_total_tokens(a)
        mw._cached_total_tokens(b)
        assert counter.calls == 2
