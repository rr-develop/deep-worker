"""``grok_provider/chat_model.py`` must stay syntactically valid.

The path is derived from the repo root rather than from ``parents[1]``:
these tests used to live inside the provider package, where the sibling
``chat_model.py`` sat one level up. They now live in the repo-root
``tests/`` tree, so the old relative hop pointed at ``tests/chat_model.py``
— a file that does not exist — and the test failed on FileNotFoundError
instead of ever compiling anything.
"""
import py_compile
from pathlib import Path


def test_chat_model_compiles_on_python311() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    chat_model = (
        repo_root / "deep_worker" / "providers" / "grok_provider" / "chat_model.py"
    )
    assert chat_model.is_file(), f"target missing: {chat_model}"

    py_compile.compile(chat_model, doraise=True)
