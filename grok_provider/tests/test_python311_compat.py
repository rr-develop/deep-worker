import py_compile
from pathlib import Path


def test_chat_model_compiles_on_python311() -> None:
    chat_model = Path(__file__).resolve().parents[1] / "chat_model.py"

    py_compile.compile(chat_model, doraise=True)
