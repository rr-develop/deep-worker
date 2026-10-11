from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepagents.middleware.project_memory import ProjectMemoryMiddleware
from deepagents.utils.project_index import has_semantic_index


def _make_index(root: Path) -> None:
    index = root / ".vector_index"
    (index / "lancedb").mkdir(parents=True)
    (index / "docstore").mkdir()
    (index / "docstore/docstore.json").write_text("{}")
    (index / "file_hashes.json").write_text("{}")


def test_has_semantic_index_with_valid_index(tmp_path: Path) -> None:
    _make_index(tmp_path)
    assert has_semantic_index(tmp_path)


def test_has_semantic_index_without_index(tmp_path: Path) -> None:
    assert not has_semantic_index(tmp_path)


def test_middleware_always_registers_tool(tmp_path: Path) -> None:
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path)
    assert [tool.name for tool in middleware.tools] == ["project_memory"]


def test_middleware_sets_working_dir_even_without_index(tmp_path: Path) -> None:
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path)
    assert middleware.before_agent({}, SimpleNamespace()) == {
        "working_dir": str(tmp_path)
    }


def test_middleware_injects_selected_hint(tmp_path: Path) -> None:
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path)
    middleware._has_index = True
    middleware._cognitive_hint = "Project Knowledge Memory via project_memory"
    request = ModelRequest(
        model="test-model",
        messages=[],
        system_message=SystemMessage(content="Original prompt"),
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert "project_memory" in str(req.system_message.content)
        return ModelResponse(result=[])

    middleware.wrap_model_call(request, handler)
