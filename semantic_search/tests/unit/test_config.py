"""Unit tests for semantic_search.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_search.config import (
    CODE_EXTENSIONS,
    EXCLUDED_DIRS,
    INDEXED_EXTENSIONS,
    LANGUAGE_MAP,
    IndexConfig,
    RepoIndex,
)


class TestIndexedExtensions:
    def test_contains_python(self) -> None:
        assert ".py" in INDEXED_EXTENSIONS
        assert ".pyi" in INDEXED_EXTENSIONS

    def test_contains_typescript(self) -> None:
        assert ".ts" in INDEXED_EXTENSIONS
        assert ".tsx" in INDEXED_EXTENSIONS

    def test_contains_javascript(self) -> None:
        assert ".js" in INDEXED_EXTENSIONS
        assert ".jsx" in INDEXED_EXTENSIONS

    def test_contains_jvm_languages(self) -> None:
        assert ".java" in INDEXED_EXTENSIONS
        assert ".kt" in INDEXED_EXTENSIONS
        assert ".kts" in INDEXED_EXTENSIONS

    def test_contains_c_cpp(self) -> None:
        assert ".c" in INDEXED_EXTENSIONS
        assert ".cpp" in INDEXED_EXTENSIONS
        assert ".cc" in INDEXED_EXTENSIONS
        assert ".h" in INDEXED_EXTENSIONS
        assert ".hpp" in INDEXED_EXTENSIONS

    def test_contains_systems_languages(self) -> None:
        assert ".go" in INDEXED_EXTENSIONS
        assert ".rs" in INDEXED_EXTENSIONS

    def test_contains_docs_and_config(self) -> None:
        assert ".md" in INDEXED_EXTENSIONS
        assert ".toml" in INDEXED_EXTENSIONS
        assert ".yaml" in INDEXED_EXTENSIONS
        assert ".yml" in INDEXED_EXTENSIONS

    def test_is_frozenset(self) -> None:
        assert isinstance(INDEXED_EXTENSIONS, frozenset)


class TestCodeExtensions:
    def test_code_extensions_subset_of_indexed(self) -> None:
        assert CODE_EXTENSIONS.issubset(INDEXED_EXTENSIONS)

    def test_docs_not_in_code_extensions(self) -> None:
        assert ".md" not in CODE_EXTENSIONS
        assert ".toml" not in CODE_EXTENSIONS
        assert ".yaml" not in CODE_EXTENSIONS
        assert ".yml" not in CODE_EXTENSIONS

    def test_code_files_present(self) -> None:
        for ext in (".py", ".ts", ".java", ".kt", ".c", ".cpp", ".go", ".rs"):
            assert ext in CODE_EXTENSIONS

    def test_is_frozenset(self) -> None:
        assert isinstance(CODE_EXTENSIONS, frozenset)


class TestLanguageMap:
    def test_python_mapping(self) -> None:
        assert LANGUAGE_MAP[".py"] == "python"
        assert LANGUAGE_MAP[".pyi"] == "python"

    def test_javascript_mapping(self) -> None:
        assert LANGUAGE_MAP[".js"] == "javascript"
        assert LANGUAGE_MAP[".jsx"] == "javascript"

    def test_typescript_mapping(self) -> None:
        assert LANGUAGE_MAP[".ts"] == "typescript"
        assert LANGUAGE_MAP[".tsx"] == "typescript"

    def test_java_mapping(self) -> None:
        assert LANGUAGE_MAP[".java"] == "java"

    def test_kotlin_mapping(self) -> None:
        assert LANGUAGE_MAP[".kt"] == "kotlin"
        assert LANGUAGE_MAP[".kts"] == "kotlin"

    def test_c_mapping(self) -> None:
        assert LANGUAGE_MAP[".c"] == "c"
        assert LANGUAGE_MAP[".h"] == "c"

    def test_cpp_mapping(self) -> None:
        assert LANGUAGE_MAP[".cpp"] == "cpp"
        assert LANGUAGE_MAP[".cc"] == "cpp"
        assert LANGUAGE_MAP[".hpp"] == "cpp"

    def test_go_mapping(self) -> None:
        assert LANGUAGE_MAP[".go"] == "go"

    def test_rust_mapping(self) -> None:
        assert LANGUAGE_MAP[".rs"] == "rust"

    def test_all_code_extensions_have_mapping(self) -> None:
        for ext in CODE_EXTENSIONS:
            assert ext in LANGUAGE_MAP, f"Missing LANGUAGE_MAP entry for {ext}"


class TestExcludedDirs:
    def test_excludes_git(self) -> None:
        assert ".git" in EXCLUDED_DIRS

    def test_excludes_node_modules(self) -> None:
        assert "node_modules" in EXCLUDED_DIRS

    def test_excludes_pycache(self) -> None:
        assert "__pycache__" in EXCLUDED_DIRS

    def test_excludes_vector_index(self) -> None:
        assert ".vector_index" in EXCLUDED_DIRS

    def test_excludes_venv(self) -> None:
        assert ".venv" in EXCLUDED_DIRS
        assert "venv" in EXCLUDED_DIRS

    def test_excludes_build_artifacts(self) -> None:
        assert "dist" in EXCLUDED_DIRS
        assert "build" in EXCLUDED_DIRS

    def test_is_frozenset(self) -> None:
        assert isinstance(EXCLUDED_DIRS, frozenset)


class TestIndexConfig:
    def test_defaults_are_sensible(self) -> None:
        config = IndexConfig()
        assert config.chunk_lines > 0
        assert config.chunk_lines_overlap >= 0
        assert config.chunk_lines_overlap < config.chunk_lines
        assert config.max_chars > 0
        assert config.child_chunk_size > 0
        # parent_chunk_size can be None (flat structure) or > child_chunk_size
        if config.parent_chunk_size is not None:
            assert config.parent_chunk_size > config.child_chunk_size
        assert 0.0 < config.simple_ratio_thresh < 1.0
        assert config.embedding_model != ""

    def test_is_frozen(self) -> None:
        config = IndexConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.chunk_lines = 999  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = IndexConfig(chunk_lines=20, parent_chunk_size=1024)
        assert config.chunk_lines == 20
        assert config.parent_chunk_size == 1024


class TestRepoIndex:
    def test_index_dir_is_inside_repo(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        assert repo_index.index_dir.parent == tmp_path

    def test_index_dir_name(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        assert repo_index.index_dir.name == ".vector_index"

    def test_lancedb_uri_is_string(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        assert isinstance(repo_index.lancedb_uri, str)
        assert ".vector_index" in repo_index.lancedb_uri

    def test_docstore_path_inside_index_dir(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        assert repo_index.docstore_path.parent == repo_index.index_dir

    def test_exists_false_when_no_index(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        assert repo_index.exists() is False

    def test_exists_true_when_index_created(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        repo_index.index_dir.mkdir()
        assert repo_index.exists() is True

    def test_is_frozen(self, tmp_path: Path) -> None:
        repo_index = RepoIndex(repo_path=tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            repo_index.repo_path = Path("/other")  # type: ignore[misc]
