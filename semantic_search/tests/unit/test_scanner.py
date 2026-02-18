"""Unit tests for semantic_search.scanner."""

from __future__ import annotations

from pathlib import Path

from semantic_search.scanner import (
    compute_file_hash,
    get_language,
    is_code_file,
    scan_files,
)


class TestScanFiles:
    def test_finds_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def foo(): pass")

        result = list(scan_files(tmp_path))

        assert len(result) == 2
        assert all(p.suffix == ".py" for p in result)

    def test_finds_multiple_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "app.ts").write_text("const x = 1;")
        (tmp_path / "README.md").write_text("# Hello")
        (tmp_path / "config.toml").write_text("[tool]")

        result = set(p.name for p in scan_files(tmp_path))

        assert "app.ts" in result
        assert "README.md" in result
        assert "config.toml" in result

    def test_excludes_git_directory(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        (tmp_path / "main.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all(".git" not in str(p) for p in result)

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module.exports = {}")
        (tmp_path / "index.ts").write_text("export {}")

        result = list(scan_files(tmp_path))

        # Check path components, not substring, to avoid false positives
        # from pytest temp dir names like "test_excludes_node_modules0"
        assert all("node_modules" not in p.parts for p in result)
        assert any(p.name == "index.ts" for p in result)

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-312.pyc").write_bytes(b"\x00\x00")
        (tmp_path / "module.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all("__pycache__" not in str(p) for p in result)

    def test_excludes_vector_index(self, tmp_path: Path) -> None:
        vi = tmp_path / ".vector_index"
        vi.mkdir()
        (vi / "metadata.json").write_text("{}")
        (tmp_path / "app.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all(".vector_index" not in str(p) for p in result)

    def test_excludes_venv(self, tmp_path: Path) -> None:
        venv = tmp_path / ".venv"
        venv.mkdir()
        lib = venv / "lib" / "python3.12" / "site-packages"
        lib.mkdir(parents=True)
        (lib / "foo.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all(".venv" not in str(p) for p in result)
        assert any(p.name == "main.py" for p in result)

    def test_excludes_dist_and_build(self, tmp_path: Path) -> None:
        for dir_name in ("dist", "build"):
            d = tmp_path / dir_name
            d.mkdir()
            (d / "bundle.js").write_text("// bundle")
        (tmp_path / "src.ts").write_text("export {}")

        result = list(scan_files(tmp_path))

        # Check path components to avoid false positives from pytest dir names
        assert all("dist" not in p.parts and "build" not in p.parts for p in result)

    def test_skips_unknown_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "binary.exe").write_bytes(b"\x4d\x5a")
        (tmp_path / "main.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all(p.suffix == ".py" for p in result)

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "src" / "utils"
        sub.mkdir(parents=True)
        (sub / "helper.py").write_text("def help(): pass")
        (tmp_path / "main.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        names = {p.name for p in result}
        assert "helper.py" in names
        assert "main.py" in names

    def test_empty_directory_returns_no_files(self, tmp_path: Path) -> None:
        result = list(scan_files(tmp_path))
        assert result == []

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            list(scan_files(tmp_path / "nonexistent"))

    def test_returns_absolute_paths(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1")

        result = list(scan_files(tmp_path))

        assert all(p.is_absolute() for p in result)


class TestGetLanguage:
    def test_python(self) -> None:
        assert get_language(Path("script.py")) == "python"
        assert get_language(Path("stub.pyi")) == "python"

    def test_javascript(self) -> None:
        assert get_language(Path("app.js")) == "javascript"
        assert get_language(Path("component.jsx")) == "javascript"

    def test_typescript(self) -> None:
        assert get_language(Path("app.ts")) == "typescript"
        assert get_language(Path("component.tsx")) == "typescript"

    def test_java(self) -> None:
        assert get_language(Path("Main.java")) == "java"

    def test_kotlin(self) -> None:
        assert get_language(Path("Main.kt")) == "kotlin"
        assert get_language(Path("build.kts")) == "kotlin"

    def test_c(self) -> None:
        assert get_language(Path("main.c")) == "c"
        assert get_language(Path("header.h")) == "c"

    def test_cpp(self) -> None:
        assert get_language(Path("main.cpp")) == "cpp"
        assert get_language(Path("main.cc")) == "cpp"
        assert get_language(Path("header.hpp")) == "cpp"

    def test_go(self) -> None:
        assert get_language(Path("main.go")) == "go"

    def test_rust(self) -> None:
        assert get_language(Path("lib.rs")) == "rust"

    def test_returns_none_for_markdown(self) -> None:
        assert get_language(Path("README.md")) is None

    def test_returns_none_for_toml(self) -> None:
        assert get_language(Path("pyproject.toml")) is None

    def test_returns_none_for_yaml(self) -> None:
        assert get_language(Path("config.yaml")) is None
        assert get_language(Path("config.yml")) is None

    def test_returns_none_for_unknown(self) -> None:
        assert get_language(Path("image.png")) is None
        assert get_language(Path("binary.exe")) is None
        assert get_language(Path("noextension")) is None


class TestIsCodeFile:
    def test_python_is_code(self) -> None:
        assert is_code_file(Path("script.py")) is True

    def test_typescript_is_code(self) -> None:
        assert is_code_file(Path("app.ts")) is True

    def test_kotlin_is_code(self) -> None:
        assert is_code_file(Path("Main.kt")) is True

    def test_cpp_is_code(self) -> None:
        assert is_code_file(Path("main.cpp")) is True

    def test_go_is_code(self) -> None:
        assert is_code_file(Path("main.go")) is True

    def test_markdown_is_not_code(self) -> None:
        assert is_code_file(Path("README.md")) is False

    def test_toml_is_not_code(self) -> None:
        assert is_code_file(Path("pyproject.toml")) is False

    def test_yaml_is_not_code(self) -> None:
        assert is_code_file(Path("config.yaml")) is False

    def test_unknown_is_not_code(self) -> None:
        assert is_code_file(Path("image.png")) is False


class TestComputeFileHash:
    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        content = "def foo(): pass\n"
        f1.write_text(content)
        f2.write_text(content)

        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def foo(): pass")
        f2.write_text("def bar(): return 42")

        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_returns_string(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        assert compute_file_hash(f) == compute_file_hash(f)

    def test_hash_changes_after_modification(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        hash_before = compute_file_hash(f)
        f.write_text("x = 2")
        hash_after = compute_file_hash(f)
        assert hash_before != hash_after
