"""Configuration constants and dataclasses for semantic_search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# File extension sets
# ---------------------------------------------------------------------------

INDEXED_EXTENSIONS: frozenset[str] = frozenset({
    # Python
    ".py", ".pyi",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx",
    # JVM languages
    ".java", ".kt", ".kts",
    # C / C++
    ".c", ".cpp", ".cc", ".h", ".hpp",
    # Systems languages
    ".go", ".rs",
    # Docs & config
    ".md", ".toml", ".yaml", ".yml",
})

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".kts",
    ".c", ".cpp", ".cc", ".h", ".hpp",
    ".go", ".rs",
})

# ---------------------------------------------------------------------------
# Language names for tree-sitter CodeSplitter
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    ".py":  "python",
    ".pyi": "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt":  "kotlin",
    ".kts": "kotlin",
    ".c":   "c",
    ".h":   "c",
    ".cpp": "cpp",
    ".cc":  "cpp",
    ".hpp": "cpp",
    ".go":  "go",
    ".rs":  "rust",
}

# ---------------------------------------------------------------------------
# Directories excluded from indexing
# ---------------------------------------------------------------------------

EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".vector_index",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "target",    # Rust / Maven
    "out",       # Java / Kotlin
    ".gradle",
    ".idea",
    ".vscode",
    "vendor",    # Go / Ruby
    ".cache",
    "coverage",
    "htmlcov",
    # host-specific virtual env names should be added by the operator
})

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexConfig:
    """All tunable parameters for the indexing pipeline."""

    # CodeSplitter settings (code files)
    chunk_lines: int = 40
    chunk_lines_overlap: int = 5
    max_chars: int = 2000

    # HierarchicalNodeParser chunk sizes in tokens
    # parent_chunk_size > child_chunk_size
    parent_chunk_size: int = 512
    child_chunk_size: int = 128

    # AutoMergingRetriever: fraction of children that triggers parent merge
    simple_ratio_thresh: float = 0.5

    # Hugging Face embedding model name
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class RepoIndex:
    """Represents the index storage location for a single repository."""

    repo_path: Path

    @property
    def index_dir(self) -> Path:
        """Root directory for this repo's vector index."""
        return self.repo_path / ".vector_index"

    @property
    def lancedb_uri(self) -> str:
        """URI passed to LanceDBVectorStore."""
        return str(self.index_dir / "lancedb")

    @property
    def docstore_path(self) -> Path:
        """Directory for SimpleDocumentStore persistence."""
        return self.index_dir / "docstore"

    @property
    def hashes_path(self) -> Path:
        """JSON file tracking per-file content hashes."""
        return self.index_dir / "file_hashes.json"

    def exists(self) -> bool:
        """Return True if the index directory exists."""
        return self.index_dir.exists()
