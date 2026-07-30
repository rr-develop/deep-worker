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

    # Chunk sizes in tokens (flat structure by default)
    # Set parent_chunk_size to enable hierarchical retrieval with AutoMergingRetriever.
    # If None, uses flat structure with only child_chunk_size (default: LlamaIndex 1024).
    parent_chunk_size: int | None = None
    child_chunk_size: int = 1024

    # AutoMergingRetriever: fraction of children that triggers parent merge
    # (only used when parent_chunk_size is not None)
    simple_ratio_thresh: float = 0.5

    # Hugging Face embedding model name.
    # Multilingual model — supports Russian/English cross-lingual search.
    # See docs/SEMANTIC_SEARCH_DESIGN.md § Embeddings for alternatives.
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Number of nodes WRITTEN to the vector store per transaction.
    # CRITICAL: nodes MUST be written in batches, never one-at-a-time —
    # LanceDB creates a new data fragment + manifest version per add() call,
    # so per-node inserts degrade to O(n) transactions (thousands of .txn
    # files) and crawl at ~1 insert/sec on large histories. Batching cuts
    # this to ceil(n / index_batch_size) transactions. This is the WRITE
    # batch size and is INDEPENDENT of the progress step below. See
    # docs/DESIGN_FIX_BATCHED_INDEX_INSERT.md.
    index_batch_size: int = 1000

    # How many nodes to embed per progress tick. This is the PROGRESS
    # granularity ONLY — it has nothing to do with when batches are written
    # to LanceDB (index_batch_size). Smaller → smoother bar / more accurate
    # ETA, at the cost of slightly less-batched embedding calls. Embedding a
    # slice is cheap; the write stays batched at index_batch_size.
    embed_progress_step: int = 64

    # How many message groups to chunk per "Analyzing history…" progress tick
    # (Phase 1, before embedding). This is the PROGRESS granularity for the
    # chunking pass ONLY — it keeps the long tokenization/chunking pass from
    # sitting silent on large histories. Deliberately a SMALL step and
    # INDEPENDENT of index_batch_size (the write batch). See
    # docs/DESIGN_HISTORY_INDEX_ANALYZE_PHASE_PROGRESS.md.
    analyze_progress_step: int = 32


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
