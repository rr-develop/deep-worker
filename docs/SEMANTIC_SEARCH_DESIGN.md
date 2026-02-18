# Semantic Search Design

**Status:** Implemented
**Date:** 2026-02-18

---

## Overview

A semantic search module for repositories based on LlamaIndex + LanceDB. Designed for use by LangChain agents, as well as manually via CLI. Each repository stores its own vector index inside itself.

---

## File Structure

```
<repository root>/
├── semantic_search/                   ← package (modeled after arpyc/)
│   ├── pyproject.toml
│   ├── semantic_search/               ← Python package
│   │   ├── __init__.py                ← public API
│   │   ├── py.typed                   ← PEP 561 marker
│   │   ├── exceptions.py              ← IndexNotFoundError, IndexingError
│   │   ├── config.py                  ← constants + IndexConfig, RepoIndex
│   │   ├── scanner.py                 ← filesystem traversal
│   │   ├── indexer.py                 ← indexing pipeline
│   │   └── retriever.py               ← search pipeline
│   └── tests/                         ← tests (separate from the code)
│       ├── unit/                      ← 101 tests, fast (MockEmbedding + mocked LlamaIndex)
│       │   ├── test_config.py
│       │   ├── test_scanner.py
│       │   ├── test_indexer.py
│       │   └── test_retriever.py
│       └── integration/               ← 17 tests, real LanceDB + MockEmbedding
│           └── test_pipeline.py
├── tools/
│   └── semantic_search.py             ← LangChain @tool wrappers for agents
├── scripts/
│   └── index_repo.py                  ← CLI for manual indexing
└── requirements-semantic-search.txt   ← dependencies

# Storage inside each repository:
<project root>/
└── .vector_index/
    ├── lancedb/                        ← table with vectors (LanceDB)
    ├── docstore/
    │   └── docstore.json               ← SimpleDocumentStore (all hierarchy nodes)
    └── file_hashes.json                ← MD5 hashes of files for incrementality
```

`.vector_index/` is added to the `.gitignore` of each repository.

---

## Dependencies

```
llama-index-core>=0.10.0
llama-index-vector-stores-lancedb>=0.4.0
llama-index-embeddings-huggingface>=0.4.0
lancedb>=0.20.0
tree-sitter-language-pack>=0.13.0
pandas>=2.0.0
```

Installation:
```bash
pip install -r requirements-semantic-search.txt
```

---

## Indexed File Types

| Category     | Extensions                                      | Splitter       |
|--------------|-------------------------------------------------|----------------|
| Python       | `.py`, `.pyi`                                   | CodeSplitter   |
| JS/TS        | `.js`, `.ts`, `.jsx`, `.tsx`                    | CodeSplitter   |
| JVM          | `.java`, `.kt`, `.kts`                          | CodeSplitter   |
| C/C++        | `.c`, `.cpp`, `.cc`, `.h`, `.hpp`               | CodeSplitter   |
| Systems      | `.go`, `.rs`                                    | CodeSplitter   |
| Docs/Config  | `.md`, `.toml`, `.yaml`, `.yml`                 | SentenceSplitter |

**Exclusions:** `.git/`, `node_modules/`, `__pycache__/`, `dist/`, `build/`, `.vector_index/`, `.venv/`, `venv/`, `target/`, `vendor/` and others (see `config.EXCLUDED_DIRS`).

---

## Indexing Architecture

### Component Graph

```
scan_files(repo_path)
    ↓ Iterator[Path] — only indexable files
_load_document(file_path, repo_path)
    ↓ Document with metadata: file_path, language, extension, repo_path
_split_document(doc, config)
    ├── CodeSplitter(language, chunk_lines=40) — for code (tree-sitter)
    └── SentenceSplitter(chunk_size=128)       — for text/configs
    ↓ leaf_nodes
_make_hierarchy(doc, leaf_nodes, config)
    ├── SentenceSplitter(chunk_size=512) → parent_nodes
    └── links leaf → parent via RelatedNodeInfo
    ↓ all_nodes = parent_nodes + leaf_nodes
storage_context.docstore.add_documents(all_nodes)   ← all nodes for AutoMerging
index.insert_nodes(leaf_nodes)                       ← only leaves into LanceDB
```

### Incremental Indexing

The core of incrementality is `file_hashes.json`. On each run:

| File          | Action                                                     |
|---------------|------------------------------------------------------------|
| **New**       | `_add_document()` → embed → insert into LanceDB           |
| **Modified**  | `delete_ref_doc()` + `_add_document()`                     |
| **Deleted**   | `delete_ref_doc()` — from LanceDB and docstore             |
| **Unchanged** | skipped entirely                                           |

`VectorStoreIndex.delete_ref_doc(doc_id, delete_from_docstore=True)` atomically deletes all nodes of a document from both stores.

### LanceDB Modes

| Situation                     | `mode`       |
|-------------------------------|--------------|
| First run (no hashes)         | `"overwrite"` — creates the table |
| Subsequent run                | `"append"` — preserves existing data |
| `reset=True`                  | Directory is deleted → `"overwrite"` |

---

## Search Architecture

### Hierarchical Retrieval (AutoMergingRetriever)

```
query
    ↓ embedding
LanceDB ANN search (over leaf_nodes, top_k * 2 candidates)
    ↓ leaf NodeWithScore[]
AutoMergingRetriever
    ├── Groups leaves by parent_node
    ├── If the share of leaves of one parent ≥ simple_ratio_thresh (0.5)
    │       → replaces the group with the parent node (more context)
    └── Otherwise → returns the leaves themselves
    ↓ top_k results
SearchResult(file_path, score, content, language, node_type)
```

`node_type`:
- `"leaf"` — a precise small fragment (function, paragraph)
- `"parent"` — extended context (class, section)

### Metadata

Each `SearchResult` contains:
- `file_path` — absolute path to the source file
- `score` — cosine similarity `[0, 1]`
- `content` — the fragment text
- `language` — `"python"`, `"typescript"`, etc., or `None` for non-code
- `node_type` — `"leaf"` or `"parent"`

---

## API

### Package `semantic_search`

```python
from semantic_search import index_repository, search, IndexConfig

# Indexing
stats = index_repository(
    repo_path=Path("/path/to/my_project"),
    config=IndexConfig(),   # optional, sensible defaults exist
    reset=False,            # True = reindex from scratch
)
# stats.added, stats.updated, stats.deleted, stats.skipped, stats.total_nodes

# Search
results = search(
    query="HTTP authentication middleware",
    repo_path=Path("/path/to/my_project"),
    top_k=5,
)
for r in results:
    print(f"{r.file_path}  score={r.score:.3f}  [{r.language}]")
    print(r.content)
```

### LangChain Tools (`tools/semantic_search.py`)

```python
from tools.semantic_search import SEMANTIC_SEARCH_TOOLS
# SEMANTIC_SEARCH_TOOLS = [index_repository_tool, semantic_search_tool]

agent = create_agent(tools=[*SEMANTIC_SEARCH_TOOLS, ...])
```

### CLI (`scripts/index_repo.py`)

```bash
# Incremental indexing
python scripts/index_repo.py /path/to/my_project

# Full reindexing
python scripts/index_repo.py /path/to/my_project --reset

# Statistics without reindexing
python scripts/index_repo.py /path/to/my_project --stats
```

---

## Embeddings

**Model:** `BAAI/bge-small-en-v1.5`

| Parameter          | Value                  |
|--------------------|------------------------|
| Model size         | ~130 MB                |
| Storage            | `~/.cache/huggingface/`|
| Vector dimensionality | 384                 |
| Works offline      | Yes (after the first download) |

Tests use `MockEmbedding(embed_dim=16/32)` — it does not require downloading the model.

---

## `IndexConfig` Parameters

| Parameter             | Default                   | Description                                 |
|-----------------------|---------------------------|---------------------------------------------|
| `chunk_lines`         | `40`                      | Lines per CodeSplitter chunk                |
| `chunk_lines_overlap` | `5`                       | Overlap in lines                            |
| `max_chars`           | `2000`                    | Maximum chunk size in characters            |
| `parent_chunk_size`   | `512`                     | Parent node size in tokens                  |
| `child_chunk_size`    | `128`                     | Child node size in tokens                   |
| `simple_ratio_thresh` | `0.5`                     | Merge threshold in AutoMergingRetriever     |
| `embedding_model`     | `"BAAI/bge-small-en-v1.5"` | HuggingFace model name                    |

---

## Testing

```bash
cd semantic_search

# All tests (118)
python3 -m pytest tests/ -v

# Unit tests only (fast, ~7 sec)
python3 -m pytest tests/unit/ -v

# Integration tests only (MockEmbedding, real LanceDB)
python3 -m pytest tests/integration/ -v
```

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| Kotlin tree-sitter | The grammar is supported, but less mature than Python/JS |
| Multilingual queries | `bge-small-en` is optimized for English; non-English queries work worse |
| First indexing | Slow (all files + model download); subsequent runs are fast (diff only) |
| LanceDB `table_names()` | Deprecated in v0.29+, emits warnings — does not affect operation |
