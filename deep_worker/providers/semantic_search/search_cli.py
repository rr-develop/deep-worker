"""Command-line interface for semantic search over vector indexes.

This module provides a CLI tool for searching indexed repositories using
natural language queries. It automatically discovers vector indexes by
walking up the directory tree from the current or specified directory.

Usage:
    # Search with a query string
    python -m semantic_search.search_cli -q "find authentication code"

    # Search with a query from a file
    python -m semantic_search.search_cli -qf query.txt

    # Search in a specific repository
    python -m semantic_search.search_cli -q "session management" -d /path/to/repo

    # Retrieve more results
    python -m semantic_search.search_cli -q "error handling" -k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .exceptions import IndexNotFoundError
from .retriever import SearchResult, search


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace with fields:
            - query: str | None
            - query_file: Path | None
            - directory: Path | None
            - top_k: int

    Raises:
        SystemExit: If arguments are invalid or mutually exclusive.
    """
    parser = argparse.ArgumentParser(
        description="Search code repositories using semantic similarity.",
        epilog="Either -q or -qf must be provided, but not both.",
    )

    # Query input (mutually exclusive group)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "-q",
        "--query",
        type=str,
        help="Search query as a string",
    )
    query_group.add_argument(
        "-qf",
        "--query-file",
        type=Path,
        dest="query_file",
        help="Path to file containing the search query",
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Repository directory to search (default: auto-discover from current directory)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return (default: 5)",
    )

    return parser.parse_args(argv)


def load_query_from_file(query_file: Path) -> str:
    """Load query text from a file.

    Reads the file, strips whitespace, and joins multiple lines into
    a single query string.

    Args:
        query_file: Path to the query file.

    Returns:
        Query string with lines joined by spaces.

    Raises:
        FileNotFoundError: If the query file does not exist.
        ValueError: If the query file is empty after stripping whitespace.
    """
    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")

    text = query_file.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Query file is empty: {query_file}")

    # Join multiple lines into a single query
    # Replace newlines with spaces and collapse multiple spaces
    return " ".join(text.split())


def find_vector_index(start_dir: Path) -> Path:
    """Find a vector index by walking up the directory tree.

    Searches for a `.vector_index` directory starting from *start_dir*
    and walking up to parent directories until one is found or the
    filesystem root is reached.

    Args:
        start_dir: Directory to start searching from.

    Returns:
        Path to the repository root containing `.vector_index/`.

    Raises:
        IndexNotFoundError: If no index is found anywhere in the parent chain.
    """
    current = start_dir.resolve()

    # Walk up the directory tree
    while True:
        index_dir = current / ".vector_index"
        if index_dir.exists() and index_dir.is_dir():
            return current

        # Stop at filesystem root
        parent = current.parent
        if parent == current:
            # Reached root without finding an index
            raise IndexNotFoundError(
                f"No vector index found for {start_dir} or any parent directory.\n"
                f"Run index_repository() first to create an index."
            )

        current = parent


def format_result(result: SearchResult, index: int, repo_path: Path) -> str:
    """Format a single search result for display.

    Args:
        result: The search result to format.
        index: Result number (1-based).
        repo_path: Repository root path for computing relative paths.

    Returns:
        Formatted string representation of the result.
    """
    # Compute relative path for nicer display
    try:
        file_path = Path(result.file_path)
        if file_path.is_absolute():
            rel_path = file_path.relative_to(repo_path)
        else:
            rel_path = file_path
    except ValueError:
        # If file_path is not relative to repo_path, use absolute path
        rel_path = Path(result.file_path)

    # Format score as percentage
    score_pct = result.score * 100

    # Build the output
    lines = [
        f"\n{'─' * 80}",
        f"Result {index}: {rel_path}",
        f"Score: {score_pct:.1f}%  |  Language: {result.language or 'text'}  |  Type: {result.node_type}",
        f"{'─' * 80}",
        result.content,
    ]

    return "\n".join(lines)


def main() -> int:
    """Main entry point for the search CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    try:
        args = parse_args()

        # Load query from file or use string
        if args.query_file is not None:
            query_text = load_query_from_file(args.query_file)
        else:
            query_text = args.query

        # Determine repository path
        if args.directory is not None:
            start_dir = args.directory.resolve()
        else:
            start_dir = Path.cwd()

        # Find the vector index
        repo_path = find_vector_index(start_dir)

        # Perform search
        results = search(
            query=query_text,
            repo_path=repo_path,
            top_k=args.top_k,
        )

        # Display results
        if not results:
            print("No results found for your query.")
            return 0

        print(f"\nFound {len(results)} results for: {query_text}")
        print(f"Repository: {repo_path}\n")

        for idx, result in enumerate(results, start=1):
            print(format_result(result, idx, repo_path))
            # Add separator between results (but not after the last one)
            if idx < len(results):
                print(f"\n{'~' * 64}\n")

        return 0

    except IndexNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nSearch cancelled by user.", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
