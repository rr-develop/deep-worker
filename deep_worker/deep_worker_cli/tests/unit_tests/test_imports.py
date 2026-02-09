"""Test importing files."""


def test_imports() -> None:
    """Test importing deepagents modules."""
    from deep_worker_cli import (
        agent,
        integrations,
    )
    from deep_worker_cli.main import cli_main
