"""CLI command router for deep_worker.

Routes between agent process launcher and client connector based on context.
"""

import sys


def _dispatch_lsp_bridge(argv: list[str]) -> int | None:
    """Serve ``deep-worker lsp bridge`` without importing the agent stack.

    The bridge is a *thin* per-agent stdio MCP proxy: it ensures the shared
    per-project LSP service is up and forwards tool calls to it over gRPC.
    It needs none of the agent runtime — no provider chain, no LangChain
    middleware, no checkpointer.

    Why this lives in the router rather than in ``agent_command``'s own
    (already lazy) subcommand registry: reaching that registry requires
    importing ``agent_command`` as a module, and its module-level
    ``from langchain...`` chain pulls ``langchain_text_splitters``, which
    imports ``transformers`` and ``torch`` eagerly. That tax (~4s, ~700 MB
    RSS) is paid per agent start, by a process that never uses any of it.
    Intercepting here — before that import — is what keeps the bridge thin.

    Matching is exact on the first two tokens, so an agent named ``lsp``
    (``deep-worker lsp``) is not captured, and every other invocation,
    including the remaining ``lsp`` subcommands, falls through unchanged.

    Returns:
        Exit code if this argv is the bridge; ``None`` to continue routing.
    """
    if argv[:2] != ["lsp", "bridge"]:
        return None

    import argparse

    from deep_worker_cli.lsp.service.service_cli import cmd_bridge

    parser = argparse.ArgumentParser(prog="deep-worker lsp bridge")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--registry-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv[2:])
    args.lsp_command = "bridge"
    return cmd_bridge(args)


def deep_worker_main() -> int:
    """Entry point for 'deep-worker' command (agent process).

    Returns:
        Exit code
    """
    code = _dispatch_lsp_bridge(sys.argv[1:])
    if code is not None:
        return code

    from deep_worker_cli.cli.agent_command import main as agent_main

    return agent_main()


def deep_worker_cli_main() -> int:
    """Entry point for 'deep-worker-cli' command (client connector).

    Returns:
        Exit code
    """
    from deep_worker_cli.cli.client_command import main as client_main

    return client_main()


if __name__ == "__main__":
    # Determine which command based on sys.argv[0]
    if "deep-worker-cli" in sys.argv[0]:
        sys.exit(deep_worker_cli_main())
    else:
        sys.exit(deep_worker_main())
