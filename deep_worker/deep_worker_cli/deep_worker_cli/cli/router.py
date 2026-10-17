"""CLI command router for deep_worker.

Routes between agent process launcher and client connector based on context.
"""

import socket
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


def _dispatch_web(
    argv: list[str], *, listener: socket.socket | None = None, ready: socket.socket | None = None
) -> int | None:
    """Serve ``deep-worker web`` without importing the agent stack.

    This is what makes the web UI reachable from a plain ``pip install
    deep-worker``: the server used to be startable only as
    ``python run.py`` from a git checkout, so a user without sources had
    no way in at all.

    Why the interception lives here, like ``lsp bridge``'s: reaching
    ``agent_command``'s own subcommand registry means importing that
    module, and its module-level ``from langchain...`` chain pulls
    ``langchain_text_splitters`` -> ``transformers`` -> ``torch``
    eagerly. The web server needs none of it.

    ``sys.path`` and the ``app`` package name
    -----------------------------------------
    ``uvicorn`` is given the import string ``"app.main:app"``, and ~20
    modules under ``web/app`` import each other absolutely (``from
    app.services... import``). Both require ``app`` to resolve as a
    TOP-LEVEL name. So this adds the ``web/`` directory to ``sys.path``
    — exactly what ``run.py:_bootstrap_sys_path`` already does with its
    own ``web_root`` from source. The alternative, rewriting every
    intra-package import to a relative form, would change ~20 files that
    this task never touched and that must keep working from source.

    The directory is located through ``app.paths``, the single resolver,
    so this works identically from a checkout and from inside the wheel.

    Matching is exact on the first token, so an agent named ``web``
    is NOT captured by ``deep-worker web`` alone — that form IS the
    server, and an agent of that name would have to be started by the
    normal route. Every other invocation falls through unchanged.

    Returns:
        Exit code if this argv is the web server; ``None`` to continue routing.
    """
    if argv[:1] != ["web"]:
        return None

    import argparse
    import importlib
    import os

    parser = argparse.ArgumentParser(
        prog="deep-worker web",
        description="Run the deep_worker web UI on 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port to bind (default: derived from the checkout, 7887 when installed)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes (development only)",
    )
    parser.add_argument("--foreground", action="store_true", help="Run attached to the terminal for debugging")
    parser.add_argument("--stop", action="store_true", help="Stop the managed web server on this port")
    args = parser.parse_args(argv[1:])

    # The resolver is imported through whichever name is available. From a
    # checkout ``deep_worker_cli.web`` does not exist (the web tree is a
    # sibling, staged into the wheel only at build time), so the source
    # layout is reached via the top-level ``app`` once web/ is on the path.
    # Locating web/ therefore has to come first, and it cannot use the
    # resolver it is trying to make importable.
    web_root = _locate_web_root()
    if web_root is None:
        print(
            "deep-worker web: cannot locate the web application directory.\n"
            "An installed distribution ships it as deep_worker_cli/web; a "
            "checkout has it at <repo>/deep_worker_web. Neither was found.",
            file=sys.stderr,
        )
        return 1

    if str(web_root) not in sys.path:
        sys.path.insert(0, str(web_root))

    # From a checkout the server also needs its sibling packages, the same
    # set run.py bootstraps. Installed, deep_worker_cli is already
    # importable and there is no sibling tree — hence the None branch.
    paths = importlib.import_module("app.paths")
    tree_root = paths.source_tree_root()
    if tree_root is not None:
        for sibling in (
            tree_root / "deep_worker" / "deep_worker_cli",
            tree_root / "deep_worker" / "deepagents",
        ):
            text = str(sibling)
            if sibling.is_dir() and text not in sys.path:
                sys.path.insert(0, text)

    import uvicorn

    settings = importlib.import_module("app.config").settings

    # An explicit --port must be written back onto the SETTINGS object, not
    # merely handed to uvicorn.run(). ``app.main`` reads ``settings.port``
    # independently — for its startup banner, its /health payload and
    # the URL it advertises — so passing the port to uvicorn alone produced
    # a server listening on the requested port while reporting the default
    # one (measured: bound to 39771, announced 7888, which is a DIFFERENT
    # live server on this host). Mutating settings keeps the whole process
    # on one answer.
    if args.port is not None:
        settings.port = args.port
    port = settings.port

    # The loopback invariant is not negotiable and is not exposed as a
    # flag; see app/config.py validate_security().
    try:
        settings.validate_security()
    except ValueError as exc:
        print(f"deep-worker web: refusing to start: {exc}", file=sys.stderr)
        return 1

    from deep_worker_cli.process import web_daemon

    if args.stop:
        return web_daemon.stop(port)
    if listener is not None and ready is not None:
        return web_daemon.serve(settings.host, port, settings.debug, listener, ready)
    if not args.foreground:
        return web_daemon.start(
            settings.host, port, args.reload or bool(os.environ.get("DEEP_WORKER_WEB_RELOAD"))
        )

    print(f"🔒 Binding to {settings.host}:{port} (loopback only)")
    print(f"📂 Active checkout: {tree_root or 'installed (no checkout)'}")
    print(f"📝 Open http://{settings.host}:{port} in your browser")

    # uvicorn is invoked PROGRAMMATICALLY — no shell-out to run.py, which
    # does not exist in an installed distribution.
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=port,
        reload=args.reload or bool(os.environ.get("DEEP_WORKER_WEB_RELOAD")),
        log_level="debug" if settings.debug else "info",
    )
    return 0


def _locate_web_root() -> "object | None":
    """Directory containing the ``app`` package and ``static/``, or None.

    Checked in installed-first order because that is the layout this
    subcommand exists for, and because a checkout is identifiable only
    relative to this file's own location.
    """
    import importlib.resources
    from pathlib import Path

    # Installed: <site-packages>/deep_worker_cli/web/{app,static}. Anchored
    # on the regular package ``...web.app`` rather than the namespace
    # package ``...web``: only the former yields a real filesystem path.
    try:
        app_dir = Path(str(importlib.resources.files("deep_worker_cli.web.app")))
        if (app_dir.parent / "static").is_dir():
            return app_dir.parent
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError):
        pass

    # From source: <repo>/deep_worker_web, a sibling of deep_worker/.
    # This file is <repo>/deep_worker/deep_worker_cli/deep_worker_cli/cli/.
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "deep_worker_web"
    if (candidate / "app").is_dir() and (candidate / "static").is_dir():
        return candidate

    return None


def deep_worker_main() -> int:
    """Entry point for 'deep-worker' command (agent process).

    Returns:
        Exit code
    """
    code = _dispatch_lsp_bridge(sys.argv[1:])
    if code is not None:
        return code

    code = _dispatch_web(sys.argv[1:])
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
