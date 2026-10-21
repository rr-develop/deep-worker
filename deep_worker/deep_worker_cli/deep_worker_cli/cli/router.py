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
    """Serve ``deep-worker --web`` without importing the agent stack.

    This is what makes the web UI reachable from a plain ``pip install
    deep-worker``: the server used to be startable only as
    ``python run.py`` from a git checkout, so a user without sources had
    no way in at all.

    Why a FLAG and not a positional subcommand
    ------------------------------------------
    The first positional of ``deep-worker`` is an agent NAME, so a
    positional ``web`` necessarily competes with any agent called
    ``web`` — one of them has to lose, silently. A flag has no such
    conflict, which is why ``--web`` is the supported spelling and the
    only one that appears in any help text.

    The pre-flag positional spelling was retired for exactly that
    reason: it made ``deep-worker web`` serve the UI on the default
    port instead of starting an agent named "web". Nothing invoked it
    (no systemd unit, crontab entry, script or doc; the servers running
    on this host are ``python3 -m deep_worker_cli.process.web_daemon
    --port N --listen-fd 3 --ready-fd 5``, PPID 1), so it was removed
    rather than deprecated, and the name is an agent's again.

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

    Returns:
        Exit code if this argv is the web server; ``None`` to continue routing.
    """
    if argv[:1] == ["--web"]:
        web_argv = argv[1:]
        invoked_as = "deep-worker --web"
    else:
        return None

    import argparse
    import importlib
    import os

    parser = argparse.ArgumentParser(
        prog=invoked_as,
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
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report whether a managed server is running on this port",
    )
    parser.epilog = (
        "examples:\n"
        "  deep-worker --web                 start in the background on the "
        "default port\n"
        "  deep-worker --web --port 8080     start on port 8080\n"
        "  deep-worker --web --status        is it running?\n"
        "  deep-worker --web --stop          stop the managed server\n"
        "  deep-worker --web --foreground    run attached, for debugging\n"
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    # ``--web start`` / ``--web status`` are the shapes people try first;
    # argparse would answer the bare "unrecognized arguments: start", which
    # reads like the flag itself is broken.
    _VERB_FLAGS = {"start": None, "stop": "--stop", "status": "--status", "restart": None}
    if web_argv[:1] and web_argv[0] in _VERB_FLAGS:
        flag = _VERB_FLAGS[web_argv[0]]
        hint = f"use '{flag}'" if flag else "the bare form already starts the server"
        print(
            f"{invoked_as}: '{web_argv[0]}' is not a verb here — {hint}.\n"
            f"Run '{invoked_as} --help' for the available controls.",
            file=sys.stderr,
        )
        return 2

    args = parser.parse_args(web_argv)

    # The resolver is imported through whichever name is available. From a
    # checkout ``deep_worker_cli.web`` does not exist (the web tree is a
    # sibling, staged into the wheel only at build time), so the source
    # layout is reached via the top-level ``app`` once web/ is on the path.
    # Locating web/ therefore has to come first, and it cannot use the
    # resolver it is trying to make importable.
    web_root = _locate_web_root()
    if web_root is None:
        print(
            "deep-worker --web: cannot locate the web application directory.\n"
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
        print(f"deep-worker --web: refusing to start: {exc}", file=sys.stderr)
        return 1

    from deep_worker_cli.process import web_daemon

    if args.status:
        return web_daemon.status(port)
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


def _dispatch_version(argv: list[str]) -> int | None:
    """Serve ``deep-worker --version`` without the agent stack.

    The flag did not exist anywhere, so it fell through to the agent parser
    and died with "unrecognized arguments" — reported from a host running
    0.0.21 from PyPI.

    The version is read from the INSTALLED distribution metadata, falling
    back to ``_version.py`` for a source checkout where no distribution is
    registered. That order matters: metadata is what ``pip show`` reports,
    so the two can never disagree for a pip user.

    Spelled as a FLAG only: a positional ``version`` would compete with an
    agent of that name, so ``deep-worker version`` keeps starting the agent
    called "version" and ``--version`` reports the version.

    The flag is honoured ANYWHERE in argv, not just first. While it was
    first-only, ``deep-worker <agent> --version`` fell through to the agent
    parser, whose own ``--version`` action still carried the ancestor's
    program name — so the same flag printed two different products
    depending on its position.

    Scanning is safe against options that take a VALUE: argparse refuses
    to consume ``--version`` as one (``--fork --version`` fails with
    "expected one argument", and even ``-r --version`` with ``nargs='?'``
    leaves the token alone), so in that position it is still the flag.

    The scan stops at ``--`` because argparse's own rule is that
    everything after it is positional: it reads ``deep-worker --
    --version`` as the agent NAME ``--version``. Honouring the flag there
    would report the version instead of starting that agent, contradicting
    the FLAG-only contract above — the same reason ``version`` as a bare
    positional keeps starting an agent.

    Returns:
        Exit code if this argv asked for the version; ``None`` otherwise.
    """
    asked = False
    for arg in argv:
        if arg == "--":
            break
        if arg in {"--version", "-V"}:
            asked = True
            break

    if not asked:
        return None

    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            resolved = version("deep-worker")
        except PackageNotFoundError:
            from deep_worker_cli._version import __version__ as resolved
    except Exception:  # pragma: no cover - defensive, metadata is stdlib
        from deep_worker_cli._version import __version__ as resolved

    print(f"deep-worker {resolved}")
    return 0


def _dispatch_help(argv: list[str]) -> int | None:
    """Answer ``--help`` even when the agent runtime cannot be imported.

    The real help text is the agent parser's: it is the single source of
    truth for the ~60 agent options, and duplicating a second listing here
    would guarantee the two drift apart. So this delegates to
    ``create_parser()`` and only takes over when that import fails.

    That fallback is not hypothetical. ``agent_command`` pulls the whole
    agent stack at module level (yaml, langchain, …), so in an install
    whose runtime dependencies are incomplete ``--help`` died with
    ``ModuleNotFoundError``. Help is precisely what a user reaches for
    when something else is already broken, so it must still answer —
    with the flags that need no runtime at all.

    Returns:
        Exit code if this argv asked for help; ``None`` otherwise.
    """
    if argv[:1] not in (["--help"], ["-h"]):
        return None

    try:
        from deep_worker_cli.cli.agent_command import create_parser
    except Exception as exc:  # noqa: BLE001 - any import failure must degrade
        # Degraded help: still a real help page on stdout with exit 0, so
        # `deep-worker --help | grep` keeps working in a broken install.
        # The runtime problem is reported on stderr, where it does not
        # corrupt the listing.
        print(
            f"warning: the agent runtime failed to import "
            f"({type(exc).__name__}: {exc}),\n"
            "so the per-agent options are omitted below.",
            file=sys.stderr,
        )
        print(
            "usage: deep-worker [AGENT_NAME] [options]\n"
            "\n"
            "A positional is always an agent NAME. The web UI and the "
            "version are\n"
            "flags, so an agent called 'web' or 'version' keeps working.\n"
            "\n"
            "options that need no agent runtime:\n"
            "  --web                 Serve the deep_worker web UI on "
            "127.0.0.1, detached\n"
            "                        from the terminal (--port N, --status, "
            "--stop,\n"
            "                        --foreground); see --web --help\n"
            "  --version, -V         Print the installed deep-worker version "
            "and exit\n"
            "\n"
            "examples:\n"
            "  deep-worker myagent             start (or attach to) an agent\n"
            "  deep-worker --web               start the web UI in the "
            "background\n"
            "  deep-worker --web --port 8080   serve on port 8080\n"
            "  deep-worker --web --status      is the web UI running?\n"
            "  deep-worker --web --stop        stop it\n"
            "  deep-worker --version           print the version\n"
        )
        return 0

    create_parser().print_help()
    return 0


def deep_worker_main() -> int:
    """Entry point for 'deep-worker' command (agent process).

    Returns:
        Exit code
    """
    code = _dispatch_lsp_bridge(sys.argv[1:])
    if code is not None:
        return code

    code = _dispatch_version(sys.argv[1:])
    if code is not None:
        return code

    code = _dispatch_help(sys.argv[1:])
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
