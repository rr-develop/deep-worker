"""``python -m deep_worker_cli.lsp.bridge_main`` — the bridge's own entry.

The ``deep-worker`` wrapper execs this module for ``lsp bridge`` instead
of ``deep_worker_cli.cli.agent_command``. That is the whole point: ``-m``
IMPORTS its target, and a module's import block runs before any of its
code, so the agent stack's eager ``langchain`` → ``transformers`` →
``torch`` chain is paid merely by naming ``agent_command`` as the target.
No hook inside ``agent_command.main()`` can be early enough. Measured on
2026-10-09: eight live bridges via the old path, 162 heavy lines in
``/proc/<pid>/maps`` and 263–663 MB RSS each, for modules the bridge
never touches.

This module must therefore stay import-light. It deliberately reuses the
router's ``_dispatch_lsp_bridge`` rather than re-deriving the argument
parsing: two copies of the same contract would drift, and the router's
copy is the one the installed console-script path still uses. Importing
``deep_worker_cli.cli.router`` is safe — it holds no module-level agent
imports; ``agent_command`` is imported lazily inside its functions.

The argv this receives is already stripped of the ``lsp bridge`` pair by
the wrapper, so it is re-prefixed here to hand the dispatcher the exact
shape it documents.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Serve the bridge; return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)

    from deep_worker_cli.cli.router import _dispatch_lsp_bridge

    code = _dispatch_lsp_bridge(["lsp", "bridge", *args])
    if code is None:
        # Unreachable via the wrapper, which only execs this module for
        # the bridge. Treated as a loud failure rather than a silent
        # zero: a bridge that did not start must not look like one that
        # started and found nothing to do.
        print(
            "error: bridge dispatch rejected its own argv: "
            f"{args!r}",
            file=sys.stderr,
        )
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
