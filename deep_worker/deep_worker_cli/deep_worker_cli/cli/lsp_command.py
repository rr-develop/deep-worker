"""``deep-worker lsp`` subcommand — manages the optional LSP navigation feature.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md`` §4.2 / §10 (decision #1).

    deep-worker lsp enable  [--scope user|project|agent:<name>] [--force]
    deep-worker lsp disable [--scope ...]
    deep-worker lsp status  [--scope ...]

The ``mcp.json`` file is created with mode ``0o600`` in a ``0o700`` directory.

A convenience flag at agent creation is intentionally NOT added in v1 —
enabling stays an explicit operation on the config.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from deep_worker_cli.cli.cwd import invocation_cwd
from deep_worker_cli.lsp.enable import (
    SERENA_SERVER_NAME,
    disable_lsp_at_path,
    enable_lsp_at_path,
    is_lsp_enabled_at_path,
)
from deep_worker_cli.lsp.gradle_init import (
    GRADLE_INIT_SCRIPT_NAME,
    InstallState,
    default_init_d_dir,
    gradle_init_state,
    install_gradle_init_script,
    is_gradle_init_installed,
    uninstall_gradle_init_script,
)
from deep_worker_cli.lsp.serena_config import SERENA_COMMAND, resolve_workspace_root
from deep_worker_cli.lsp.service.service_cli import (
    dispatch_service_command,
    register_service_subcommands,
)

_INSTALL_HINT = "uv tool install -p 3.13 serena-agent"


def _scope_to_path(scope: str) -> Path:
    """The same scope contract as ``deep-worker mcp``."""
    home = Path.home()
    if scope == "user":
        return home / ".deepagents" / "mcp.json"
    if scope == "project":
        return resolve_workspace_root(invocation_cwd()) / ".deepagents" / "mcp.json"
    if scope.startswith("agent:"):
        name = scope.split(":", 1)[1].strip()
        if not name:
            raise SystemExit("--scope agent:<name> requires a non-empty agent name")
        return home / ".deepagents" / name / "mcp.json"
    raise SystemExit(
        f"unknown --scope {scope!r}; expected user | project | agent:<name>"
    )


def _config_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "config", None)
    if explicit:
        return Path(explicit)
    return _scope_to_path(getattr(args, "scope", "project"))


def _workspace_root(args: argparse.Namespace) -> Path:
    explicit_root = getattr(args, "project_root", None)
    return resolve_workspace_root(invocation_cwd(), explicit=explicit_root)


def setup_lsp_parser(subparsers, make_help_action=None) -> None:
    """Register the ``lsp`` group on the top-level parser."""
    lsp = subparsers.add_parser(
        "lsp", help="Manage optional LSP code-navigation (Serena MCP)"
    )
    lsp_sub = lsp.add_subparsers(dest="lsp_command")

    for name, help_text in (
        ("enable", "Enable LSP navigation for a scope"),
        ("disable", "Disable LSP navigation for a scope"),
        ("status", "Show whether LSP navigation is enabled"),
    ):
        p = lsp_sub.add_parser(name, help=help_text)
        p.add_argument(
            "--scope",
            default="project",
            help="project (default) | user | agent:<name>",
        )
        p.add_argument(
            "--config",
            default=None,
            help="explicit path to mcp.json (overrides --scope)",
        )
        if name == "enable":
            p.add_argument(
                "--project-root",
                default=None,
                help="workspace root Serena should open (default: git root of cwd)",
            )
            p.add_argument(
                "--force",
                action="store_true",
                help="overwrite an existing serena entry",
            )
            p.add_argument(
                "--no-auto-approve",
                action="store_true",
                help="require HITL approval for Serena tools (default: auto-approved)",
            )
            p.add_argument(
                "--no-gradle-init",
                action="store_true",
                help=(
                    "skip installing the Kotlin/Android Gradle init script into "
                    "$GRADLE_USER_HOME/init.d (cross-file navigation in AGP "
                    "projects stays broken without it)"
                ),
            )
            p.add_argument(
                "--no-detect-languages",
                action="store_true",
                help=(
                    "skip reconciling .serena/project.yml with the languages "
                    "actually present in the project"
                ),
            )

    languages = lsp_sub.add_parser(
        "languages",
        help="Show or change the languages Serena starts servers for",
    )
    languages.add_argument(
        "--project-root",
        default=None,
        help="workspace root (default: git root of cwd)",
    )
    languages.add_argument(
        "--add",
        action="append",
        metavar="LANG",
        help="enable a language (repeatable)",
    )
    languages.add_argument(
        "--remove",
        action="append",
        metavar="LANG",
        help="disable a language (repeatable)",
    )
    languages.add_argument(
        "--detect",
        action="store_true",
        help=(
            "add every language above the 10%% share threshold "
            "(respects .gitignore)"
        ),
    )

    register_service_subcommands(lsp_sub)


def handle_lsp_command(args: argparse.Namespace) -> int:
    command = getattr(args, "lsp_command", None)
    if command == "enable":
        return _cmd_enable(args)
    if command == "disable":
        return _cmd_disable(args)
    if command == "status":
        return _cmd_status(args)
    if command == "languages":
        return _cmd_languages(args)
    service_code = dispatch_service_command(args)
    if service_code is not None:
        return service_code
    print(
        "Usage: deep-worker lsp {enable|disable|status} [--scope ...]\n"
        "       deep-worker lsp languages [--add LANG] [--remove LANG] "
        "[--detect]\n"
        "       deep-worker lsp {serve|stop|service-status} [--project-root DIR]"
    )
    return 2


def _warn_if_serena_missing() -> bool:
    if shutil.which(SERENA_COMMAND):
        return True
    print(
        f"warning: '{SERENA_COMMAND}' not found in PATH — tools will not appear "
        f"until it is installed:\n    {_INSTALL_HINT}"
    )
    return False


def _restart_hint(project_root: Path) -> str:
    """A7: exact safe command, without self-serving restart."""
    return (
        "changes apply after the shared service is stopped:\n"
        f"    deep-worker lsp stop --project-root {project_root}"
    )


def _reconcile_languages(project_root: Path, *, quiet_failure: bool) -> int:
    """Write the detected languages into ``project.yml`` (A4).

    Called both from ``enable`` (there a detection failure must not
    break the command — languages are not its main goal), and from the
    explicit ``languages --detect`` (there failure must be loud: the user
    asked for exactly this, and "success without changes" would be a lie).
    """
    from deep_worker_cli.lsp.languages import (
        LanguageConfigError,
        apply_language_changes,
        create_config,
        detected_languages,
        language_facts,
        merge_languages,
        project_config_path,
    )

    try:
        facts = language_facts(project_root)
        detected = detected_languages(
            facts.get("composition", {}), facts.get("priority", {})
        )
        if not detected:
            msg = (
                f"no supported source files found under {project_root}; "
                f"languages left unchanged"
            )
            if quiet_failure:
                print(f"warning: {msg}")
                return 0
            print(f"error: {msg}")
            return 1

        config = project_config_path(project_root)
        if not config.exists():
            create_config(project_root, detected)
            print(f"created {config} with languages: {', '.join(detected)}")
            print(_restart_hint(project_root))
            return 0

        from deep_worker_cli.lsp.languages import read_languages

        merged = merge_languages(read_languages(config), detected)
        if apply_language_changes(config, desired=merged):
            print(f"languages updated in {config}: {', '.join(merged)}")
            print(_restart_hint(project_root))
        else:
            print(f"languages already up to date: {', '.join(merged)}")
        return 0
    except LanguageConfigError as e:
        if quiet_failure:
            print(f"warning: language detection skipped ({e})")
            return 0
        print(f"error: {e}")
        return 1


def _cmd_languages(args: argparse.Namespace) -> int:
    """``deep-worker lsp languages [--add X] [--remove Y] [--detect]``."""
    from deep_worker_cli.lsp.languages import (
        LanguageConfigError,
        apply_language_changes,
        detected_languages,
        language_facts,
        merge_languages,
        project_config_path,
        read_languages,
        validate_language_name,
    )

    project_root = _workspace_root(args)
    config = project_config_path(project_root)
    adds = list(dict.fromkeys(getattr(args, "add", None) or []))
    removes = list(dict.fromkeys(getattr(args, "remove", None) or []))
    detect = bool(getattr(args, "detect", False))

    # A5: adding and removing the same language at once is a conflict,
    # not a hidden priority: silence would hide a typo.
    conflict = sorted(set(adds) & set(removes))
    if conflict:
        print(f"error: {', '.join(conflict)} passed to both --add and --remove")
        return 1

    if not adds and not removes:
        return _cmd_languages_show(args, project_root, config, detect)

    try:
        facts = language_facts(project_root)
        known = facts.get("known", [])
        adds = [validate_language_name(name, known) for name in adds]
        removes = [validate_language_name(name, known) for name in removes]

        current = read_languages(config)
        if not config.exists():
            print(f"error: {config} does not exist; run 'deep-worker lsp enable' first")
            return 1

        # A5: fixed order — detect → add → remove.
        desired = list(current)
        if detect:
            desired = merge_languages(
                desired,
                detected_languages(
                    facts.get("composition", {}), facts.get("priority", {})
                ),
            )
        desired = merge_languages(desired, adds)
        desired = [name for name in desired if name not in set(removes)]

        if not desired:
            print(
                "error: refusing to leave the project with no languages: symbol "
                "tools would stop working. Use 'deep-worker lsp disable' instead."
            )
            return 1

        if apply_language_changes(config, desired=desired):
            print(f"languages updated in {config}: {', '.join(desired)}")
            print(_restart_hint(project_root))
        else:
            print(f"languages already up to date: {', '.join(desired)}")
        return 0
    except LanguageConfigError as e:
        print(f"error: {e}")
        return 1


def _cmd_languages_show(
    args: argparse.Namespace,
    project_root: Path,
    config: Path,
    detect: bool,
) -> int:
    """Without --add/--remove: show the state (A5 — strictly read-only).

    The only exception is the explicit ``--detect``, which the user
    requested precisely as a mutation.
    """
    from deep_worker_cli.lsp.languages import (
        LanguageConfigError,
        detected_languages,
        language_facts,
        read_languages,
    )

    if detect:
        return _reconcile_languages(project_root, quiet_failure=False)

    print(f"project root: {project_root}")
    if config.exists():
        current = read_languages(config)
        print(f"configured   : {', '.join(current) or '(none)'}")
    else:
        current = []
        print(f"configured   : (not configured — {config} does not exist)")

    try:
        facts = language_facts(project_root)
    except LanguageConfigError as e:
        print(f"detection unavailable: {e}")
        return 1

    composition = facts.get("composition", {})
    if not composition:
        print("detected     : (no supported source files found)")
        return 0
    print("detected (% of recognized source files):")
    for name, pct in sorted(composition.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {pct:6.2f}%")
    missing = [
        name
        for name in detected_languages(composition, facts.get("priority", {}))
        if name not in current
    ]
    if missing:
        print(
            f"above threshold but NOT configured: {', '.join(missing)}\n"
            f"    add them with: deep-worker lsp languages --detect"
        )
    return 0


def _cmd_enable(args: argparse.Namespace) -> int:
    path = _config_path(args)
    workspace_root = _workspace_root(args)
    try:
        changed = enable_lsp_at_path(
            path,
            workspace_root=workspace_root,
            force=bool(getattr(args, "force", False)),
            auto_approve=not bool(getattr(args, "no_auto_approve", False)),
        )
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 1

    if changed:
        print(f"LSP navigation enabled in {path}")
    else:
        print(
            f"LSP navigation already enabled in {path} (use --force to overwrite)"
        )
    # Print the resolved root explicitly: its silent substitution was
    # exactly the bug (the config drifted into the install directory).
    print(f"workspace root: {workspace_root}")

    if not bool(getattr(args, "no_gradle_init", False)):
        _install_gradle_init(force=bool(getattr(args, "force", False)))

    # A4: the language reconciliation runs ALWAYS, not only when
    # mcp.json changed. Otherwise auto-detection would never fire for
    # already-enabled projects — that is exactly how Python fell out of
    # the language set. The no-op above applies only to the MCP write.
    if not bool(getattr(args, "no_detect_languages", False)):
        _reconcile_languages(workspace_root, quiet_failure=True)

    _warn_if_serena_missing()
    print("Restart the agent for the change to take effect.")
    return 0


def _install_gradle_init(*, force: bool) -> None:
    """Install the Gradle init script for Kotlin/Android.

    Does not affect the return code: the MCP config is already written, and
    a failed install only degrades to the current (broken) cross-file
    navigation in AGP projects — failing ``lsp enable`` because of it is wrong.
    """
    target = default_init_d_dir() / GRADLE_INIT_SCRIPT_NAME
    try:
        installed = install_gradle_init_script(force=force)
    except OSError as e:
        print(f"warning: could not install Gradle init script at {target}: {e}")
        return

    if installed:
        print(f"gradle init script: installed at {target}")
    else:
        state = gradle_init_state()
        if state is InstallState.CURRENT:
            print(f"gradle init script: already up to date at {target}")
        elif state is InstallState.OURS_OUTDATED:
            # Not "already installed": an OLD or hand-edited copy is
            # there, and silence about it would hide broken navigation.
            print(
                f"gradle init script: {target} differs from the packaged "
                f"version — use --force to update"
            )
        else:
            print(
                f"warning: {target} exists but was not written by deep-worker — "
                f"left untouched (use --force to overwrite)"
            )
    # The file is GLOBAL: Gradle reads init scripts only from
    # $GRADLE_USER_HOME/init.d, so it affects all of the user's builds.
    print(
        "  (applies to every Gradle build of this user; "
        "remove with 'deep-worker lsp disable')"
    )


def _cmd_disable(args: argparse.Namespace) -> int:
    path = _config_path(args)
    try:
        removed = disable_lsp_at_path(path)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 1
    print(
        f"LSP navigation disabled in {path}"
        if removed
        else f"LSP navigation was not enabled in {path}"
    )

    target = default_init_d_dir() / GRADLE_INIT_SCRIPT_NAME
    try:
        # Only the file carrying our version marker is removed; a foreign
        # script with the same name is left in place.
        if uninstall_gradle_init_script():
            print(f"gradle init script: removed {target}")
    except OSError as e:
        print(f"warning: could not remove {target}: {e}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    path = _config_path(args)
    enabled = is_lsp_enabled_at_path(path)
    print(f"project: {resolve_workspace_root(invocation_cwd())}")
    print(f"config:  {path}")
    print(f"status:  {'enabled' if enabled else 'disabled'}")
    binary = shutil.which(SERENA_COMMAND)
    print(
        f"backend: {SERENA_SERVER_NAME} ({SERENA_COMMAND}) "
        + (f"found at {binary}" if binary else f"not found in PATH — {_INSTALL_HINT}")
    )
    gradle_target = default_init_d_dir() / GRADLE_INIT_SCRIPT_NAME
    print(
        "gradle init (Kotlin/Android): "
        + (
            f"installed at {gradle_target}"
            if is_gradle_init_installed()
            else f"not installed ({gradle_target})"
        )
    )
    return 0
