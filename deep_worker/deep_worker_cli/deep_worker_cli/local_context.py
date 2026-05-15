"""Middleware for injecting local context into system prompt."""

from __future__ import annotations

import json
import shutil

# S404: subprocess is required for git commands to detect project context
import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

from deep_worker_cli.config import get_glyphs


def _get_git_executable() -> str | None:
    """Get full path to git executable using shutil.which().

    Returns:
        Full path to git executable, or None if not found.
    """
    return shutil.which("git")


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

# Directories to ignore in file listings and tree views
IGNORE_PATTERNS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".coverage",
        ".eggs",
        "dist",
        "build",
    }
)


class LocalContextState(AgentState):
    """State for local context middleware."""

    local_context: NotRequired[str]
    """Formatted local context: git, cwd, files, tree."""


class LocalContextStateUpdate(TypedDict):
    """State update for local context middleware."""

    local_context: str
    """Formatted local context: git, cwd, files, tree."""


class LocalContextMiddleware(AgentMiddleware[LocalContextState, Any, Any]):
    """Middleware for injecting local context into system prompt.

    This middleware:
    1. Detects current git branch (if in a git repo)
    2. Checks if main/master branches exist locally
    3. Lists files in current directory (max 20)
    4. Shows directory tree structure (max 3 levels, 20 entries)
    5. Appends local context to system prompt
    """

    state_schema = LocalContextState

    @staticmethod
    def _get_git_info(cwd: Path) -> dict[str, str | list[str]]:
        """Gather git state information.

        Args:
            cwd: Directory to check git info in (NOT Path.cwd()!)

        Returns:
            Dict with 'branch' (current branch) and 'main_branches'
                (list of main/master if they exist). Returns empty dict if not in
                git repo.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        git_path = _get_git_executable()
        if not git_path:
            return {}

        try:
            # Get current branch
            # S603: git_path is validated via shutil.which(), args are hardcoded
            result = subprocess.run(  # noqa: S603
                [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=cwd,
                check=False,
            )
            if result.returncode != 0:
                return {}

            current_branch = result.stdout.strip()

            # Get local branches to check for main/master
            main_branches = []
            # S603: git_path is validated via shutil.which(), args are hardcoded
            result = subprocess.run(  # noqa: S603
                [git_path, "branch"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=cwd,
                check=False,
            )
            if result.returncode == 0:
                branches = set()
                for line in result.stdout.strip().split("\n"):
                    branch = line.strip().lstrip("*").strip()
                    if branch:
                        branches.add(branch)

                if "main" in branches:
                    main_branches.append("main")
                if "master" in branches:
                    main_branches.append("master")

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return {}
        else:
            return {"branch": current_branch, "main_branches": main_branches}

    @staticmethod
    def _get_file_list(cwd: Path, max_files: int = 20) -> list[str]:
        """Get list of files in specified directory (non-recursive).

        Args:
            cwd: Directory to list files from (NOT Path.cwd()!)
            max_files: Maximum number of files to show (default 20).

        Returns:
            List of file paths (sorted), truncated to max_files.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        files = []
        try:
            for item in sorted(cwd.iterdir()):
                # Skip hidden files (except .deepagents)
                if item.name.startswith(".") and item.name != ".deepagents":
                    continue

                # Skip ignored patterns
                if item.name in IGNORE_PATTERNS:
                    continue

                # Add files and dirs
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    files.append(f"{item.name}/")

                if len(files) >= max_files:
                    break

        except (OSError, PermissionError):
            return []

        return files

    @staticmethod
    def _get_directory_tree(cwd: Path, max_depth: int = 3, max_entries: int = 20) -> str:
        """Get directory tree structure.

        Args:
            cwd: Directory to build tree from (NOT Path.cwd()!)
            max_depth: Maximum depth to traverse (default 3).
            max_entries: Maximum total entries to show (default 20).

        Returns:
            Formatted tree string or empty if error.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        lines: list[str] = []
        entry_count = [0]  # Mutable for closure

        def _should_include(item: Path) -> bool:
            """Check if item should be included in tree.

            Returns:
                True if item should be included, False otherwise.
            """
            # Skip hidden files (except .deepagents)
            if item.name.startswith(".") and item.name != ".deepagents":
                return False
            # Skip ignored patterns
            return item.name not in IGNORE_PATTERNS

        def _build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            """Recursive tree builder."""
            if depth >= max_depth or entry_count[0] >= max_entries:
                return

            try:
                all_items = sorted(
                    path.iterdir(), key=lambda p: (not p.is_dir(), p.name)
                )
                # Pre-filter to get correct is_last determination
                items = [item for item in all_items if _should_include(item)]
            except (OSError, PermissionError):
                return

            glyphs = get_glyphs()
            for i, item in enumerate(items):
                if entry_count[0] >= max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    return

                is_last = i == len(items) - 1
                connector = glyphs.tree_last if is_last else glyphs.tree_branch

                display_name = f"{item.name}/" if item.is_dir() else item.name
                lines.append(f"{prefix}{connector}{display_name}")
                entry_count[0] += 1

                # Recurse into directories
                if item.is_dir() and depth + 1 < max_depth:
                    # Use 4 spaces when at last item, otherwise tree_vertical
                    extension = "    " if is_last else glyphs.tree_vertical
                    _build_tree(item, prefix + extension, depth + 1)

        try:
            lines.append(f"{cwd.name}/")
            _build_tree(cwd)
        except (OSError, PermissionError):
            return ""

        return "\n".join(lines)

    @staticmethod
    def _detect_package_manager(cwd: Path) -> str | None:
        """Detect Python package manager in use.

        Checks for lock files and config files to determine the package manager.

        Uses priority order: `uv > poetry > pipenv > pip`. First match wins if multiple
        indicators are present.

        Args:
            cwd: Directory to check package manager in (NOT Path.cwd()!)

        Returns:
            Package manager name (uv, poetry, pipenv, pip) or `None` if not detected.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        # Check for uv (uv.lock or pyproject.toml with [tool.uv])
        if (cwd / "uv.lock").exists():
            return "uv"

        # Check for poetry (poetry.lock or pyproject.toml with [tool.poetry])
        if (cwd / "poetry.lock").exists():
            return "poetry"

        # Check for pipenv
        if (cwd / "Pipfile.lock").exists() or (cwd / "Pipfile").exists():
            return "pipenv"

        # Check pyproject.toml for tool sections
        pyproject = cwd / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
            except (OSError, PermissionError, UnicodeDecodeError):
                pass
            else:
                if "[tool.uv]" in content:
                    return "uv"
                if "[tool.poetry]" in content:
                    return "poetry"
                # Has pyproject.toml but no specific tool - likely pip/setuptools
                return "pip"

        # Check for requirements.txt
        if (cwd / "requirements.txt").exists():
            return "pip"

        return None

    @staticmethod
    def _detect_node_package_manager(cwd: Path) -> str | None:
        """Detect Node.js package manager in use.

        Uses priority order: `bun > pnpm > yarn > npm`.

        First match wins if multiple lock files are present.

        Args:
            cwd: Directory to check package manager in (NOT Path.cwd()!)

        Returns:
            Package manager name (bun, pnpm, yarn, npm) or `None` if not detected.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        if (cwd / "bun.lockb").exists() or (cwd / "bun.lock").exists():
            return "bun"
        if (cwd / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (cwd / "yarn.lock").exists():
            return "yarn"
        if (cwd / "package-lock.json").exists() or (cwd / "package.json").exists():
            return "npm"

        return None

    @staticmethod
    def _get_makefile_preview(cwd: Path, max_lines: int = 20) -> str | None:
        """Get first N lines of `Makefile` if present.

        Args:
            cwd: Directory to check for Makefile in (NOT Path.cwd()!)
            max_lines: Maximum lines to show.

        Returns:
            `Makefile` preview or `None` if not found.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        makefile = cwd / "Makefile"

        if not makefile.exists():
            return None

        try:
            content = makefile.read_text()
        except (OSError, PermissionError, UnicodeDecodeError):
            return None
        else:
            lines = content.split("\n")[:max_lines]
            preview = "\n".join(lines)
            if len(content.split("\n")) > max_lines:
                preview += "\n... (truncated)"
            return preview

    @staticmethod
    def _detect_project_info(cwd: Path) -> dict[str, str | bool | None]:
        """Detect project type, language, and structure.

        Args:
            cwd: Directory to check project info in (NOT Path.cwd()!)

        Returns:
            Dict with `language`, `is_monorepo`, `project_root`, `has_venv`,
                `has_node_modules`.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        info: dict[str, str | bool | None] = {
            "language": None,
            "is_monorepo": False,
            "project_root": None,
            "has_venv": False,
            "has_node_modules": False,
        }

        # Check for virtual environments
        info["has_venv"] = (cwd / ".venv").exists() or (cwd / "venv").exists()
        info["has_node_modules"] = (cwd / "node_modules").exists()

        # Detect primary language
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            info["language"] = "python"
        elif (cwd / "package.json").exists():
            info["language"] = "javascript/typescript"
        elif (cwd / "Cargo.toml").exists():
            info["language"] = "rust"
        elif (cwd / "go.mod").exists():
            info["language"] = "go"
        elif (cwd / "pom.xml").exists() or (cwd / "build.gradle").exists():
            info["language"] = "java"

        # Detect monorepo patterns
        # Check for common monorepo indicators
        monorepo_indicators = [
            (cwd / "lerna.json").exists(),
            (cwd / "pnpm-workspace.yaml").exists(),
            (cwd / "packages").is_dir(),
            (cwd / "libs").is_dir() and (cwd / "apps").is_dir(),
            (cwd / "workspaces").is_dir(),
        ]
        info["is_monorepo"] = any(monorepo_indicators)

        # Try to find project root (look for .git or pyproject.toml up the tree)
        git_path = _get_git_executable()
        if git_path:
            try:
                # S603: git_path is validated via shutil.which(), args are hardcoded
                result = subprocess.run(  # noqa: S603
                    [git_path, "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    cwd=cwd,
                    check=False,
                )
                if result.returncode == 0:
                    info["project_root"] = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        return info

    @staticmethod
    def _detect_test_command(cwd: Path) -> str | None:
        """Detect how to run tests based on project structure.

        Args:
            cwd: Directory to check test command in (NOT Path.cwd()!)

        Returns:
            Suggested test command or `None` if not detected.
        """
        # REGRESSION PREVENTION: cwd parameter must be passed from working_dir state!
        # Do NOT use Path.cwd() here - it returns process cwd, not user's working dir

        # Check Makefile for test target
        makefile = cwd / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text()
                if "test:" in content or "tests:" in content:
                    return "make test"
            except (OSError, PermissionError, UnicodeDecodeError):
                pass

        # Python projects
        if (cwd / "pyproject.toml").exists():
            pyproject = cwd / "pyproject.toml"
            try:
                content = pyproject.read_text()
                if "[tool.pytest" in content or (cwd / "pytest.ini").exists():
                    return "pytest"
            except (OSError, PermissionError, UnicodeDecodeError):
                pass
            if (cwd / "tests").is_dir() or (cwd / "test").is_dir():
                return "pytest"

        # Node projects
        if (cwd / "package.json").exists():
            try:
                pkg = json.loads((cwd / "package.json").read_text())
                if "scripts" in pkg and "test" in pkg["scripts"]:
                    return "npm test"
            except (OSError, PermissionError, UnicodeDecodeError, json.JSONDecodeError):
                pass

        return None

    # `state: LocalContextState` is a covariant refinement of the base
    # `AgentState[Any]`. The `AgentMiddleware[LocalContextState, ...]`
    # class-level parameterization above makes this override legal under
    # ty. No suppression comments are needed here.
    def before_agent(
        self,
        state: LocalContextState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Load local context before agent execution.

        Runs once at session start to preserve prompt caching.

        Note:
            Return type is `dict[str, Any]` rather than
            `LocalContextStateUpdate` to match the base class signature. At
            runtime, TypedDict is just dict, so this is functionally equivalent
            while satisfying the type checker.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with local_context populated, or None if already set.
        """
        # ============================================================================
        # CRITICAL REGRESSION PREVENTION: USE working_dir FROM STATE, NOT Path.cwd()
        # ============================================================================
        # ALWAYS use working_dir from runtime.state when available!
        #
        # Path.cwd() returns the PROCESS's current directory, which is often "/"
        # for daemon processes, NOT the user's intended working directory.
        #
        # When an agent is started with `deep-worker myagent --dir /tmp`, the
        # working_dir="/tmp" is stored in state by ProjectMemoryMiddleware.
        # LocalContext MUST use this value, NOT Path.cwd()!
        #
        # BUG HISTORY:
        # - Before fix: Used Path.cwd() everywhere → showed "/" for agents started in /tmp
        # - After fix: Use working_dir from state → correctly shows /tmp
        #
        # CRITICAL: We must check working_dir EVERY TIME because ProjectMemoryMiddleware
        # may set it AFTER our first before_agent() call. Do NOT cache if working_dir
        # was not available on first call!
        #
        # TEST: test_local_context_uses_working_dir_from_state verifies this behavior
        # ============================================================================
        working_dir_str = state.get("working_dir")
        if not working_dir_str:
            # working_dir not yet set - skip context generation this time
            # It will be set by ProjectMemoryMiddleware on next call
            return None

        cwd = Path(working_dir_str)

        # Only compute context on first interaction to preserve prompt caching
        # BUT we must have working_dir first!
        if state.get("local_context"):
            return None

        sections = ["## Local Context", ""]

        # Current directory
        sections.extend([f"**Current Directory**: `{cwd}`", ""])

        # Project info (language, monorepo, root, environments)
        # REGRESSION PREVENTION: Pass cwd from working_dir state!
        project_info = self._detect_project_info(cwd)
        project_lines = []
        if project_info.get("language"):
            project_lines.append(f"Language: {project_info['language']}")
        if project_info.get("project_root") and str(
            project_info["project_root"]
        ) != str(cwd):
            project_lines.append(f"Project root: `{project_info['project_root']}`")
        if project_info.get("is_monorepo"):
            project_lines.append("Monorepo: yes")
        env_indicators = []
        if project_info.get("has_venv"):
            env_indicators.append(".venv")
        if project_info.get("has_node_modules"):
            env_indicators.append("node_modules")
        if env_indicators:
            project_lines.append(f"Environments: {', '.join(env_indicators)}")
        if project_lines:
            sections.append("**Project**:")
            sections.extend(f"- {line}" for line in project_lines)
            sections.append("")

        # Package managers
        # REGRESSION PREVENTION: Pass cwd from working_dir state!
        pkg_managers = []
        python_pkg = self._detect_package_manager(cwd)
        if python_pkg:
            pkg_managers.append(f"Python: {python_pkg}")
        node_pkg = self._detect_node_package_manager(cwd)
        if node_pkg:
            pkg_managers.append(f"Node: {node_pkg}")
        if pkg_managers:
            sections.extend([f"**Package Manager**: {', '.join(pkg_managers)}", ""])

        # Git info
        # REGRESSION PREVENTION: Pass cwd from working_dir state!
        git_info = self._get_git_info(cwd)
        if git_info:
            git_text = f"**Git**: Current branch `{git_info['branch']}`"
            if git_info.get("main_branches"):
                main_branches = ", ".join(f"`{b}`" for b in git_info["main_branches"])
                git_text += f", main branch available: {main_branches}"
            sections.extend([git_text, ""])

        # Test command
        # REGRESSION PREVENTION: Pass cwd from working_dir state!
        test_cmd = self._detect_test_command(cwd)
        if test_cmd:
            sections.extend([f"**Run Tests**: `{test_cmd}`", ""])

        # File list
        # REGRESSION PREVENTION: Pass cwd from working_dir state, NOT Path.cwd()!
        files = self._get_file_list(cwd)
        if files:
            try:
                total_items = len(list(cwd.iterdir()))
            except (OSError, PermissionError):
                total_items = len(files)

            sections.append(f"**Files** ({len(files)} shown):")
            sections.extend(f"- {file}" for file in files)
            if len(files) < total_items:
                remaining = total_items - len(files)
                sections.append(f"... ({remaining} more files)")
            sections.append("")

        # Directory tree
        # REGRESSION PREVENTION: Pass cwd from working_dir state, NOT Path.cwd()!
        tree = self._get_directory_tree(cwd)
        if tree:
            sections.extend(["**Tree** (3 levels):", "```text", tree, "```", ""])

        # Makefile preview
        # REGRESSION PREVENTION: Pass cwd from working_dir state!
        makefile_preview = self._get_makefile_preview(cwd)
        if makefile_preview:
            sections.extend(
                [
                    "**Makefile** (first 20 lines):",
                    "```makefile",
                    makefile_preview,
                    "```",
                ]
            )

        local_context = "\n".join(sections)
        return {"local_context": local_context}

    @staticmethod
    def _get_modified_request(request: ModelRequest) -> ModelRequest | None:
        """Get modified request with local context injected, or None if no context.

        Args:
            request: The original model request.

        Returns:
            Modified request with local context appended, or None if no local context.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")

        if not local_context:
            return None

        # Append local context to system prompt
        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + local_context

        return request.override(system_prompt=new_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)


__all__ = ["LocalContextMiddleware"]
