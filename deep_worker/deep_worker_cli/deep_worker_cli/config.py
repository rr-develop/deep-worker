"""Configuration, constants, and model creation for the CLI."""

import json
import logging
import os
import re
import shlex
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import dotenv
from rich.console import Console

from deep_worker_cli._version import __version__
from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.model_predicate import (
    model_can_belong_to_provider as _leaf_model_can_belong_to_provider,
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# CRITICAL: Override LANGSMITH_PROJECT to route agent traces to separate project
# LangSmith reads LANGSMITH_PROJECT at invocation time, so we override it here
# and preserve the user's original value for shell commands
_deepagents_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
_original_langsmith_project = os.environ.get("LANGSMITH_PROJECT")
if _deepagents_project:
    # Override LANGSMITH_PROJECT for agent traces
    os.environ["LANGSMITH_PROJECT"] = _deepagents_project

# E402: Now safe to import LangChain modules
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

# Color scheme
COLORS = {
    "primary": "#10b981",
    "primary_dev": "#f97316",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}


# Charset mode configuration
class CharsetMode(StrEnum):
    """Character set mode for TUI display."""

    UNICODE = "unicode"
    ASCII = "ascii"
    AUTO = "auto"


class WebSearchPolicy(StrEnum):
    """Tri-state universal web-search policy (DESIGN_PROVIDERS_FROM_WHEEL §3.3).

    Controls whether the Claude-Code-SDK-backed ``web_search`` delegation
    tool is resolved and bound into the agent toolset:

    - ``AUTO`` (default): bind the tool when the SDK fork is available;
      otherwise log once and skip the tool — the agent still starts.
    - ``ENABLED`` (explicit ``--enable-web-search``): bind the tool, or fail
      loudly with ``CapabilityError`` (with remedy) when the SDK is missing.
    - ``DISABLED`` (explicit ``--disable-web-search``): NEVER resolve or
      import the claude provider / SDK; the tool is never bound.
    """

    AUTO = "auto"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True)
class Glyphs:
    """Character glyphs for TUI display."""

    tool_prefix: str  # ⏺ vs (*)
    ellipsis: str  # … vs ...
    checkmark: str  # ✓ vs [OK]
    error: str  # ✗ vs [X]
    circle_empty: str  # ○ vs [ ]
    circle_filled: str  # ● vs [*]
    output_prefix: str  # ⎿ vs L
    spinner_frames: tuple[str, ...]  # Braille vs ASCII spinner
    pause: str  # ⏸ vs ||
    newline: str  # ⏎ vs \\n
    warning: str  # ⚠ vs [!]
    arrow_up: str  # up arrow vs ^
    arrow_down: str  # down arrow vs v
    bullet: str  # bullet vs -
    cursor: str  # cursor vs >

    # Box-drawing characters
    box_vertical: str  # │ vs |
    box_horizontal: str  # ─ vs -
    box_double_horizontal: str  # ═ vs =

    # Diff-specific
    gutter_bar: str  # ▌ vs |

    # Tree connectors (full prefixes for tree display)
    tree_branch: str  # "├── " vs "+-- "
    tree_last: str  # "└── " vs "`-- "
    tree_vertical: str  # "│   " vs "|   "


UNICODE_GLYPHS = Glyphs(
    tool_prefix="⏺",
    ellipsis="…",
    checkmark="✓",
    error="✗",
    circle_empty="○",
    circle_filled="●",
    output_prefix="⎿",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    pause="⏸",
    newline="⏎",
    warning="⚠",
    arrow_up="↑",
    arrow_down="↓",
    bullet="•",
    cursor="›",  # noqa: RUF001
    # Box-drawing characters
    box_vertical="│",
    box_horizontal="─",
    box_double_horizontal="═",
    gutter_bar="▌",
    tree_branch="├── ",
    tree_last="└── ",
    tree_vertical="│   ",
)

ASCII_GLYPHS = Glyphs(
    tool_prefix="(*)",
    ellipsis="...",
    checkmark="[OK]",
    error="[X]",
    circle_empty="[ ]",
    circle_filled="[*]",
    output_prefix="L",
    spinner_frames=("(-)", "(\\)", "(|)", "(/)"),
    pause="||",
    newline="\\n",
    warning="[!]",
    arrow_up="^",
    arrow_down="v",
    bullet="-",
    cursor=">",
    # Box-drawing characters
    box_vertical="|",
    box_horizontal="-",
    box_double_horizontal="=",
    gutter_bar="|",
    tree_branch="+-- ",
    tree_last="`-- ",
    tree_vertical="|   ",
)

# Module-level cache for detected glyphs
_glyphs_cache: Glyphs | None = None

# Module-level cache for editable install detection
_editable_cache: bool | None = None


def _is_editable_install() -> bool:
    """Check if deepagents-cli is installed in editable mode.

    Uses PEP 610 direct_url.json metadata to detect editable installs.

    Returns:
        True if installed in editable mode, False otherwise.
    """
    global _editable_cache  # noqa: PLW0603
    if _editable_cache is not None:
        return _editable_cache

    try:
        dist = distribution("deepagents-cli")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            data = json.loads(direct_url)
            _editable_cache = data.get("dir_info", {}).get("editable", False)
        else:
            _editable_cache = False
    except (PackageNotFoundError, FileNotFoundError, json.JSONDecodeError, TypeError):
        _editable_cache = False

    return _editable_cache


def _detect_charset_mode() -> CharsetMode:
    """Auto-detect terminal charset capabilities.

    Returns:
        The detected CharsetMode based on environment and terminal encoding.
    """
    env_mode = os.environ.get("UI_CHARSET_MODE", "auto").lower()
    if env_mode == "unicode":
        return CharsetMode.UNICODE
    if env_mode == "ascii":
        return CharsetMode.ASCII

    # Auto: check stdout encoding and LANG
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return CharsetMode.UNICODE
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    if "utf" in lang.lower():
        return CharsetMode.UNICODE
    return CharsetMode.ASCII


def get_glyphs() -> Glyphs:
    """Get the glyph set for the current charset mode.

    Returns:
        The appropriate Glyphs instance based on charset mode detection.
    """
    global _glyphs_cache  # noqa: PLW0603
    if _glyphs_cache is not None:
        return _glyphs_cache

    mode = _detect_charset_mode()
    _glyphs_cache = ASCII_GLYPHS if mode == CharsetMode.ASCII else UNICODE_GLYPHS
    return _glyphs_cache


def reset_glyphs_cache() -> None:
    """Reset the glyphs cache (for testing)."""
    global _glyphs_cache  # noqa: PLW0603
    _glyphs_cache = None


# Text art banners (Unicode and ASCII variants)

_UNICODE_BANNER = f"""
██████╗  ███████╗ ███████╗ ██████╗    ▄▓▓▄
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗  ▓•███▙
██║  ██║ █████╗   █████╗   ██████╔╝  ░▀▀████▙▖
██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝      █▓████▙▖
██████╔╝ ███████╗ ███████╗ ██║          ▝█▓█████▙
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝           ░▜█▓████▙
                                          ░█▀█▛▀▀▜▙▄
                                        ░▀░▀▒▛░░  ▝▀▘

 █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                                  v{__version__}
"""
_ASCII_BANNER = f"""
 ____  ____  ____  ____
|  _ \\| ___|| ___||  _ \\
| | | | |_  | |_  | |_) |
| |_| |  _| |  _| |  __/
|____/|____||____||_|

    _    ____  ____  _   _  _____  ____
   / \\  / ___|| ___|| \\ | ||_   _|/ ___|
  / _ \\| |  _ | |_  |  \\| |  | |  \\___ \\
 / ___ \\ |_| ||  _| | |\\  |  | |   ___) |
/_/   \\_\\____||____||_| \\_|  |_|  |____/
                                  v{__version__}
"""


def get_banner() -> str:
    """Get the appropriate banner for the current charset mode.

    Returns:
        The text art banner string (Unicode or ASCII based on charset mode).
        Includes "(local)" suffix when installed in editable mode.
    """
    if _detect_charset_mode() == CharsetMode.ASCII:
        banner = _ASCII_BANNER
    else:
        banner = _UNICODE_BANNER

    if _is_editable_install():
        banner = banner.replace(f"v{__version__}", f"v{__version__} (local)")

    return banner


# Interactive commands
COMMANDS = {
    "clear": "Clear screen and reset conversation",
    "help": "Show help information",
    "remember": "Review conversation and update memory/skills",
    "tokens": "Show token usage for current thread",
    "quit": "Exit the CLI",
    "exit": "Exit the CLI",
}


# Maximum argument length for display
MAX_ARG_LENGTH = 150

# ============================================================================
# CRITICAL: NO RECURSION LIMIT ON AGENTS!
# ============================================================================
# recursion_limit is STRICTLY FORBIDDEN. Agents must be able to perform
# unlimited tool calls. Any limit causes GraphRecursionError crashes on
# long-running tasks (e.g. mass file edits across hundreds of files).
#
# DO NOT add recursion_limit back. DO NOT set it to any value.
# If LangGraph requires a value, it defaults to a safe internal default.
# ============================================================================
config: RunnableConfig = {}

# Rich console instance
console = Console(highlight=False)


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def _find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific AGENTS.md file(s).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/AGENTS.md
    2. project_root/AGENTS.md

    Both files will be loaded and combined if both exist.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of paths to project AGENTS.md files (may contain 0, 1, or 2 paths).
    """
    paths = []

    # Check .deepagents/AGENTS.md (preferred)
    deepagents_md = project_root / ".deepagents" / "AGENTS.md"
    if deepagents_md.exists():
        paths.append(deepagents_md)

    # Check root AGENTS.md (fallback, but also include if both exist)
    root_md = project_root / "AGENTS.md"
    if root_md.exists():
        paths.append(root_md)

    return paths


def parse_shell_allow_list(allow_list_str: str | None) -> list[str] | None:
    """Parse shell allow-list from string.

    Args:
        allow_list_str: Comma-separated list of commands, or "recommended" for
            safe defaults.

            Can also include "recommended" in the list to merge with custom commands.

    Returns:
        List of allowed commands, or None if no allow-list configured.
    """
    if not allow_list_str:
        return None

    # Special value "recommended" uses our curated safe list
    if allow_list_str.strip().lower() == "recommended":
        return list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    # Split by comma and strip whitespace
    commands = [cmd.strip() for cmd in allow_list_str.split(",") if cmd.strip()]

    # If "recommended" is in the list, merge with recommended commands
    result = []
    for cmd in commands:
        if cmd.lower() == "recommended":
            result.extend(RECOMMENDED_SAFE_SHELL_COMMANDS)
        else:
            result.append(cmd)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in result:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique


# ==============================================================================
# Provider Context Limits and Summarization Configuration
# ==============================================================================
# Context window sizes (in tokens) for different AI providers
# Used as fallback when model profile doesn't provide max_input_tokens
# Sources: Official documentation from Anthropic, DeepSeek, OpenAI, Google

PROVIDER_CONTEXT_LIMITS: dict[str, int] = {
    # Claude Code short names (used by claude-code provider)
    "opus": 1_000_000,
    "sonnet": 200_000,
    "haiku": 200_000,
    # Anthropic Claude models (full names)
    "claude-opus-4.6": 1_000_000,  # Requires anthropic-beta header
    "claude-sonnet-4.6": 1_000_000,  # Requires anthropic-beta header
    "claude-sonnet-4.5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4.5": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # DeepSeek models
    "deepseek-chat": 128_000,  # V3.2 API (legacy alias)
    "deepseek-reasoner": 128_000,  # V3.2 API (legacy alias)
    "deepseek-v3": 128_000,
    "deepseek-v3.1": 128_000,
    "deepseek-v3.2": 128_000,
    "deepseek-v4-flash": 1_000_000,  # 1M context
    "deepseek-v4-pro": 1_000_000,  # 1M context
    "deepseek-coder-v2": 128_000,
    "deepseek-r1": 64_000,  # R1 has smaller context
    # OpenAI models
    "gpt-5": 400_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5-turbo": 16_385,
    # Google Gemini models
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    # Mistral models
    "mistral-large": 128_000,
    # Cohere models
    "command-r-plus": 128_000,
    # Fallback for unknown models
    "default": 100_000,
}


@dataclass(frozen=True)
class SummarizationConfig:
    """Configuration for SummarizationMiddleware based on model context limit.

    This dataclass holds all settings needed to configure automatic
    conversation summarization, dynamically chosen based on the model's
    context window size.

    Attributes:
        trigger_fraction: Fraction of context (0.0-1.0) at which to trigger
            summarization. E.g., 0.80 means trigger at 80% full.
        keep_fraction: Fraction of context (0.0-1.0) to keep after
            summarization. E.g., 0.15 means keep last 15%.
        truncate_after_messages: Truncate tool call arguments in messages
            older than this count.
        max_arg_length: Maximum length for truncated tool call arguments.
        context_limit: Model's maximum input tokens (for reference).
        trigger_tokens: Absolute token count at which summarization triggers.
        keep_tokens: Absolute token count to keep after summarization.
    """

    trigger_fraction: float
    keep_fraction: float
    truncate_after_messages: int
    max_arg_length: int
    # Metadata for logging/debugging
    context_limit: int
    trigger_tokens: int
    keep_tokens: int


def get_model_context_limit(
    model_name: str | None,
    model_profile: dict[str, Any] | None,
) -> int:
    """Get context limit for a model.

    Priority order:
    1. model_profile["max_input_tokens"] (from LangChain model introspection)
    2. PROVIDER_CONTEXT_LIMITS[model_name] (known model mapping)
    3. PROVIDER_CONTEXT_LIMITS["default"] (fallback: 100K tokens)

    Args:
        model_name: Model name (e.g., "claude-sonnet-4.5" or
            "anthropic:claude-sonnet-4.5"). May include provider prefix.
        model_profile: Model profile dict from LangChain, may contain
            "max_input_tokens" key.

    Returns:
        Context limit in tokens.

    Examples:
        >>> get_model_context_limit("claude-sonnet-4.5", None)
        200000
        >>> get_model_context_limit("deepseek-chat", None)
        128000
        >>> get_model_context_limit(None, {"max_input_tokens": 150000})
        150000
    """
    # Priority 1: Model profile from LangChain
    if model_profile and isinstance(model_profile.get("max_input_tokens"), int):
        return model_profile["max_input_tokens"]

    # Priority 2: Known model mapping
    if model_name:
        # Try exact match first
        if model_name in PROVIDER_CONTEXT_LIMITS:
            return PROVIDER_CONTEXT_LIMITS[model_name]

        # Try partial match (e.g., "anthropic:claude-sonnet-4.5" → "claude-sonnet-4.5")
        # or "openrouter/anthropic/claude-sonnet-4.5" → "claude-sonnet-4.5"
        for known_model, limit in PROVIDER_CONTEXT_LIMITS.items():
            if known_model != "default" and known_model in model_name:
                return limit

    # Priority 3: Fallback
    return PROVIDER_CONTEXT_LIMITS["default"]


def get_summarization_config(
    model_context_limit: int | None,
    model_name: str | None = None,
    override_trigger: float | None = None,
    override_keep: float | None = None,
) -> SummarizationConfig:
    """Get summarization configuration based on model context limit.

    Automatically selects optimal trigger/keep fractions and argument
    truncation settings based on the model's context window size.

    Configuration ranges:
    - 1M+ tokens: trigger=0.85, keep=0.10 (e.g., Claude Opus 4.6)
    - 400K tokens: trigger=0.80, keep=0.12 (e.g., GPT-5)
    - 200K tokens: trigger=0.80, keep=0.15 (e.g., Claude Sonnet 4.5)
    - 128K tokens: trigger=0.75, keep=0.20 (e.g., DeepSeek V3)
    - 64K tokens: trigger=0.70, keep=0.25 (e.g., DeepSeek R1)
    - 16K tokens: trigger=0.60, keep=0.30 (e.g., GPT-3.5)
    - <16K tokens: trigger=0.50, keep=0.35 (very small models)

    Args:
        model_context_limit: Max input tokens from model profile. If None,
            defaults to 100K tokens.
        model_name: Model name for logging (not used for logic currently).
        override_trigger: Override trigger fraction from CLI/settings.
            Takes precedence over automatic selection.
        override_keep: Override keep fraction from CLI/settings.
            Takes precedence over automatic selection.

    Returns:
        SummarizationConfig with recommended settings.

    Examples:
        >>> config = get_summarization_config(200_000)
        >>> config.trigger_fraction
        0.80
        >>> config.trigger_tokens
        160000
        >>> config.keep_tokens
        30000
    """
    # Determine context limit (fallback to 100K if None)
    context_limit = model_context_limit if model_context_limit is not None else 100_000

    # Select trigger/keep fractions based on context size
    if context_limit >= 1_000_000:
        # Very large context (1M+): Claude Opus 4.6, Gemini 1.5 Pro
        trigger_fraction = 0.85
        keep_fraction = 0.10
    elif context_limit >= 400_000:
        # Large context (400K): GPT-5
        trigger_fraction = 0.80
        keep_fraction = 0.12
    elif context_limit >= 200_000:
        # Medium-large context (200K): Claude Sonnet 4.5, Gemini 2.0
        trigger_fraction = 0.80
        keep_fraction = 0.15
    elif context_limit >= 128_000:
        # Medium context (128K): DeepSeek V3, GPT-4, Mistral
        trigger_fraction = 0.75
        keep_fraction = 0.20
    elif context_limit >= 64_000:
        # Small-medium context (64K): DeepSeek R1
        trigger_fraction = 0.70
        keep_fraction = 0.25
    elif context_limit >= 16_000:
        # Small context (16K): GPT-3.5 Turbo
        trigger_fraction = 0.60
        keep_fraction = 0.30
    else:
        # Very small context (<16K): old models
        trigger_fraction = 0.50
        keep_fraction = 0.35

    # Apply overrides from CLI/settings if provided
    if override_trigger is not None:
        trigger_fraction = override_trigger
    if override_keep is not None:
        keep_fraction = override_keep

    # Determine argument truncation settings based on context size
    if context_limit >= 200_000:
        # Large context: more aggressive truncation (more room)
        truncate_after_messages = 20
        max_arg_length = 500
    elif context_limit >= 64_000:
        # Medium context: moderate truncation
        truncate_after_messages = 15
        max_arg_length = 300
    else:
        # Small context: early truncation to save space
        truncate_after_messages = 10
        max_arg_length = 200

    # Calculate absolute token counts
    trigger_tokens = int(context_limit * trigger_fraction)
    keep_tokens = int(context_limit * keep_fraction)

    return SummarizationConfig(
        trigger_fraction=trigger_fraction,
        keep_fraction=keep_fraction,
        truncate_after_messages=truncate_after_messages,
        max_arg_length=max_arg_length,
        context_limit=context_limit,
        trigger_tokens=trigger_tokens,
        keep_tokens=keep_tokens,
    )


@dataclass
class Settings:
    """Global settings and environment detection for deepagents-cli.

    This class is initialized once at startup and provides access to:
    - Available models and API keys
    - Current project information
    - Tool availability (e.g., Tavily)
    - File system paths
    - Summarization configuration

    Attributes:
        openai_api_key: OpenAI API key if available.
        anthropic_api_key: Anthropic API key if available.
        google_api_key: Google API key if available.
        tavily_api_key: Tavily API key if available.
        google_cloud_project: Google Cloud project ID for VertexAI
            authentication.
        deepagents_langchain_project: LangSmith project name for deepagents
            agent tracing.
        user_langchain_project: Original LANGSMITH_PROJECT from environment
            (for user code).
        model_name: Currently active model name (set after model creation).
        model_provider: Provider identifier (e.g. openai, anthropic, google,
            vertexai).
        model_context_limit: Maximum input token count from the model profile.
        project_root: Current project root directory (if in a git project).
        shell_allow_list: List of shell commands that don't require approval.
        summarization_enabled: Enable automatic conversation summarization.
        summarization_trigger_fraction: When to trigger summarization (0.0-1.0).
        summarization_keep_fraction: How much history to keep after summarization.
        summarization_truncate_after: Truncate old arguments after N messages.
        summarization_max_arg_length: Max length for old tool call arguments.
        summarization_history_backend: History storage backend ("temp" or "persistent").
    """

    # API keys
    deepseek_api_key: str | None  # Deepseek (DEFAULT provider)
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    tavily_api_key: str | None

    # Google Cloud configuration (for VertexAI)
    google_cloud_project: str | None

    # LangSmith configuration
    deepagents_langchain_project: str | None  # For deepagents agent tracing
    user_langchain_project: str | None  # Original LANGSMITH_PROJECT for user code

    # Model configuration
    model_name: str | None = None  # Currently active model name
    model_provider: str | None = None  # Provider (openai, anthropic, google)
    model_context_limit: int | None = None  # Max input tokens from model profile

    # Project information
    project_root: Path | None = None

    # Shell command allow-list for auto-approval
    shell_allow_list: list[str] | None = None

    # Summarization configuration
    summarization_enabled: bool = True
    summarization_trigger_fraction: float | None = None  # Auto-detect from model
    summarization_keep_fraction: float | None = None  # Auto-detect from model
    summarization_truncate_after: int | None = None  # Auto-detect from model
    summarization_max_arg_length: int | None = None  # Auto-detect from model
    summarization_history_backend: str = "temp"  # "temp" or "persistent"

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> "Settings":
        """Create settings by detecting the current environment.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            Settings instance with detected configuration
        """
        # Detect API keys (Deepseek is DEFAULT provider)
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        google_cloud_project = os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Detect LangSmith configuration
        # DEEPAGENTS_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # Note: LANGSMITH_PROJECT was already overridden at module import time (above)
        # so we use the saved original value, not the current os.environ value
        deepagents_langchain_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
        user_langchain_project = _original_langsmith_project  # Use saved original!

        # Detect project
        project_root = _find_project_root(start_path)

        # Parse shell command allow-list from environment
        # Format: comma-separated list of commands (e.g., "ls,cat,grep,pwd")
        # Special value "recommended" uses RECOMMENDED_SAFE_SHELL_COMMANDS
        shell_allow_list_str = os.environ.get("DEEPAGENTS_SHELL_ALLOW_LIST")
        shell_allow_list = parse_shell_allow_list(shell_allow_list_str)

        # Parse summarization configuration from environment
        summarization_enabled_str = os.environ.get("DEEPAGENTS_SUMMARIZATION_ENABLED")
        if summarization_enabled_str is not None:
            summarization_enabled = summarization_enabled_str.lower() == "true"
        else:
            summarization_enabled = True  # Default: enabled

        # Parse optional float/int values (None if not set or invalid)
        def parse_float(env_var: str) -> float | None:
            value = os.environ.get(env_var)
            if value is None:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        def parse_int(env_var: str) -> int | None:
            value = os.environ.get(env_var)
            if value is None:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        summarization_trigger = parse_float("DEEPAGENTS_SUMMARIZATION_TRIGGER")
        summarization_keep = parse_float("DEEPAGENTS_SUMMARIZATION_KEEP")
        summarization_truncate = parse_int("DEEPAGENTS_SUMMARIZATION_TRUNCATE_AFTER")
        summarization_max_arg = parse_int("DEEPAGENTS_SUMMARIZATION_MAX_ARG_LENGTH")
        summarization_backend = os.environ.get(
            "DEEPAGENTS_SUMMARIZATION_HISTORY_BACKEND", "temp"
        )

        return cls(
            deepseek_api_key=deepseek_key,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
            shell_allow_list=shell_allow_list,
            summarization_enabled=summarization_enabled,
            summarization_trigger_fraction=summarization_trigger,
            summarization_keep_fraction=summarization_keep,
            summarization_truncate_after=summarization_truncate,
            summarization_max_arg_length=summarization_max_arg,
            summarization_history_backend=summarization_backend,
        )

    @property
    def has_deepseek(self) -> bool:
        """Check if Deepseek API key is configured (DEFAULT provider)."""
        return self.deepseek_api_key is not None

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None

    @property
    def has_vertex_ai(self) -> bool:
        """Check if VertexAI is available (Google Cloud project set, no API key).

        VertexAI uses Application Default Credentials (ADC) for authentication,
        so if GOOGLE_CLOUD_PROJECT is set and GOOGLE_API_KEY is not, we assume
        VertexAI.
        """
        return self.google_cloud_project is not None and self.google_api_key is None

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None

    @property
    def has_deepagents_langchain_project(self) -> bool:
        """Check if deepagents LangChain project name is configured."""
        return self.deepagents_langchain_project is not None

    @property
    def has_project(self) -> bool:
        """Check if currently in a git project."""
        return self.project_root is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """Get the base user-level .deepagents directory.

        Returns:
            Path to ~/.deepagents
        """
        return Path.home() / ".deepagents"

    @staticmethod
    def get_user_agent_md_path(agent_name: str) -> Path:
        """Get user-level AGENTS.md path for a specific agent.

        Returns path regardless of whether the file exists.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to the agent data AGENTS.md file. For storage v3 this is
            the AGENTS.md in the per-agent data directory; legacy v2 agents
            keep using ~/.deepagents/{agent_name}/AGENTS.md.
        """
        from deep_worker_cli.storage_paths import deepagents_data_dir
        return deepagents_data_dir(agent_name, create=False) / "AGENTS.md"

    def get_project_agent_md_path(self) -> Path | None:
        """Get project-level AGENTS.md path.

        Returns path regardless of whether the file exists.

        Returns:
            Path to {project_root}/.deepagents/AGENTS.md, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "AGENTS.md"

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """Validate to prevent invalid filesystem paths and security issues.

        Returns:
            True if the agent name is valid, False otherwise.
        """
        if not agent_name or not agent_name.strip():
            return False
        # Allow alphanumeric, hyphens, underscores, colons, and whitespace
        # Colon is allowed for TaskManager naming (e.g., "agent:tm")
        return bool(re.match(r"^[a-zA-Z0-9_\-:\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the global agent directory path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to the agent data directory. For storage v3 this is the
            per-agent data directory; legacy v2 agents keep using
            ~/.deepagents/{agent_name}.

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, colons, and spaces."
            )
            raise ValueError(msg)
        from deep_worker_cli.storage_paths import deepagents_data_dir
        return deepagents_data_dir(agent_name, create=False)

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """Ensure the global agent directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to the agent data directory. For storage v3 this is the
            per-agent data directory; legacy v2 agents keep using
            ~/.deepagents/{agent_name}.

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, colons, and spaces."
            )
            raise ValueError(msg)
        from deep_worker_cli.storage_paths import deepagents_data_dir
        agent_dir = deepagents_data_dir(agent_name, create=True)
        return agent_dir

    def ensure_project_deepagents_dir(self) -> Path | None:
        """Ensure the project .deepagents directory exists and return its path.

        Returns:
            Path to project .deepagents directory, or None if not in a project
        """
        if not self.project_root:
            return None

        project_deepagents_dir = self.project_root / ".deepagents"
        project_deepagents_dir.mkdir(parents=True, exist_ok=True)
        return project_deepagents_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """Get user-level skills directory path for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """Ensure user-level skills directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        from deep_worker_cli.storage_paths import deepagents_data_dir, ensure_private_dir

        agent_dir = deepagents_data_dir(agent_name, create=True)
        return ensure_private_dir(agent_dir / "skills")

    def get_project_skills_dir(self) -> Path | None:
        """Get project-level skills directory path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """Ensure project-level skills directory exists and return its path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        if skills_dir is None:
            return None
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_user_agents_dir(self, agent_name: str) -> Path:
        """Get user-level agents directory path for custom subagent definitions.

        Args:
            agent_name: Name of the CLI agent (e.g., "deepagents")

        Returns:
            Path to ~/.deepagents/{agent_name}/agents/
        """
        return self.get_agent_dir(agent_name) / "agents"

    def get_project_agents_dir(self) -> Path | None:
        """Get project-level agents directory path for custom subagent definitions.

        Returns:
            Path to {project_root}/.deepagents/agents/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agents"

    @property
    def user_agents_dir(self) -> Path:
        """Get the base user-level `.agents` directory (`~/.agents`).

        Returns:
            Path to `~/.agents`
        """
        return Path.home() / ".agents"

    def get_user_agent_skills_dir(self) -> Path:
        """Get user-level `~/.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `~/.agents/skills/`
        """
        return self.user_agents_dir / "skills"

    def get_project_agent_skills_dir(self) -> Path | None:
        """Get project-level `.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `{project_root}/.agents/skills/`, or `None` if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".agents" / "skills"

    @staticmethod
    def get_built_in_skills_dir() -> Path:
        """Get the directory containing built-in skills that ship with the CLI.

        Returns:
            Path to the `built_in_skills/` directory within the package.
        """
        return Path(__file__).parent / "built_in_skills"


# Global settings instance (initialized once)
settings = Settings.from_environment()


class SessionState:
    """Mutable session state shared across the app, adapter, and agent.

    Tracks runtime flags like auto-approve that can be toggled during a
    session via keybindings or the HITL approval menu's "Auto-approve all"
    option.

    The `auto_approve` flag controls whether tool calls (shell execution, file
    writes/edits, web search, URL fetch) require user confirmation before running.
    """

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        """Initialize session state with optional flags.

        Args:
            auto_approve: Whether to auto-approve tool calls without
                prompting.

                Can be toggled at runtime via Shift+Tab or the HITL
                approval menu.
            no_splash: Whether to skip displaying the splash screen on startup.
        """
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        self.thread_id = str(uuid.uuid4())

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return the new state.

        Called by the Shift+Tab keybinding in the Textual app.

        When auto-approve is on, all tool calls execute without prompting.

        Returns:
            The new `auto_approve` state after toggling.
        """
        self.auto_approve = not self.auto_approve
        return self.auto_approve


SHELL_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell", "execute"})
"""Tool names recognized as shell/command-execution tools.

Only `'execute'` is registered by the SDK and CLI backends in practice.
`'bash'` and `'shell'` are legacy names carried over and kept as
backwards-compatible aliases.
"""

DANGEROUS_SHELL_PATTERNS = (
    "$(",  # Command substitution
    "`",  # Backtick command substitution
    "$'",  # ANSI-C quoting (can encode dangerous chars via escape sequences)
    "\n",  # Newline (command injection)
    "\r",  # Carriage return (command injection)
    "\t",  # Tab (can be used for injection in some shells)
    "<(",  # Process substitution (input)
    ">(",  # Process substitution (output)
    "<<<",  # Here-string
    "<<",  # Here-doc (can embed commands)
    ">>",  # Append redirect
    ">",  # Output redirect
    "<",  # Input redirect
    "${",  # Variable expansion with braces (can run commands via ${var:-$(cmd)})
)

# Recommended safe shell commands for non-interactive mode.
# These commands are primarily read-only and do not modify the filesystem
# when used without shell redirection operators (which the dangerous-patterns
# check blocks).
#
# EXCLUDED (dangerous - listed on GTFOBins/LOOBins or can modify system):
# - All shells: bash, sh, zsh, fish, dash, ksh, csh, tcsh, etc.
# - Editors: vim, vi, nano, emacs, ed, etc. (can spawn shells)
# - Interpreters: python, perl, ruby, node, php, lua, awk, gawk, etc.
# - Package managers: pip, npm, gem, apt, yum, brew, etc.
# - Compilers: gcc, cc, make, cmake, etc.
# - Network tools: curl, wget, nc, ssh, scp, ftp, telnet, etc.
# - Archivers with shell escape: tar, zip, 7z, etc.
# - System modifiers: chmod, chown, chattr, mv, rm, cp, dd, etc.
# - Privilege tools: sudo, su, doas, pkexec, etc.
# - Process tools: env, xargs, find (with -exec), etc.
# - Git (can run hooks), docker, kubectl, etc.
#
# SAFE commands included below are primarily readers/formatters. File write and
# injection are prevented by the dangerous-patterns check that blocks redirects,
# command substitution, and other shell metacharacters.
RECOMMENDED_SAFE_SHELL_COMMANDS = (
    # Directory listing
    "ls",
    "dir",
    # File content viewing (read-only)
    "cat",
    "head",
    "tail",
    # Text searching (read-only)
    "grep",
    "wc",
    "strings",
    # Text processing (read-only, no shell execution)
    "cut",
    "tr",
    "diff",
    "md5sum",
    "sha256sum",
    # Path utilities
    "pwd",
    "which",
    # System info (read-only)
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "uptime",
    "nproc",
    "lscpu",
    "lsmem",
    # Process viewing (read-only)
    "ps",
)


def contains_dangerous_patterns(command: str) -> bool:
    """Check if a command contains dangerous shell patterns.

    These patterns can be used to bypass allow-list validation by embedding
    arbitrary commands within seemingly safe commands. The check includes
    both literal substring patterns (redirects, substitution operators, etc.)
    and regex patterns for bare variable expansion (`$VAR`) and the background
    operator (`&`).

    Args:
        command: The shell command to check.

    Returns:
        True if dangerous patterns are found, False otherwise.
    """
    if any(pattern in command for pattern in DANGEROUS_SHELL_PATTERNS):
        return True

    # Bare variable expansion ($VAR without braces) can leak sensitive paths.
    # We already block ${ and $( above; this catches plain $HOME, $IFS, etc.
    if re.search(r"\$[A-Za-z_]", command):
        return True

    # Standalone & (background execution) changes the execution model and
    # should not be allowed.  We check for & that is NOT part of &&.
    return bool(re.search(r"(?<![&])&(?![&])", command))


def is_shell_command_allowed(command: str, allow_list: list[str] | None) -> bool:
    """Check if a shell command is in the allow-list.

    The allow-list matches against the first token of the command (the executable name).
    This allows read-only commands like ls, cat, grep, etc. to be auto-approved.

    SECURITY: This function rejects commands containing dangerous shell patterns
    (command substitution, redirects, process substitution, etc.) BEFORE parsing,
    to prevent injection attacks that could bypass the allow-list.

    Args:
        command: The full shell command to check
        allow_list: List of allowed command names (e.g., ["ls", "cat", "grep"])

    Returns:
        True if the command is allowed, False otherwise.
    """
    if not allow_list or not command or not command.strip():
        return False

    # SECURITY: Check for dangerous patterns BEFORE any parsing
    # This prevents injection attacks like: ls "$(rm -rf /)"
    if contains_dangerous_patterns(command):
        return False

    allow_set = set(allow_list)

    # Extract the first command token
    # Handle pipes and other shell operators by checking each command in the pipeline
    # Split by compound operators first (&&, ||), then single-char operators (|, ;).
    # Note: standalone & (background) is blocked by contains_dangerous_patterns above.
    segments = re.split(r"&&|\|\||[|;]", command)

    # Track if we found at least one valid command
    found_command = False

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        try:
            # Try to parse as shell command to extract the executable name
            tokens = shlex.split(segment)
            if tokens:
                found_command = True
                cmd_name = tokens[0]
                # Check if this command is in the allow set
                if cmd_name not in allow_set:
                    return False
        except ValueError:
            # If we can't parse it, be conservative and require approval
            return False

    # All segments are allowed (and we found at least one command)
    return found_command


def get_langsmith_project_name() -> str | None:
    """Resolve the LangSmith project name if tracing is configured.

    Checks for the required API key and tracing environment variables.
    When both are present, resolves the project name with priority:
    `settings.deepagents_langchain_project` (from
    `DEEPAGENTS_LANGSMITH_PROJECT`), then `LANGSMITH_PROJECT` from the
    environment (note: this may already have been overridden at import
    time to match `DEEPAGENTS_LANGSMITH_PROJECT`), then `'default'`.

    Returns:
        Project name string when LangSmith tracing is active, None otherwise.
    """
    langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
        "LANGCHAIN_API_KEY"
    )
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get(
        "LANGCHAIN_TRACING_V2"
    )
    if not (langsmith_key and langsmith_tracing):
        return None

    return (
        settings.deepagents_langchain_project
        or os.environ.get("LANGSMITH_PROJECT")
        or "default"
    )


def fetch_langsmith_project_url(project_name: str) -> str | None:
    """Fetch the LangSmith project URL via the LangSmith client.

    This is a blocking network call. In async contexts, run it in a thread
    (e.g. via `asyncio.to_thread`).

    Returns None (with a debug log) on any expected failure: missing
    `langsmith` package, network errors, invalid project names, or client
    initialization issues.

    Args:
        project_name: LangSmith project name to look up.

    Returns:
        Project URL string if found, None otherwise.
    """
    try:
        from langsmith import Client

        project = Client().read_project(project_name=project_name)
    except (ImportError, OSError, ValueError, RuntimeError):
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=True,
        )
        return None
    else:
        return project.url or None


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (AGENTS.md) is handled separately by the middleware.

    Returns:
        The default agent instructions as a string.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def _detect_provider(model_name: str) -> str | None:
    """Auto-detect provider from model name.

    Args:
        model_name: Model name to detect provider from

    Returns:
        Provider name (deepseek, openai, anthropic, google, vertexai, claude-code) or None if can't detect
    """
    model_lower = model_name.lower()

    # Check for model name patterns
    if "deepseek" in model_lower:
        return "deepseek"
    if any(x in model_lower for x in ["gpt", "o1", "o3"]):
        return "openai"

    # Check for Claude Code simple names (sonnet, haiku, opus, inherit)
    # These are Claude Code-specific short names that Claude SDK resolves
    if model_lower in ("sonnet", "haiku", "opus", "inherit"):
        return "claude-code"

    if "claude" in model_lower:
        if not settings.has_anthropic and settings.has_vertex_ai:
            return "vertexai"
        return "anthropic"
    if "gemini" in model_lower:
        if settings.has_vertex_ai:
            return "vertexai"
        return "google"

    return None


# The predicate itself lives in the dependency-free leaf module
# ``deep_worker_cli.model_predicate`` so that the web frontend can import
# it without pulling in langchain. It is re-exported here because this is the
# import path every existing caller already uses (agent_command,
# agent_service, agent_config). Do NOT reimplement it here: a hand-copied
# duplicate in the web router is exactly what let the stale Claude Code
# whitelist survive in two places and silently drop ``--model fable``
# (see a related internal incident analysis (not included here)).
model_can_belong_to_provider = _leaf_model_can_belong_to_provider


# Provider → (env_var, hardcoded_default) mapping.
# Single source of truth for default model names.
_PROVIDER_DEFAULT_MODELS: dict[str, tuple[str | None, str | None]] = {
    "deepseek": ("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "openai": ("OPENAI_MODEL", "gpt-5.2"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
    "google": ("GOOGLE_MODEL", "gemini-3-pro-preview"),
    "vertexai": ("VERTEX_AI_MODEL", "gemini-3-pro-preview"),
    # Claude Code: do NOT pin a model. The real default depends on the
    # account type — Opus 5 for Anthropic API / Max / Team Premium /
    # Enterprise PAYG / Bedrock / AWS / GCP, Sonnet 5 for Pro / Team
    # Standard / Enterprise seats, Sonnet 4.5 on Microsoft Foundry (see
    # code.claude.com/docs/en/model-config#default-model-setting).
    # deep-worker cannot know which applies, so it passes no ``--model``
    # and lets the claude CLI resolve its own default — the same contract
    # codex and grok already use. The previous hardcoded ``"sonnet"``
    # silently overrode a rejected ``--model fable`` (see a related internal
    # incident analysis (not included here)). CLAUDE_CODE_MODEL pins it when
    # the operator wants.
    "claude-code": ("CLAUDE_CODE_MODEL", None),
    # Codex: do NOT pin a model version. Let Codex CLI pick its own
    # default (which evolves with the CLI). If the user sets
    # CODEX_MODEL, we honour it; otherwise we pass nothing and the
    # CLI chooses. See DESIGN_CODEX_PROVIDER.md §15 Q2.
    "codex": ("CODEX_MODEL", None),
    # Grok: do NOT pin a model version. Let the grok CLI pick its own
    # default (which evolves with the CLI). If the user sets GROK_MODEL we
    # honour it; otherwise we pass nothing. See DESIGN_GROK_PROVIDER.md §16.
    "grok": ("GROK_MODEL", None),
}


def get_default_model_name(provider: str) -> str | None:
    """Get the default model name for a provider.

    Checks environment variable first, then falls back to hardcoded default.
    Returns None for unknown providers.

    Args:
        provider: Provider name (e.g. "deepseek", "openai", "anthropic")

    Returns:
        Default model name string, or None if provider is unknown.
    """
    entry = _PROVIDER_DEFAULT_MODELS.get(provider)
    if entry is None:
        return None
    env_var, fallback = entry
    if env_var:
        return os.environ.get(env_var, fallback)
    return fallback


def create_model(
    model_name_override: str | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    user: str | None = None,
    kind: str | None = None,
    stats_callback_factory: Callable[..., Any] | None = None,
    usage_event_callback_factory: Callable[..., Any] | None = None,
) -> BaseChatModel:
    """Create the appropriate model based on provider and model name.

    Args:
        model_name_override: DEPRECATED - First positional arg for backward compatibility.
                            Use model_name keyword arg instead.
        provider: Provider name ("deepseek", "openai", "anthropic", "google", "vertexai", "claude-code")
                 If None, auto-detects based on available API keys or model name.
        model_name: Model name (e.g., "deepseek-chat", "gpt-4o", "sonnet", "haiku", "opus")
                   If None, uses environment variable or provider default.
        user: Optional user identifier for API-level KV-cache isolation
              (e.g. DeepSeek's ``user_id``). Passed via ``model_kwargs``.
              When set, different users get separate KV caches so that
              concurrent agents do not evict each other's prefixed cache
              entries. See DESIGN_CACHE_USER_ID_AND_MODEL_UPDATE.md.
        kind: Token-stats kind ("agent" | "tm" | "other"); when a stats
              factory is provided, this tags every recorded usage record.
        stats_callback_factory: Optional ``(kind, *, provider, model_name)``
              factory producing the LangChain callback handler for standard
              providers (DESIGN_AGENT_TOKEN_STATS.md §5.1). Called AFTER
              provider/model resolution so the handler records into the
              resolved provider bucket (failover-aware).
        usage_event_callback_factory: Optional ``(kind, *, provider, model_name)``
              factory producing the event callback installed via
              ``model.set_usage_event_callback`` on custom providers
              (claude-code/codex/grok).

    Returns:
        ChatModel instance (OpenAI, Anthropic, Google, or ClaudeCode)

    Raises:
        SystemExit if no API key is configured or model provider can't be determined
    """
    # Support legacy positional argument for backward compatibility
    # Old API: create_model("model-name")
    # New API: create_model(provider="...", model_name="...")
    if model_name_override and not model_name:
        model_name = model_name_override

    # Determine provider and model
    # CASE 1: Explicit provider provided - use it (model_name may or may not be provided)
    if provider:
        # Validate provider has credentials (except claude-code which doesn't need API key)
        if provider == "deepseek" and not settings.has_deepseek:
            console.print(f"[bold red]Error:[/bold red] Provider '{provider}' requires DEEPSEEK_API_KEY")
            sys.exit(1)
        elif provider == "openai" and not settings.has_openai:
            console.print(f"[bold red]Error:[/bold red] Provider '{provider}' requires OPENAI_API_KEY")
            sys.exit(1)
        elif provider == "anthropic" and not settings.has_anthropic:
            console.print(f"[bold red]Error:[/bold red] Provider '{provider}' requires ANTHROPIC_API_KEY")
            sys.exit(1)
        elif provider == "google" and not settings.has_google:
            console.print(f"[bold red]Error:[/bold red] Provider '{provider}' requires GOOGLE_API_KEY")
            sys.exit(1)
        elif provider == "vertexai" and not settings.has_vertex_ai:
            console.print(
                f"[bold red]Error:[/bold red] Provider '{provider}' requires GOOGLE_CLOUD_PROJECT"
            )
            sys.exit(1)
        # claude-code and codex don't need API key validation
        # (they use cached local CLI login)

        # Provider is valid and has credentials (or doesn't need them)
        # If model_name is None, get from environment or use provider default
        if not model_name:
            model_name = get_default_model_name(provider)
        # Continue with this provider and model_name

    # CASE 2: Only model_name provided (old behavior) - auto-detect provider
    elif model_name:
        # Use provided model, auto-detect provider
        provider = _detect_provider(model_name)
        if not provider:
            console.print(
                "[bold red]Error:[/bold red] Could not detect provider "
                f"from model name: {model_name}"
            )
            console.print("\nSupported model name patterns:")
            console.print("  - Deepseek: deepseek-* (DEFAULT PROVIDER)")
            console.print("  - OpenAI: gpt-*, o1-*, o3-*")
            console.print("  - Anthropic: claude-*")
            console.print("  - Google: gemini-* (requires GOOGLE_API_KEY)")
            console.print(
                "  - VertexAI: claude-*/gemini-* (requires GOOGLE_CLOUD_PROJECT, "
                "uses Application Default Credentials)"
            )
            sys.exit(1)

        # Check if credentials for detected provider are available
        if provider == "deepseek" and not settings.has_deepseek:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' "
                "requires DEEPSEEK_API_KEY"
            )
            sys.exit(1)
        elif provider == "openai" and not settings.has_openai:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' "
                "requires OPENAI_API_KEY"
            )
            sys.exit(1)
        elif provider == "anthropic" and not settings.has_anthropic:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' "
                "requires ANTHROPIC_API_KEY"
            )
            sys.exit(1)
        elif provider == "google" and not settings.has_google:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' "
                "requires GOOGLE_API_KEY"
            )
            sys.exit(1)
        elif provider == "vertexai" and not settings.has_vertex_ai:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' requires "
                "GOOGLE_CLOUD_PROJECT to be set"
            )
            console.print("\nPlease set GOOGLE_CLOUD_PROJECT environment variable.")
            console.print("Also ensure you have authenticated with:")
            console.print("  gcloud auth application-default login")
            sys.exit(1)

        # model_name already set from parameter

    # CASE 3: No provider or model_name provided - auto-detect (fallback for backward compatibility)
    # ============================================================================
    # ⚠️⚠️⚠️ CRITICALLY IMPORTANT FOR FUTURE AGENTS ⚠️⚠️⚠️
    # ============================================================================
    # DO NOT CHANGE THE PROVIDER PRIORITY ORDER!
    # DO NOT REMOVE THE has_deepseek CHECK!
    #
    # This code is used by the OLD main.py entry point (python3 -m deep_worker_cli.main)
    # which must NOT be used for testing or production!
    #
    # THE ONLY CORRECT way to launch:
    # ✅ deep-worker [agent-name]  - command installed as `deep-worker` on PATH
    #
    # WRONG (DO NOT USE):
    # ❌ python3 -m deep_worker_cli.main
    # ❌ python3 -m deep_worker_cli.cli.main
    #
    # The deep-worker command uses agent_command.py which explicitly sets provider="deepseek"
    # by default, so it now PASSES the provider explicitly into create_model().
    #
    # See: DEEPSEEK_PROVIDER.md for details
    # ============================================================================
    # Provider auto-detection by API key priority (Deepseek FIRST)
    else:
        if settings.has_deepseek:
            provider = "deepseek"
            model_name = get_default_model_name("deepseek")
        elif settings.has_openai:
            provider = "openai"
            model_name = get_default_model_name("openai")
        elif settings.has_anthropic:
            provider = "anthropic"
            model_name = get_default_model_name("anthropic")
        elif settings.has_google:
            provider = "google"
            model_name = get_default_model_name("google")
        elif settings.has_vertex_ai:
            provider = "vertexai"
            model_name = get_default_model_name("vertexai")
        else:
            console.print("[bold red]Error:[/bold red] No credentials configured.")
            console.print("\nPlease set one of the following environment variables:")
            console.print("  - DEEPSEEK_API_KEY   (for Deepseek models - DEFAULT PROVIDER)")
            console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5.2)")
            console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
            console.print("  - GOOGLE_API_KEY     (for Google Gemini models)")
            console.print(
                "  - GOOGLE_CLOUD_PROJECT (for VertexAI models, "
                "with Application Default Credentials)"
            )
            console.print("\nExample:")
            console.print("  export DEEPSEEK_API_KEY=your_api_key_here")
            console.print("\nOr add it to your .env file.")
            sys.exit(1)

    # Store model info in settings for display
    settings.model_name = model_name
    settings.model_provider = provider

    # Create the model
    model: BaseChatModel
    if provider == "deepseek":
        # ⚠️ Use ChatDeepSeek, NOT a bare ChatOpenAI.
        #
        # The default deepseek model (deepseek-v4-flash) is a THINKING /
        # reasoning model: it returns ``reasoning_content`` alongside
        # ``content``, and DeepSeek's API REQUIRES that ``reasoning_content``
        # be passed BACK on every subsequent turn. A plain ``ChatOpenAI`` does
        # not understand that non-standard field, so it dropped it on the
        # round-trip and the next call was rejected with:
        #   400 - "The `reasoning_content` in the thinking mode must be passed
        #          back to the API."
        # → the agent crashed to ERROR (observed in a past incident). We use
        # ``ReasoningChatDeepSeek`` (subclass of ``langchain_deepseek``'s
        # ``ChatDeepSeek``): ChatDeepSeek READS reasoning_content inbound but
        # its 1.0.1 ``_get_request_payload`` does NOT re-send it outbound, so
        # the 400 still fired on multi-step turns. Our subclass re-injects
        # reasoning_content onto outgoing assistant messages, closing the loop.
        # DO NOT revert to bare ChatDeepSeek (let alone ChatOpenAI) until the
        # upstream library re-sends reasoning_content itself.
        from deep_worker_cli.deepseek_model import ReasoningChatDeepSeek

        ds_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_base": "https://api.deepseek.com",
            "api_key": settings.deepseek_api_key,
            # Explicit guarantee for token statistics — NOT the fix for the
            # 2026-08-03 incident (that was a normalizer gap: the aggregate
            # ``AIMessage.usage_metadata`` shape was not understood; see
            # ``_from_langchain_usage_metadata``). Live-verified: DeepSeek
            # streams usage even WITHOUT this flag.
            #
            # We set it anyway because statistics are wanted by default and
            # the upstream default is fragile: ``stream_options=
            # {"include_usage": true}`` is only sent when
            # ``self.stream_usage`` is on, and langchain_openai auto-enables
            # it ONLY for the default base URL (``chat_models/base.py``
            # requires ``openai_api_base is None``). We always pass an
            # explicit ``api_base``, so the flag would otherwise stay off and
            # any endpoint/config that does omit usage without it would
            # silently produce ``no_usage`` records.
            "stream_usage": True,
        }
        if user is not None:
            # Pass user_id for KV-cache isolation (DeepSeek API).
            # Different values (e.g. "agent_name" vs "agent_name:tm")
            # get separate caches so concurrent threads don't evict
            # each other's prefix-cache entries.
            #
            # NOTE: only set model_kwargs when there IS a user — ChatDeepSeek
            # rejects ``model_kwargs=None`` (it iterates the dict), unlike the
            # old ChatOpenAI path which tolerated None.
            ds_kwargs["model_kwargs"] = {"user": user}

        model = ReasoningChatDeepSeek(**ds_kwargs)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        # stream_usage=True: same policy as the deepseek branch above — an
        # explicit guarantee that streamed responses carry token counts for
        # the statistics subsystem, rather than relying on the upstream
        # base-URL heuristic.
        model = ChatOpenAI(model=model_name, stream_usage=True)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name=model_name,
            max_tokens=20_000,
        )
    elif provider == "google":
        try:
            # Optional dependency; install `langchain-google-genai`
            # (core dependency of the wheel) to enable Gemini models. The
            # import is guarded so an environment without the package gets
            # an actionable CapabilityError instead of a raw
            # ModuleNotFoundError.
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise CapabilityError(
                capability="google",
                remedy="pip install langchain-google-genai",
            )

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
        )
    elif provider == "vertexai":
        assert model_name is not None  # Type narrowing: model_name is set above
        model_lower = model_name.lower()

        if "claude" in model_lower:
            try:
                # Optional dependency; install `langchain-google-vertexai`
                # to enable Vertex AI's Anthropic proxy. The import is
                # guarded by the enclosing try/except ImportError.
                from langchain_google_vertexai.model_garden import (  # ty: ignore[unresolved-import]
                    ChatAnthropicVertex,
                )
            except ImportError:
                console.print(
                    "[bold red]Error:[/bold red] langchain-google-vertexai "
                    "package is required for this model"
                )
                console.print("\nInstall it with:")
                console.print("  pip install deepagents-cli[vertexai]", markup=False)
                sys.exit(1)

            model = ChatAnthropicVertex(
                # Remove version tag (e.g., "claude-haiku-4-5@20251015" ->
                # "claude-haiku-4-5"). ChatAnthropicVertex expects just the base
                # model name without the @version suffix.
                model_name=model_name,
                project=settings.google_cloud_project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
                max_tokens=20_000,
            )
        else:
            try:
                # Optional dependency; the vertexai extra installs
                # `langchain-google-vertexai` which pulls in
                # `langchain-google-genai`. The import is guarded so an
                # environment without the package gets an actionable
                # CapabilityError instead of a raw ModuleNotFoundError.
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise CapabilityError(
                    capability="vertexai",
                    remedy="pip install deep-worker[vertexai]",
                )

            model = ChatGoogleGenerativeAI(
                model=model_name,
                project=settings.google_cloud_project,
                vertexai=True,
                temperature=0,
                max_tokens=None,
            )
    elif provider == "claude-code":
        # ============================================================================
        # CRITICAL: CLAUDE_CODE_PROVIDER MUST BE IMPORTED FROM SOURCE CODE!
        # ============================================================================
        # Import claude_code_provider from a local source checkout
        # This is a local package that provides LangChain wrapper for Claude Code SDK
        # ============================================================================
        _repo_root = Path(__file__).parent.parent.parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        from claude_code_provider import ClaudeCodeChatModel

        # Claude Code accepts simple model names (sonnet, haiku, opus, inherit)
        # or full model names (claude-sonnet-4-5-20250929)
        # The Claude SDK automatically resolves simple names to latest versions
        model = ClaudeCodeChatModel(
            model=model_name,
            max_tokens=20_000,
        )
    elif provider == "codex":
        # ============================================================================
        # CODEX provider — vendored package + codex CLI (see DESIGN_CODEX_PROVIDER.md).
        # ============================================================================
        # Resolve the provider origin through provider_paths (source root on
        # sys.path in source mode; no-op in installed mode). See
        # deep_worker_cli/provider_paths.py and DESIGN_PROVIDERS_FROM_WHEEL §3.1.
        # ``model_name`` may be None — in that case the provider does NOT
        # pass ``--model`` to ``codex exec`` and the CLI picks its own
        # current default (Q2: do not pin model version).
        # ============================================================================
        from deep_worker_cli.provider_paths import ensure_provider_importable

        resolution = ensure_provider_importable("deep_worker.providers.codex_provider")
        if resolution.unavailable:
            raise CapabilityError(
                capability="codex",
                remedy=resolution.remedy or resolution.unavailable,
            )

        from deep_worker.providers.codex_provider import CodexChatModel

        model = CodexChatModel(
            model=model_name,            # may be None — Codex picks default
            max_tokens=20_000,
            # sandbox_mode/approval_policy defaults match Q3 (always-skip-prompts)
        )
    elif provider == "grok":
        # ============================================================================
        # GROK provider — vendored package + grok CLI (see DESIGN_GROK_PROVIDER.md).
        # ============================================================================
        # Resolve the provider origin through provider_paths (source root on
        # sys.path in source mode; no-op in installed mode). See
        # deep_worker_cli/provider_paths.py and DESIGN_PROVIDERS_FROM_WHEEL §3.1.
        # ``model_name`` may be None — in that case the provider does NOT
        # pin a model and the grok CLI picks its own current default. Drives
        # ``grok agent stdio`` over ACP.
        # ============================================================================
        from deep_worker_cli.provider_paths import ensure_provider_importable

        resolution = ensure_provider_importable("deep_worker.providers.grok_provider")
        if resolution.unavailable:
            raise CapabilityError(
                capability="grok",
                remedy=resolution.remedy or resolution.unavailable,
            )

        from deep_worker.providers.grok_provider import GrokChatModel

        model = GrokChatModel(
            model=model_name,            # may be None — grok picks default
            max_tokens=20_000,
        )
    else:
        # Should not reach here due to earlier validation
        console.print(f"[bold red]Error:[/bold red] Unknown provider: {provider}")
        sys.exit(1)

    # Extract context limit from model profile (if available)
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int):
        settings.model_context_limit = profile["max_input_tokens"]

    # Token-stats instrumentation (DESIGN_AGENT_TOKEN_STATS.md §5.1, §7).
    # Standard LangChain providers get an instance-level AsyncCallbackHandler;
    # custom providers (claude-code/codex/grok) get ONLY the event callback via
    # ``set_usage_event_callback`` — a double source is impossible structurally.
    if stats_callback_factory is not None or usage_event_callback_factory is not None:
        _attach_token_stats(
            model,
            provider=provider,
            model_name=model_name or "",
            kind=kind or "other",
            stats_callback_factory=stats_callback_factory,
            usage_event_callback_factory=usage_event_callback_factory,
        )

    return model


def _attach_token_stats(
    model: BaseChatModel,
    *,
    provider: str,
    model_name: str,
    kind: str,
    stats_callback_factory: Callable[..., Any] | None,
    usage_event_callback_factory: Callable[..., Any] | None,
) -> None:
    """Attach token-stats instrumentation to a freshly created model (§5.1).

    Standard LangChain providers (deepseek/openai/anthropic/google/vertexai)
    receive a ``TokenStatsCallbackHandler`` on the model instance (inherited by
    ``bind_tools`` copies → summarization included). Custom providers
    (claude-code/codex/grok) receive ONLY ``set_usage_event_callback(cb)`` —
    never the LangChain handler, so a double usage source is impossible
    structurally (MAJOR-2, DESIGN_AGENT_TOKEN_STATS.md).
    """
    from deep_worker_cli.stats.token_stats import (
        CUSTOM_PROVIDERS,
        STANDARD_PROVIDERS,
    )

    try:
        if provider in STANDARD_PROVIDERS and stats_callback_factory is not None:
            from deep_worker_cli.stats.callback import TokenStatsCallbackHandler

            handler = stats_callback_factory(kind, provider=provider, model_name=model_name)
            if not isinstance(handler, TokenStatsCallbackHandler):
                logger.warning("token-stats: factory returned %r, skipping", type(handler).__name__)
                return
            existing = list(getattr(model, "callbacks", None) or [])
            if not any(isinstance(cb, TokenStatsCallbackHandler) for cb in existing):
                model.callbacks = [*existing, handler]
        elif provider in CUSTOM_PROVIDERS and usage_event_callback_factory is not None:
            callback = usage_event_callback_factory(kind, provider=provider, model_name=model_name)
            setter = getattr(model, "set_usage_event_callback", None)
            if callable(setter):
                setter(callback)
            else:
                logger.warning(
                    "token-stats: provider %r has no set_usage_event_callback — "
                    "usage events will not be recorded",
                    provider,
                )
    except Exception:  # noqa: BLE001 — instrumentation must never break create_model
        logger.exception(
            "token-stats: failed to attach instrumentation (provider=%s)", provider
        )


def validate_model_capabilities(model: BaseChatModel, model_name: str) -> None:
    """Validate that the model has required capabilities for `deepagents`.

    Checks the model's profile (if available) to ensure it supports tool calling, which
    is required for agent functionality. Issues warnings for models without profiles or
    with limited context windows.

    Args:
        model: The instantiated model to validate.
        model_name: Model name for error/warning messages.

    Note:
        This validation is best-effort. Models without profiles will pass with
        a warning. Exits via sys.exit(1) if model profile explicitly indicates
        tool_calling=False.
    """
    profile = getattr(model, "profile", None)

    if profile is None:
        # Model doesn't have profile data - warn but allow
        console.print(
            f"[dim][yellow]Note:[/yellow] No capability profile for "
            f"'{model_name}'. Cannot verify tool calling support.[/dim]"
        )
        return

    if not isinstance(profile, dict):
        return

    # Check required capability: tool_calling
    tool_calling = profile.get("tool_calling")
    if tool_calling is False:
        console.print(
            f"[bold red]Error:[/bold red] Model '{model_name}' "
            "does not support tool calling."
        )
        console.print(
            "\nDeep Agents requires tool calling for agent functionality. "
            "Please choose a model that supports tool calling."
        )
        console.print("\nSee MODELS.md for supported models.")
        sys.exit(1)

    # Warn about potentially limited context (< 8k tokens)
    max_input_tokens = profile.get("max_input_tokens")
    if max_input_tokens and max_input_tokens < 8000:
        console.print(
            f"[dim][yellow]Warning:[/yellow] Model '{model_name}' has limited context "
            f"({max_input_tokens:,} tokens). Agent performance may be affected.[/dim]"
        )
