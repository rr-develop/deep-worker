"""Agent configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from deep_worker_cli.runtime.provider_chain import all_provider_names
from deep_worker_cli.tool_selection import parse_tool_selection

# Load environment variables from .env file
load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for a deep_worker agent.

    Attributes:
        model: Model name or identifier
        working_dir: Working directory for the agent
        provider: AI provider name ("deepseek", "gemini", "openai")
        system_prompt: Optional system prompt for the agent
        tools: Tool selection (``docs/DESIGN_TOOL_SELECTION.md``). Empty —
            the default — means every tool, unchanged. ``["-*"]`` denies all;
            ``["-*", "read_file"]`` denies all but the named; ``["-execute"]``
            removes only the named.
        backend_type: Type of backend ("state", "filesystem", etc.)
        max_tokens: Maximum tokens for model responses
        temperature: Sampling temperature for model
        api_key: Optional API key (defaults to DEEPSEEK_API_KEY from .env)
    """

    working_dir: Path
    model: str | None = None  # None = use provider default
    provider: str = "deepseek"
    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)
    backend_type: str = "state"
    max_tokens: int = 20000
    temperature: float = 1.0
    api_key: str | None = None
    web_search_policy: str = "auto"  # "auto" | "enabled" | "disabled" (WebSearchPolicy)
    _dynamic_display_model: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert working_dir to Path if it's a string
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)

        # Validate model (allow None - will use provider's default)
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a non-empty string or None")

        # Validate provider. Built-ins PLUS config-declared API providers
        # (~/.deep_worker/providers.json) so adding a standard OpenAI-
        # compatible vendor needs no code change.
        valid_providers = list(all_provider_names())
        if self.provider not in valid_providers:
            raise ValueError(
                f"provider must be one of {valid_providers}, got '{self.provider}'"
            )

        # Auto-load API key from environment if not provided
        if self.api_key is None:
            if self.provider == "claude-code-sdk":
                # Claude Code doesn't require API key (uses local installation)
                self.api_key = None
            elif self.provider == "claude-code":
                # claude-code authenticates from Claude Code's own
                # credential store (~/.claude.json subscription) — no key.
                self.api_key = None
            elif self.provider == "codex":
                # Codex CLI uses cached login at ~/.codex/auth.json — no key needed.
                self.api_key = None
            elif self.provider == "grok":
                # Grok CLI uses the supergrok subscription cache at
                # ~/.grok/auth.json — no separate API key needed.
                self.api_key = None
            elif self.provider == "deepseek":
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "DEEPSEEK_API_KEY not found in environment variables. "
                        "Add it to .env file: DEEPSEEK_API_KEY=sk-..."
                    )
            elif self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY not found in environment variables. "
                        "Add it to .env file: GOOGLE_API_KEY=AIza..."
                    )
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "OPENAI_API_KEY not found in environment variables. "
                        "Add it to .env file: OPENAI_API_KEY=sk-..."
                    )

        # Validate temperature
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {self.temperature}")

        # Validate max_tokens
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        # Validate the tool selection eagerly. A contradictory spec caught
        # at agent-construction time is a puzzle; caught here it names the
        # field the user actually wrote.
        parse_tool_selection(self.tools)

        # Validate web_search_policy (tri-state)
        if self.web_search_policy not in ("auto", "enabled", "disabled"):
            raise ValueError(
                f"web_search_policy must be one of 'auto', 'enabled', 'disabled', "
                f"got '{self.web_search_policy}'"
            )

    @property
    def resolved_model(self) -> str | None:
        """Return the actual model name, resolving provider default if needed.

        If self.model is set, returns it directly.
        Otherwise, uses get_default_model_name() to resolve the provider
        default (checking env vars first, then hardcoded fallback).

        Returns ``None`` when the provider has no default and the user
        did not specify a model — this is the correct value for every
        CLI-wrapper provider: **codex**, **grok** and (since 2026-07-28)
        **claude-code-sdk**. Each wraps a CLI that resolves its own current
        default when no ``--model`` is passed, and for claude-code-sdk that
        default legitimately varies by account type, so pinning one here
        would silently override the operator (see ``config.py``
        ``_PROVIDER_DEFAULT_MODELS``).
        ``None`` MUST be preserved through to ``create_model``;
        it must NOT be coerced to the literal ``"unknown"``, which the
        Codex API rejects with HTTP 400 (observed in a past incident).
        For display, use ``display_model`` instead.
        """
        if self.model:
            return self.model
        from deep_worker_cli.config import get_default_model_name
        return get_default_model_name(self.provider)

    def set_dynamic_display_model(self, model: str | None) -> None:
        from deep_worker_cli.config import model_can_belong_to_provider

        if (
            isinstance(model, str)
            and model.strip()
            and model.strip().lower() != "unknown"
            and model_can_belong_to_provider(self.provider, model)
        ):
            self._dynamic_display_model = model.strip()

    @property
    def dynamic_display_model(self) -> str | None:
        return self._dynamic_display_model

    @property
    def display_model(self) -> str:
        """Human-facing model label for UI / logs ONLY.

        Falls back to ``"unknown"`` when no concrete model can be
        resolved. NEVER feed this into ``create_model`` or persist it as
        the launch_config model — that is what caused a past incident. Use
        ``resolved_model`` (which may be ``None``) for the data path.
        """
        return self.dynamic_display_model or self.resolved_model or "unknown"

    @classmethod
    def from_file(cls, config_path: Path) -> "AgentConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with config_path.open() as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise ValueError("Configuration file must contain a YAML dictionary")

            # Extract known fields
            config_data: dict[str, Any] = {
                "model": data.get("model"),
                "working_dir": data.get("working_dir"),
                "provider": data.get("provider", "deepseek"),
                "system_prompt": data.get("system_prompt"),
                "tools": data.get("tools", []),
                "backend_type": data.get("backend_type", "state"),
                "max_tokens": data.get("max_tokens", 20000),
                "temperature": data.get("temperature", 1.0),
                "api_key": data.get("api_key"),  # Optional, auto-loaded if None
                "web_search_policy": data.get("web_search_policy", "auto"),
            }

            # Validate required fields
            if not config_data["model"]:
                raise ValueError("Configuration must specify 'model'")
            if not config_data["working_dir"]:
                raise ValueError("Configuration must specify 'working_dir'")

            return cls(**config_data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e
