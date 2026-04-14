"""Agent configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

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
        tools: List of tool names or configurations
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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert working_dir to Path if it's a string
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)

        # Validate model (allow None - will use provider's default)
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a non-empty string or None")

        # Validate provider
        valid_providers = ["deepseek", "gemini", "openai", "claude-code"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"provider must be one of {valid_providers}, got '{self.provider}'"
            )

        # Auto-load API key from environment if not provided
        if self.api_key is None:
            if self.provider == "claude-code":
                # Claude Code doesn't require API key (uses local installation)
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

    @property
    def resolved_model(self) -> str:
        """Return the actual model name, resolving provider default if needed.

        If self.model is set, returns it directly.
        Otherwise, uses get_default_model_name() to resolve the provider default
        (checking env vars first, then hardcoded fallback).
        """
        if self.model:
            return self.model
        from deep_worker_cli.config import get_default_model_name
        return get_default_model_name(self.provider) or "unknown"

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
            }

            # Validate required fields
            if not config_data["model"]:
                raise ValueError("Configuration must specify 'model'")
            if not config_data["working_dir"]:
                raise ValueError("Configuration must specify 'working_dir'")

            return cls(**config_data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e
