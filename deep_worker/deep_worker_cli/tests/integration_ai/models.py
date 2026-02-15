"""Pydantic models for AI-powered integration testing infrastructure.

All models use strict type checking and validation for reliability.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AgentStatus(str, Enum):
    """Agent lifecycle status."""

    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class AgentFixtureInfo(BaseModel):
    """Information about an agent managed by agent_fixture.

    This model represents an agent session created or reused by the
    agent_fixture context manager for testing.
    """

    agent_name: str = Field(
        ...,
        description="Name of the agent (must start with 'test_' for test agents)",
        pattern=r"^test_[a-z0-9_]+$",
    )

    working_dir: Path = Field(
        ...,
        description="Working directory for the agent",
    )

    socket_path: Path = Field(
        ...,
        description="Path to Unix socket for IPC",
    )

    pid: Optional[int] = Field(
        None,
        description="Process ID of the agent",
    )

    status: AgentStatus = Field(
        default=AgentStatus.RUNNING,
        description="Current agent status",
    )

    created_by_fixture: bool = Field(
        ...,
        description="Whether this agent was created by the fixture (vs pre-existing)",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the agent was started",
    )

    model_config = {"frozen": False}

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Ensure agent name follows test naming convention."""
        if not v.startswith("test_"):
            raise ValueError(
                f"Agent name must start with 'test_' for safety. Got: {v}"
            )
        return v

    @field_validator("working_dir", "socket_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class CheckReplyResult(BaseModel):
    """Result of AI-powered reply validation.

    This model represents the outcome of using a meta-query to validate
    whether an AI response correctly answers a given query.
    """

    is_valid: bool = Field(
        ...,
        description="Whether the reply correctly answers the query",
    )

    comment: str = Field(
        default="",
        description="Explanation if invalid, empty if valid",
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) if available",
    )

    model_used: str = Field(
        default="haiku",
        description="Claude model used for validation (haiku/sonnet/opus)",
    )

    validation_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time taken for validation in milliseconds",
    )

    cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated cost in USD for this validation",
    )

    model_config = {"frozen": True}

    def __bool__(self) -> bool:
        """Allow usage in boolean context: if result: ..."""
        return self.is_valid

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.is_valid:
            return f"✓ Valid (model: {self.model_used})"
        return f"✗ Invalid: {self.comment} (model: {self.model_used})"


class MetaQueryRequest(BaseModel):
    """Request for meta-query validation.

    This model structures the request to validate an AI response using
    another AI query (meta-query).
    """

    original_query: str = Field(
        ...,
        min_length=1,
        description="The original query that was sent to the agent",
    )

    reply_text: str = Field(
        ...,
        min_length=1,
        description="The agent's reply to validate",
    )

    validation_task: str = Field(
        default="Validate if the reply correctly answers the query",
        description="Specific validation task for the meta-query",
    )

    model: Literal["haiku", "sonnet", "opus"] = Field(
        default="haiku",
        description="Claude model to use for validation",
    )

    additional_context: Optional[str] = Field(
        None,
        description="Additional context for validation (optional)",
    )

    strict_mode: bool = Field(
        default=False,
        description="Whether to use strict validation (more expensive)",
    )

    model_config = {"frozen": True}

    def build_prompt(self) -> str:
        """Build structured meta-query prompt with injection protection.

        Returns:
            Formatted prompt string safe from text injection.
        """
        sections = [
            "You are an expert validator. Your task is to determine if a reply correctly answers a query.",
            "",
            "=== ORIGINAL QUERY ===",
            self.original_query,
            "",
            "=== REPLY TO VALIDATE ===",
            self.reply_text,
            "",
        ]

        if self.additional_context:
            sections.extend(
                [
                    "=== ADDITIONAL CONTEXT ===",
                    self.additional_context,
                    "",
                ]
            )

        sections.extend(
            [
                "=== VALIDATION TASK ===",
                self.validation_task,
                "",
                "=== INSTRUCTIONS ===",
                "Analyze whether the reply correctly answers the original query.",
                "",
                "Respond ONLY with valid JSON in this exact format:",
                "{",
                '  "is_valid": true or false,',
                '  "comment": "explanation if invalid, empty string if valid",',
                '  "confidence": 0.0 to 1.0',
                "}",
                "",
                "Rules:",
                "- is_valid: true if reply answers query correctly",
                "- comment: explain what's wrong if invalid, empty if valid",
                "- confidence: your confidence level (0.0 = no confidence, 1.0 = certain)",
            ]
        )

        if self.strict_mode:
            sections.extend(
                [
                    "",
                    "STRICT MODE: Be very precise and demanding in validation.",
                ]
            )

        return "\n".join(sections)


class ValidationError(Exception):
    """Exception raised when validation fails unexpectedly.

    This is different from a reply being invalid (which is a normal test failure).
    This is raised when the validation infrastructure itself fails.
    """

    def __init__(
        self,
        message: str,
        original_query: Optional[str] = None,
        reply_text: Optional[str] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error description
            original_query: The query being validated (if available)
            reply_text: The reply being validated (if available)
            raw_response: Raw response from validation query (if available)
        """
        super().__init__(message)
        self.message = message
        self.original_query = original_query
        self.reply_text = reply_text
        self.raw_response = raw_response

    def __str__(self) -> str:
        """Detailed error message."""
        parts = [f"ValidationError: {self.message}"]

        if self.original_query:
            parts.append(f"\nOriginal query: {self.original_query[:100]}...")

        if self.reply_text:
            parts.append(f"\nReply text: {self.reply_text[:100]}...")

        if self.raw_response:
            parts.append(f"\nRaw validation response: {self.raw_response[:200]}...")

        return "".join(parts)


class AgentProcessInfo(BaseModel):
    """Information about a running agent process.

    Used for querying agent status and debugging.
    """

    agent_name: str = Field(..., description="Agent name")
    pid: int = Field(..., description="Process ID")
    status: AgentStatus = Field(..., description="Current status")
    socket_path: Path = Field(..., description="IPC socket path")
    working_dir: Path = Field(..., description="Working directory")
    uptime_seconds: Optional[float] = Field(
        None, description="Time since agent started"
    )
    memory_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_percent: Optional[float] = Field(None, description="CPU usage percentage")

    model_config = {"frozen": True}


class TestExecutionMetrics(BaseModel):
    """Metrics collected during test execution.

    Tracks validation costs, timing, and resource usage.
    """

    test_name: str = Field(..., description="Test function name")
    validations_count: int = Field(default=0, description="Number of validations")
    validations_passed: int = Field(default=0, description="Passed validations")
    validations_failed: int = Field(default=0, description="Failed validations")

    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    total_time_ms: float = Field(default=0.0, description="Total time in milliseconds")

    model_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Count of validations per model (haiku/sonnet/opus)",
    )

    started_at: datetime = Field(
        default_factory=datetime.now, description="Test start time"
    )
    completed_at: Optional[datetime] = Field(None, description="Test completion time")

    model_config = {"frozen": False}

    def add_validation(self, result: CheckReplyResult) -> None:
        """Record a validation result."""
        self.validations_count += 1

        if result.is_valid:
            self.validations_passed += 1
        else:
            self.validations_failed += 1

        if result.cost_usd:
            self.total_cost_usd += result.cost_usd

        if result.validation_time_ms:
            self.total_time_ms += result.validation_time_ms

        # Track model usage
        model = result.model_used
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

    def complete(self) -> None:
        """Mark test as completed."""
        self.completed_at = datetime.now()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate test duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Test: {self.test_name}",
            f"Validations: {self.validations_count} ({self.validations_passed} passed, {self.validations_failed} failed)",
            f"Total cost: ${self.total_cost_usd:.4f} USD",
            f"Total time: {self.total_time_ms:.0f}ms",
        ]

        if self.model_usage:
            usage_str = ", ".join(
                f"{model}: {count}" for model, count in self.model_usage.items()
            )
            lines.append(f"Model usage: {usage_str}")

        if self.duration_seconds:
            lines.append(f"Duration: {self.duration_seconds:.2f}s")

        return "\n".join(lines)
