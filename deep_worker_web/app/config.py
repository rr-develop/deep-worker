"""Configuration for deep-worker-web application.

CRITICAL SECURITY REQUIREMENT:
==============================
This application MUST ONLY run on 127.0.0.1:7887 (localhost).
NO OTHER addresses or ports are allowed for security reasons.

This application is designed for LOCAL TRUSTED ENVIRONMENT ONLY:
- No authentication/authorization
- Direct access to deep-worker agents
- Full access to agent capabilities
- Must NEVER be exposed to external networks

Violating this requirement creates serious security risks.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with strict security validation."""

    host: str = "127.0.0.1"
    port: int = 7887
    debug: bool = True

    class Config:
        """Pydantic configuration."""

        env_file = ".env"

    def validate_security(self) -> bool:
        """Validate that the application is only running on localhost:7887.

        Returns:
            True if security check passes

        Raises:
            ValueError: If host is not 127.0.0.1 or port is not 7887
        """
        if self.host != "127.0.0.1":
            raise ValueError(
                f"SECURITY ERROR: Host must be 127.0.0.1 only! Current: {self.host}\n"
                "This application must NEVER be exposed to external networks.\n"
                "Running on 0.0.0.0 or any other address is STRICTLY PROHIBITED."
            )

        if self.port != 7887:
            raise ValueError(
                f"SECURITY ERROR: Port must be 7887 only! Current: {self.port}\n"
                "This is a security requirement to prevent accidental exposure."
            )

        return True


# Initialize settings and validate security immediately
settings = Settings()
settings.validate_security()
