"""StockAI Configuration Management.

Handles settings from environment variables, config files, and CLI flags.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="STOCKAI_",
        extra="ignore",
    )

    # LLM API Keys
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Data Source API Keys
    firecrawl_api_key: str = Field(default="", alias="FIRECRAWL_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Model settings
    model: str = Field(default="gemini-3-flash-preview", description="Default LLM model")

    # Database
    db_path: str = Field(default="data/stockai.db", description="SQLite database path")

    # Cache
    cache_ttl: int = Field(default=900, description="Cache TTL in seconds")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Default focus
    default_index: str = Field(default="IDX30", description="Default stock index")

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    @property
    def db_full_path(self) -> Path:
        """Get full database path.

        Checks STOCKAI_DB_PATH env var first, falls back to relative path.
        """
        import os
        env_path = os.environ.get("STOCKAI_DB_PATH")
        if env_path:
            return Path(env_path)
        return self.project_root / self.db_path

    @property
    def has_google_api(self) -> bool:
        """Check if Google API key is configured."""
        return bool(self.google_api_key)

    @property
    def has_openai_api(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_anthropic_api(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_firecrawl_api(self) -> bool:
        """Check if Firecrawl API key is configured."""
        return bool(self.firecrawl_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Model mapping for LangChain
MODEL_MAP = {
    # Google Gemini models
    "gemini-3-flash-preview": "google_genai:gemini-3-flash-preview",
    "gemini-3-pro-preview": "google_genai:gemini-3-pro-preview",
    "gemini-2.5-flash": "google_genai:gemini-2.5-flash",
    "gemini-2.5-pro": "google_genai:gemini-2.5-pro",
    "gemini-2.0-flash": "google_genai:gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google_genai:gemini-2.0-flash-lite",
    # OpenAI models
    "gpt-4o": "openai:gpt-4o",
    "gpt-4": "openai:gpt-4",
    "gpt-4-turbo": "openai:gpt-4-turbo",
    # Anthropic models
    "claude-sonnet": "anthropic:claude-3-5-sonnet-20241022",
    "claude-opus": "anthropic:claude-3-opus-20240229",
}


def get_model_identifier(model_name: str) -> str:
    """Get LangChain model identifier from friendly name."""
    return MODEL_MAP.get(model_name, model_name)
