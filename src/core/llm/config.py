"""YAML-driven configuration parser for LLM client connection and model choices."""

from typing import Dict, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

from src.core.logger import get_logger

logger = get_logger(__name__)


class LLMConfig(BaseModel):
    """Configuration representing API endpoints, model tags, and cache intervals."""

    provider: str = Field(default="mock", description="LLM provider key ('openai', 'azure', 'gemini', 'anthropic', 'local', 'mock')")
    model: str = Field(default="gpt-4o", description="Model version tag identifier")
    temperature: float = Field(default=0.0, description="Sampling temperature settings parameter")
    max_tokens: int = Field(default=2000, description="Max response completion limit")
    cache_ttl_seconds: int = Field(default=86400, description="Prompt cache duration lifetime in seconds (default 24h)")
    timeout_seconds: int = Field(default=60, description="Connection timeout limits in seconds")
    
    # Provider-specific configurations
    api_key: Optional[str] = Field(default=None, description="API authorization secret key")
    api_base: Optional[str] = Field(default=None, description="Base endpoint target URI")
    azure_endpoint: Optional[str] = Field(default=None, description="Azure API target URI endpoint")
    azure_api_version: Optional[str] = Field(default=None, description="Azure API version tag")

    # Financial cost tracking estimations per million tokens
    cost_per_million_input: float = Field(default=5.00, description="Pricing in USD per million prompt tokens")
    cost_per_million_output: float = Field(default=15.00, description="Pricing in USD per million completion tokens")

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Path] = None) -> "LLMConfig":
        """Load connection configurations from YAML file or return defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            LLMConfig: Parsed configuration object.
        """
        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                # Resolve llm subset block if present
                llm_content = content.get("llm", content)
                logger.info(f"Loaded LLM configurations from: {yaml_path}")
                return cls(**llm_content)
            except Exception as e:
                logger.warning(f"Failed to load yaml config from {yaml_path}: {e}. Using defaults.")
        return cls()
