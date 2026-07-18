"""Configuration schema for LangGraph workflow execution."""

from pathlib import Path
from pydantic import BaseModel, Field

from src.core.paths import Paths


class WorkflowConfig(BaseModel):
    """Runtime controls for retries, checkpointing, and output behavior."""

    max_retries: int = Field(default=1, ge=0)
    checkpoint_mode: str = Field(default="memory", pattern="^(memory|file|none)$")
    checkpoint_dir: Path = Field(default=Paths.WORKSPACE_DIR / "checkpoints")
    persist_execution: bool = True
    template_type: str = "executive"
