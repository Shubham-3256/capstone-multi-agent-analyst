"""Schemas representing standardized AI agent response structures."""

from typing import Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Schema tracking structured execution outcomes returned from AI agent invocations."""

    agent_name: str = Field(
        ...,
        description="Identified name of the executing agent",
        examples=["DataCleanerAgent"]
    )
    task_name: str = Field(
        ...,
        description="Name of the objective description executed by the agent",
        examples=["impute_missing_values"]
    )
    status: str = Field(
        ...,
        description="Execution status outcome (e.g. success, failed)",
        examples=["success"]
    )
    output_content: str = Field(
        ...,
        description="Primary textual output or summary explanation returned by the agent",
        examples=["Imputed 12 missing age rows using group median strategies."]
    )
    structured_data: dict[str, Any] | None = Field(
        default=None,
        description="Key-value mapping of any raw data outputs generated during execution",
        examples=[{"imputed_count": 12, "strategy": "median"}]
    )
    thoughts: list[str] | None = Field(
        default=None,
        description="Chain-of-thought steps, reflections, or decision actions logged by the agent",
        examples=[["Detected 12 nulls in column 'age'.", "Inferred 'age' column data-type is float.", "Determined median age is 38.5."]]
    )
    execution_time_seconds: float = Field(
        ...,
        description="Time taken to run the task in seconds",
        examples=[2.41]
    )
