"""Structured response parser validating and deserializing JSON outputs to Pydantic models."""

import json
from typing import Type, TypeVar, Any
from pydantic import BaseModel

from src.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredParser:
    """Deserializes raw JSON strings to Pydantic schemas, correcting formatting errors."""

    @staticmethod
    def parse_response(response_text: str, model_class: Type[T]) -> T:
        """Parse text string to model class. Handles Markdown code blocks and trailing errors.

        Args:
            response_text: Raw string response from LLM.
            model_class: Target Pydantic model class.

        Returns:
            T: Parsed Pydantic model instance.
        """
        logger.info(f"StructuredParser: Parsing response to model: {model_class.__name__}")
        clean_text = response_text.strip()

        # 1. Clean Markdown code blocks wrap (e.g. ```json ... ```)
        if "```" in clean_text:
            parts = clean_text.split("```")
            for part in parts:
                part_clean = part.strip()
                if part_clean.startswith("json"):
                    part_clean = part_clean[4:].strip()
                if part_clean.startswith("{") and part_clean.endswith("}"):
                    clean_text = part_clean
                    break

        # 2. Attempt parsing JSON
        try:
            parsed_json = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"StructuredParser: Failed to decode JSON content: {e}. Raw: {clean_text}")
            raise ValueError(f"Malformed JSON content returned: {str(e)}") from e

        # 3. Validate Pydantic model schema
        try:
            return model_class.model_validate(parsed_json)
        except Exception as e:
            logger.error(f"StructuredParser: Schema validation failed for model '{model_class.__name__}': {e}")
            raise ValueError(f"Pydantic schema validation mismatch: {str(e)}") from e
