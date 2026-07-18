"""Unit tests for StructuredParser JSON cleanups."""

from pydantic import BaseModel
from src.core.llm.parser import StructuredParser


class DummyModel(BaseModel):
    name: str
    value: int


def test_structured_parser_clean_json():
    """Test parsing clean JSON strings to Pydantic models."""
    json_str = '{"name": "test_object", "value": 100}'
    model = StructuredParser.parse_response(json_str, DummyModel)
    
    assert model.name == "test_object"
    assert model.value == 100


def test_structured_parser_markdown_wrap():
    """Test parsing Markdown wrapped json blocks to models."""
    markdown_str = """
    ```json
    {
        "name": "wrapped_object",
        "value": 42
    }
    ```
    """
    model = StructuredParser.parse_response(markdown_str, DummyModel)
    
    assert model.name == "wrapped_object"
    assert model.value == 42
