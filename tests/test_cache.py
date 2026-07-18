"""Unit tests for the local memory-and-file serialization cache provider."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.cache import OptimizationCache, compress_prompt_context


def test_cache_set_and_get() -> None:
    """Test standard object serialization caching works."""
    cache_key = "test_object_key_123"
    test_value = {"nested_list": [1, 2, 3], "status": "active"}
    
    # Assert missing cache returns None
    assert OptimizationCache.get(cache_key) is None
    
    # Save cache
    OptimizationCache.set(cache_key, test_value)
    
    # Fetch cache
    fetched = OptimizationCache.get(cache_key)
    assert fetched == test_value
    
    # Delete cache entry
    OptimizationCache.delete(cache_key)
    assert OptimizationCache.get(cache_key) is None


def test_prompt_response_caching() -> None:
    """Test text prompt caching."""
    prompt = "Predict monthly charges for customer C1."
    response = "The predicted monthly charges is 72.50 USD."
    
    assert OptimizationCache.get_prompt_response(prompt) is None
    
    OptimizationCache.set_prompt_response(prompt, response)
    
    assert OptimizationCache.get_prompt_response(prompt) == response
    
    # Clear cache
    OptimizationCache.clear()
    assert OptimizationCache.get_prompt_response(prompt) is None


def test_compress_prompt_context() -> None:
    """Test prompt whitespace and duplicate line reduction."""
    raw_prompt = """
    
    System Instructions:
    
    Analyze Churn
    
    
    Output results.
    
    """
    compressed = compress_prompt_context(raw_prompt)
    expected = "System Instructions:\nAnalyze Churn\nOutput results."
    assert compressed == expected
