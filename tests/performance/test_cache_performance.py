"""Performance tests profiling LLM Client cache hits efficiency and generation speedups."""

import time

from src.core.llm.client import LLMClient
from src.core.llm.config import LLMConfig


def test_llm_cache_speedups():
    """Verify that cached completions return in microsecond bounds, avoiding external API mock requests."""
    config = LLMConfig(provider="mock")
    client = LLMClient(config)

    prompt = "Create a summary of target variables classifications."

    # 1. First run: uncached generation
    start_time = time.time()
    res1 = client.generate(prompt, bypass_cache=True)
    uncached_duration = time.time() - start_time

    # 2. Second run: cached retrieval (using cache)
    # Prime cache
    client.generate(prompt, bypass_cache=False)

    start_time = time.time()
    res2 = client.generate(prompt, bypass_cache=False)
    cached_duration = time.time() - start_time

    assert res1 == res2

    # Assert cached query is significantly faster than uncached mock call
    assert cached_duration < uncached_duration
    # Cached duration should be near-instant (under 50ms for CI runner stability)
    assert cached_duration < 0.05
