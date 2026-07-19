"""Unit tests for LLMClient connection wraps."""

from src.core.llm.client import LLMClient
from src.core.llm.config import LLMConfig
from src.core.llm.cost_tracker import CostTracker


def test_llm_client_mock():
    """Test generating text utilizing the mock connection handler."""
    config = LLMConfig(provider="mock", model="gpt-4o")
    client = LLMClient(config=config)

    # Generate summary
    resp = client.generate(
        "Provide an executive summary of findings with headline and key_takeaways.",
        bypass_cache=True,
    )
    assert "headline" in resp

    # Assert cost tracked successfully
    summary = CostTracker.get_session_summary()
    assert summary["session_input_tokens"] > 0
    assert summary["session_cost"] >= 0.0
