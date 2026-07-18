"""Tests for LangGraph topology construction."""

from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph


def test_graph_compiles_with_injected_callbacks():
    """Graph construction does not require real external agents."""
    graph = WorkflowGraph(config=WorkflowConfig(checkpoint_mode="none", persist_execution=False), callbacks={})
    assert graph.graph is not None
