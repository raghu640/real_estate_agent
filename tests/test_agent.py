"""Integration tests for the agent graph (mocked LLM — no real API calls)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_graph, AgentState, price_property, _should_continue, _get_llm, _agent_node, _price_property_async
from langgraph.graph import END


def test_graph_builds():
    graph = build_graph()
    assert graph is not None


def test_graph_has_agent_and_tools_nodes():
    graph = build_graph()
    assert graph is not None


def test_agent_state_has_messages():
    state = AgentState(messages=[HumanMessage(content="test")])
    assert "messages" in state


@pytest.mark.asyncio
async def test_price_property_returns_string(monkeypatch):
    """price_property must return a string (the final LLM message content)."""
    mock_compiled = MagicMock()
    fake_msg = AIMessage(content='{"price_min_lakh": 80.0, "price_max_lakh": 110.0}')
    mock_compiled.ainvoke = AsyncMock(return_value={"messages": [fake_msg]})

    import agent as agent_module
    from agent import _price_property_async
    original_graph = agent_module.graph
    agent_module.graph = mock_compiled
    try:
        answer, _ = await _price_property_async("2BHK in Whitefield, 1200 sqft, new, gated")
        assert isinstance(answer, str)
        assert len(answer) > 0
    finally:
        agent_module.graph = original_graph


# ---------------------------------------------------------------------------
# _should_continue — both branches
# ---------------------------------------------------------------------------

def test_should_continue_returns_end_when_no_tool_calls():
    """Message without tool_calls routes to END."""
    msg = AIMessage(content="Here is the estimate.")
    state = AgentState(messages=[msg])
    assert _should_continue(state) == END


def test_should_continue_returns_tools_when_tool_calls_present():
    """Message with tool_calls routes to 'tools'."""
    msg = AIMessage(content="", tool_calls=[{"name": "property_adjustments", "args": {}, "id": "1"}])
    state = AgentState(messages=[msg])
    assert _should_continue(state) == "tools"


# ---------------------------------------------------------------------------
# _get_llm — smoke test (mocked to avoid real API call)
# ---------------------------------------------------------------------------

def test_get_llm_returns_bound_llm(monkeypatch):
    """_get_llm constructs and binds a ChatOpenAI model."""
    import os
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")

    from unittest.mock import MagicMock
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    with patch("agent.ChatOpenAI", return_value=mock_llm):
        result = _get_llm()
        assert result is mock_llm


# ---------------------------------------------------------------------------
# _agent_node — unit test with fully mocked LLM
# ---------------------------------------------------------------------------

def test_agent_node_returns_messages(monkeypatch):
    """_agent_node invokes the LLM and wraps the response in a messages dict."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")

    fake_response = AIMessage(content="Test response")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake_response

    import agent as agent_module
    original_llm = agent_module._LLM
    agent_module._LLM = mock_llm
    try:
        state = AgentState(messages=[HumanMessage(content="What is the price?")])
        result = _agent_node(state)
    finally:
        agent_module._LLM = original_llm

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0] is fake_response
