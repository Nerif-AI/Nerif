"""Integration test: agent with tracing enabled, no real LLM calls."""

from unittest.mock import patch

from nerif.agent import NerifAgent
from nerif.observability.tracing import (
    TraceCollector,
    TracingCallbackHandler,
)
from nerif.utils.callbacks import CallbackManager


def test_agent_creates_span_when_tracing_active():
    collector = TraceCollector()
    handler = TracingCallbackHandler(collector)
    cb = CallbackManager()
    cb.add_handler(handler)

    agent = NerifAgent(model="gpt-4o", callbacks=cb)
    token = collector.activate()

    try:
        with patch.object(agent.model, "chat", return_value="mocked response"):
            result = agent.run("hello")

        assert result.content == "mocked response"
        assert collector.last_trace_id is not None
    finally:
        collector.deactivate(token)


def test_agent_works_without_tracing():
    """Verify agent still works when no tracing is active."""
    agent = NerifAgent(model="gpt-4o")
    with patch.object(agent.model, "chat", return_value="no trace"):
        result = agent.run("hello")
    assert result.content == "no trace"
