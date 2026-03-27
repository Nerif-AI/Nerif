from nerif.observability.recorder import ExecutionRecording, IterationSnapshot


def test_iteration_snapshot_serialization():
    snap = IterationSnapshot(
        iteration=1, span_id="span123",
        agent_state={"model": "gpt-4o", "system_prompt": "test", "temperature": 0.0,
                      "max_tokens": None, "max_iterations": 10},
        input_message="hello", output="world",
        tool_calls=[], token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        timestamp=1000.0,
    )
    d = snap.to_dict()
    restored = IterationSnapshot.from_dict(d)
    assert restored.iteration == 1
    assert restored.input_message == "hello"
    assert restored.output == "world"


def test_execution_recording_serialization():
    snap = IterationSnapshot(
        iteration=1, span_id="s1",
        agent_state={"model": "gpt-4o", "system_prompt": "hi", "temperature": 0.0,
                      "max_tokens": None, "max_iterations": 10},
        input_message="hi", output="hello",
        tool_calls=[], token_usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        timestamp=100.0,
    )
    recording = ExecutionRecording(
        trace_id="trace1", agent_name="test_agent", model="gpt-4o",
        snapshots=[snap],
        final_result={"content": "hello", "tool_calls": [], "token_usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, "latency_ms": 100.0, "iterations": 1, "model": "gpt-4o"},
        metadata={},
    )
    d = recording.to_dict()
    restored = ExecutionRecording.from_dict(d)
    assert restored.trace_id == "trace1"
    assert restored.agent_name == "test_agent"
    assert len(restored.snapshots) == 1
    assert restored.snapshots[0].output == "hello"
