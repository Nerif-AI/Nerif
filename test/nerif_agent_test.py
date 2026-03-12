"""Tests for the agent module."""



from nerif.agent import NerifAgent, Tool, tool


class TestTool:
    def test_tool_creation(self):
        t = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            func=lambda a, b: a + b,
        )
        assert t.name == "add"
        assert t.execute(a=2, b=3) == 5

    def test_tool_to_openai_format(self):
        t = Tool(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )
        result = t.to_openai_tool()
        assert result["type"] == "function"
        assert result["function"]["name"] == "test"

    def test_tool_decorator(self):
        @tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
            },
        )
        def multiply(x, y):
            return x * y

        assert isinstance(multiply, Tool)
        assert multiply.name == "multiply"
        assert multiply.execute(x=3, y=4) == 12


class TestNerifAgent:
    def test_register_tool(self):
        agent = NerifAgent(model="gpt-4o")
        t = Tool(
            name="test",
            description="Test",
            parameters={"type": "object", "properties": {}},
            func=lambda: "ok",
        )
        agent.register_tool(t)
        assert "test" in agent.tools

    def test_register_tools(self):
        agent = NerifAgent(model="gpt-4o")
        tools = [
            Tool(name="a", description="A", parameters={}, func=lambda: None),
            Tool(name="b", description="B", parameters={}, func=lambda: None),
        ]
        agent.register_tools(tools)
        assert len(agent.tools) == 2

    def test_reset(self):
        agent = NerifAgent(model="gpt-4o")
        agent.model.messages.append({"role": "user", "content": "hi"})
        agent.reset()
        assert len(agent.model.messages) == 1  # only system prompt

    def test_execute_tool_call_missing_tool(self):
        from nerif.model.model import ToolCallResult

        agent = NerifAgent(model="gpt-4o")
        tc = ToolCallResult(id="1", name="nonexistent", arguments="{}")
        result = agent._execute_tool_call(tc)
        assert "not found" in result

    def test_execute_tool_call_invalid_json(self):
        from nerif.model.model import ToolCallResult

        agent = NerifAgent(model="gpt-4o")
        t = Tool(name="test", description="Test", parameters={}, func=lambda: "ok")
        agent.register_tool(t)
        tc = ToolCallResult(id="1", name="test", arguments="invalid json")
        result = agent._execute_tool_call(tc)
        assert "Invalid JSON" in result

    def test_execute_tool_call_success(self):
        from nerif.model.model import ToolCallResult

        agent = NerifAgent(model="gpt-4o")
        t = Tool(
            name="greet",
            description="Greet",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
            func=lambda name: f"Hello {name}!",
        )
        agent.register_tool(t)
        tc = ToolCallResult(id="1", name="greet", arguments='{"name": "World"}')
        result = agent._execute_tool_call(tc)
        assert result == "Hello World!"
