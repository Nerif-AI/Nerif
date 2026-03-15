from typing import Any, Callable, Dict, Optional


class Tool:
    """Base class for tools that can be used by NerifAgent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
        async_func: Optional[Callable[..., Any]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.async_func = async_func

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)

    async def aexecute(self, **kwargs) -> Any:
        """Async tool execution. Uses async_func if available, else falls back to sync func."""
        if self.async_func is not None:
            return await self.async_func(**kwargs)
        return self.func(**kwargs)


def tool(name: str, description: str, parameters: Dict[str, Any], async_func: Optional[Callable] = None):
    """Decorator to create a Tool from a function."""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(name=name, description=description, parameters=parameters, func=func, async_func=async_func)

    return decorator
