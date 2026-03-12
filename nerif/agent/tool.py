from typing import Any, Callable, Dict


class Tool:
    """Base class for tools that can be used by NerifAgent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

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


def tool(name: str, description: str, parameters: Dict[str, Any]):
    """Decorator to create a Tool from a function."""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(name=name, description=description, parameters=parameters, func=func)

    return decorator
