import os as _os

_os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "Agent":
        from .core.agent import Agent as _Agent

        return _Agent
    if name == "function_tool":
        from .tools.function_tool import function_tool as _ft

        return _ft
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
