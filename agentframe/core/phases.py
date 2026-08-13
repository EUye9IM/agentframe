from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    """State-machine node names. Members are `str` subclasses, so they satisfy
    LangGraph's `isinstance(x, str)` checks and can be passed directly."""

    LLM = "LLM"
    TOOLS = "TOOLS"
    END = "__end__"
