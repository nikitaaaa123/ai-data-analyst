"""Shared data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    """Something the UI can render next to the assistant's text."""

    kind: str                 # table | chart | code | anomalies | quality | forecast
    title: str
    payload: Any              # DataFrame, plotly Figure, str...
    language: str | None = None   # for kind == "code"


@dataclass
class ToolResult:
    content: str              # what the LLM sees
    artifacts: list[Artifact] = field(default_factory=list)
    is_error: bool = False


@dataclass
class ToolCall:
    name: str
    input: dict
    ok: bool
    elapsed_ms: float
    summary: str = ""


@dataclass
class AgentTurn:
    question: str
    answer: str
    artifacts: list[Artifact] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
