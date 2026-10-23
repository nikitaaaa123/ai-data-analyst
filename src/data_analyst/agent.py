"""The analyst agent: a tool-calling loop with conversation memory."""
from __future__ import annotations

import logging

from .config import Settings
from .errors import LLMError
from .llm import LLMClient
from .models import AgentTurn
from .tools import TOOL_SCHEMAS, ToolExecutor
from .workspace import Workspace

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert data analyst embedded in a web app. Users upload CSV files and ask questions \
in natural language. You answer by calling tools that run against their data.

Rules
1. NEVER state a number you did not get from a tool result. If you need data, call run_sql (preferred) or run_pandas.
2. SQL dialect is DuckDB. Use the exact table and column names below; tables can be joined. Use \
date_trunc('month', col) for monthly buckets.
3. When the user asks to see/plot/chart/trend something, call create_chart (pick the most suitable type: bar for \
rankings, line for time, pie for shares of a whole with few categories, scatter for relationships).
4. For "anomalies/outliers/unusual" call detect_anomalies, then explain the most important findings in business terms.
5. For "data quality/clean/missing" call data_quality. For "forecast/predict" call forecast.
6. If a tool returns ERROR, read the message, fix your query and retry (at most twice) before telling the user.
7. Keep conversation context: resolve follow-ups like "now by month" or "only for 2025" using earlier turns.
8. If the question is ambiguous, state the assumption you made (e.g. "revenue = revenue column").
9. If the user asks for SQL/code, show the SQL you used in a fenced code block.

Answer format (markdown, concise):
- Lead with the direct answer and the key numbers.
- Add 1-3 short business insights when useful (what stands out, why it may matter).
- End with a section titled **How I got this** — 2-4 bullets explaining which tables/columns/filters/method you used \
and any caveats or assumptions.

Loaded data
{schema}
"""


class DataAnalystAgent:
    def __init__(self, llm: LLMClient, workspace: Workspace, settings: Settings | None = None):
        self.llm = llm
        self.ws = workspace
        self.settings = settings or workspace.settings
        self.executor = ToolExecutor(workspace)
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()

    def _system(self) -> str:
        return SYSTEM_PROMPT.format(schema=self.ws.schema_text())

    def _windowed(self) -> list[dict]:
        """Bound context size, always starting on a plain user message."""
        msgs = self.messages
        limit = self.settings.max_history_messages
        if len(msgs) <= limit:
            return msgs
        start = len(msgs) - limit
        while start < len(msgs) and not (msgs[start]["role"] == "user" and isinstance(msgs[start]["content"], str)):
            start += 1
        return msgs[start:] if start < len(msgs) else msgs[-1:]

    def ask(self, question: str) -> AgentTurn:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        turn = AgentTurn(question=question, answer="")
        try:
            for _ in range(self.settings.max_tool_iterations):
                resp = self.llm.create(self._system(), self._windowed(), TOOL_SCHEMAS)
                turn.input_tokens += resp.input_tokens
                turn.output_tokens += resp.output_tokens
                self.messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use" or not resp.tool_uses:
                    turn.answer = resp.text or "(The model returned no text.)"
                    return turn

                results = []
                for use in resp.tool_uses:
                    res, call = self.executor.execute(use["name"], use["input"] or {})
                    turn.tool_calls.append(call)
                    turn.artifacts.extend(res.artifacts)
                    results.append({"type": "tool_result", "tool_use_id": use["id"],
                                    "content": res.content, "is_error": res.is_error})
                self.messages.append({"role": "user", "content": results})

            turn.answer = ("I reached the step limit before finishing. Here is what I found so far — "
                           "try narrowing the question.")
            return turn
        except LLMError:
            del self.messages[checkpoint:]   # keep history valid for the next question
            raise
