"""LLM tool definitions and their executor.

Design choice: heavy data never flows through the LLM. Tools run against the
data locally, return a *small preview* to the model, and attach the *full*
result as UI artifacts. The model reasons over previews; the user sees
everything.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from .anomaly import detect_anomalies
from .charts import CHART_TYPES, build_chart
from .errors import AnalystError
from .forecast import forecast_series
from .models import Artifact, ToolCall, ToolResult
from .pandas_sandbox import run_pandas
from .quality import assess_quality
from .workspace import Workspace

log = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "describe_data",
        "description": "Describe loaded tables: columns, types, ranges and summary statistics. "
                       "Use for questions about what data is available.",
        "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}},
    },
    {
        "name": "run_sql",
        "description": "Run ONE read-only DuckDB SQL SELECT over the loaded tables (joins across tables are "
                       "allowed). Use for aggregations, rankings, filters and trends. Returns a preview; the "
                       "user sees the full table and the SQL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement (DuckDB dialect)."},
                "purpose": {"type": "string", "description": "One sentence: why this query answers the question."},
            },
            "required": ["sql", "purpose"],
        },
    },
    {
        "name": "run_pandas",
        "description": "Run Python (pandas as `pd`, numpy as `np`; each table is a DataFrame variable named "
                       "after the table) when SQL is awkward (pivots, rolling windows, correlations). No imports. "
                       "Assign the final answer to `result`.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}, "purpose": {"type": "string"}},
            "required": ["code", "purpose"],
        },
    },
    {
        "name": "create_chart",
        "description": "Create a chart. Provide the SQL that produces the plotted data, then map columns of that "
                       "result to axes. Aggregate in SQL first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "chart_type": {"type": "string", "enum": CHART_TYPES},
                "x": {"type": "string", "description": "Column for x axis (or pie labels)."},
                "y": {"type": "string", "description": "Column for y axis (or pie values)."},
                "color": {"type": "string", "description": "Optional grouping column."},
                "title": {"type": "string"},
            },
            "required": ["sql", "chart_type", "title"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Detect and explain anomalies (robust z-score outliers, unexpected negatives, duplicate "
                       "rows) in a table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Optional subset."},
            },
            "required": ["table"],
        },
    },
    {
        "name": "data_quality",
        "description": "Profile data quality: missing values, duplicates, constants, inconsistent categories.",
        "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
    },
    {
        "name": "forecast",
        "description": "Forecast a metric over the next months (linear trend + seasonality). Provide SQL that "
                       "returns a date column and a numeric column.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "date_column": {"type": "string"},
                "value_column": {"type": "string"},
                "periods": {"type": "integer", "description": "Months ahead (default 6)."},
            },
            "required": ["sql", "date_column", "value_column"],
        },
    },
]


def _preview(df: pd.DataFrame, n: int, truncated: bool = False) -> str:
    text = df.head(n).to_csv(index=False)
    note = f"\n[{len(df)} row(s){'+' if truncated else ''} total; showing first {min(n, len(df))}]"
    return text + note


class ToolExecutor:
    def __init__(self, workspace: Workspace):
        self.ws = workspace
        self.n = workspace.settings.llm_preview_rows
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "describe_data": self._describe,
            "run_sql": self._sql,
            "run_pandas": self._pandas,
            "create_chart": self._chart,
            "detect_anomalies": self._anomalies,
            "data_quality": self._quality,
            "forecast": self._forecast,
        }

    def execute(self, name: str, args: dict) -> tuple[ToolResult, ToolCall]:
        start = time.perf_counter()
        handler = self._handlers.get(name)
        try:
            if handler is None:
                raise AnalystError(f"Unknown tool '{name}'.")
            result = handler(**args)
        except AnalystError as exc:
            result = ToolResult(f"ERROR: {exc}", is_error=True)
        except TypeError as exc:   # bad/missing arguments from the model
            result = ToolResult(f"ERROR: invalid arguments for {name}: {exc}", is_error=True)
        except Exception as exc:   # noqa: BLE001 - never crash the chat on a tool bug
            log.exception("tool %s crashed", name)
            result = ToolResult(f"ERROR: unexpected failure in {name}: {exc}", is_error=True)
        elapsed = (time.perf_counter() - start) * 1000
        log.info("tool=%s ok=%s %.0fms", name, not result.is_error, elapsed)
        return result, ToolCall(name, args, not result.is_error, elapsed, result.content[:120])

    # -- handlers ---------------------------------------------------------
    def _tables(self, table: str | None) -> dict[str, pd.DataFrame]:
        frames = self.ws.frames()
        if not frames:
            raise AnalystError("No datasets are loaded. Ask the user to upload a CSV first.")
        if table:
            if table not in frames:
                raise AnalystError(f"Unknown table '{table}'. Available: {list(frames)}")
            return {table: frames[table]}
        return frames

    def _describe(self, table: str | None = None) -> ToolResult:
        frames = self._tables(table)
        parts, arts = [], []
        for name, df in frames.items():
            desc = df.describe(include="all").T.round(2)
            parts.append(self.ws.datasets[name].schema_text() + "\nstats:\n" + desc.head(30).to_csv())
            arts.append(Artifact("table", f"Summary statistics: {name}", desc.reset_index(names="column")))
        return ToolResult("\n\n".join(parts), arts)

    def _sql(self, sql: str, purpose: str = "") -> ToolResult:
        res = self.ws.engine.execute(sql)
        arts = [Artifact("code", purpose or "SQL", res.sql, "sql"),
                Artifact("table", "Result" + (" (truncated)" if res.truncated else ""), res.df)]
        return ToolResult(_preview(res.df, self.n, res.truncated), arts)

    def _pandas(self, code: str, purpose: str = "") -> ToolResult:
        res = run_pandas(code, self.ws.frames(), self.ws.settings.query_timeout_s, self.ws.settings.max_result_rows)
        arts = [Artifact("code", purpose or "Pandas", code, "python"), Artifact("table", "Result", res.df)]
        return ToolResult(_preview(res.df, self.n), arts)

    def _chart(self, sql: str, chart_type: str, title: str, x: str | None = None, y: str | None = None,
               color: str | None = None) -> ToolResult:
        res = self.ws.engine.execute(sql)
        fig = build_chart(res.df, chart_type, x, y, color, title)
        arts = [Artifact("code", f"SQL behind '{title}'", res.sql, "sql"), Artifact("chart", title, fig)]
        return ToolResult(f"Chart '{title}' ({chart_type}) rendered from {len(res.df)} rows.\n"
                          + _preview(res.df, 8, res.truncated), arts)

    def _anomalies(self, table: str, columns: list[str] | None = None) -> ToolResult:
        df = self._tables(table)[table]
        rep = detect_anomalies(df, columns)
        arts = [Artifact("anomalies", "Flagged anomalies", rep.table)]
        body = rep.summary
        if rep.count:
            body += "\nTop findings:\n" + "\n".join(
                f"- row {r.row} {r.id}: {r.reason}" for r in rep.table.head(10).itertuples())
        return ToolResult(body, arts)

    def _quality(self, table: str) -> ToolResult:
        df = self._tables(table)[table]
        rep = assess_quality(df, table)
        return ToolResult(rep.to_text(), [Artifact("quality", f"Data quality: {table}", rep)])

    def _forecast(self, sql: str, date_column: str, value_column: str, periods: int = 6) -> ToolResult:
        res = self.ws.engine.execute(sql)
        fc = forecast_series(res.df, date_column, value_column, periods)
        future = fc[fc["forecast"].notna()][["period", "forecast", "lower", "upper"]].copy()
        future[["forecast", "lower", "upper"]] = future[["forecast", "lower", "upper"]].round(0)
        txt = ("Linear-trend + seasonal forecast (95% band from residual std). Next periods:\n"
               + future.to_csv(index=False))
        return ToolResult(txt, [Artifact("code", "SQL behind forecast", res.sql, "sql"),
                                Artifact("forecast", f"Forecast: {value_column}", fc)])
