"""Offline "demo" LLM: a rule-based planner that speaks the same protocol as a real model.

It lets reviewers try the full pipeline (tools, charts, anomalies, SQL, context) with no
API key. It is intentionally simple and is NOT a substitute for the real model: it routes by
keywords and picks columns heuristically.
"""
from __future__ import annotations

import re
import uuid

import pandas as pd

from .llm import LLMResponse
from .workspace import Workspace

_MEASURES = ("revenue", "sales", "amount", "total", "profit", "price", "quantity", "value")
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}


class DemoLLM:
    name = "offline-demo"

    def __init__(self, workspace: Workspace):
        self.ws = workspace
        self._last_sql: str | None = None

    # -- helpers ----------------------------------------------------------
    def _fact_table(self) -> str | None:
        """The table with the most measure-like columns (usually the transactions table)."""
        return max(self.ws.datasets, default=None,
                   key=lambda t: sum(any(m in c for m in _MEASURES) for c in self.ws.datasets[t].df.columns))

    def _named_table(self, q: str) -> str | None:
        return next((t for t in self.ws.datasets if re.search(rf"\b{re.escape(t)}\b", q)), None)

    def _customer_ranking(self, fact: str, measure: str, n: int, worst: bool) -> str | None:
        cust = self.ws.datasets.get("customers")
        if cust is None or "customer_id" not in self.ws.datasets[fact].df.columns:
            return None
        label = "customer_name" if "customer_name" in cust.df.columns else "customer_id"
        return (f"SELECT c.{label}, c.customer_id, ROUND(SUM(s.{measure}), 2) AS total_{measure} "
                f"FROM {fact} s JOIN customers c USING (customer_id) GROUP BY 1, 2 "
                f"ORDER BY 3 {'ASC' if worst else 'DESC'} LIMIT {n}")

    def _measure(self, df: pd.DataFrame, q: str) -> str | None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("id")]
        for c in nums:
            if c in q or c.rstrip("s") in q:
                return c
        for m in _MEASURES:
            for c in nums:
                if m in c:
                    return c
        return nums[0] if nums else None

    def _dimension(self, df: pd.DataFrame, q: str) -> str | None:
        cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].nunique() < 0.5 * len(df)]
        singular = lambda w: w[:-1] if w.endswith("s") else w  # noqa: E731
        for c in cats:
            if c in q or singular(c) in q or any(singular(w) == c for w in re.findall(r"[a-z_]+", q)):
                return c
        for pref in ("region", "product", "category", "segment", "country"):
            if pref in cats:
                return pref
        return cats[0] if cats else None

    def _date(self, df: pd.DataFrame) -> str | None:
        return next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)

    def _top_n(self, q: str) -> int:
        m = re.search(r"top\s+(\d+|\w+)", q)
        if m:
            tok = m.group(1)
            return int(tok) if tok.isdigit() else _NUMBER_WORDS.get(tok, 5)
        return 5

    def _tool(self, name: str, args: dict, text: str = "") -> LLMResponse:
        blocks = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}", "name": name, "input": args}]
        return LLMResponse(blocks, "tool_use")

    @staticmethod
    def _final(text: str) -> LLMResponse:
        return LLMResponse([{"type": "text", "text": text}], "end_turn")

    # -- protocol ---------------------------------------------------------
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):   # we just got tool results
            return self._summarise(last["content"][0])
        q = last["content"].lower()
        fact = self._fact_table()
        table = fact
        if re.search(r"anomal|outlier|quality|missing|describe|schema|columns", q):
            table = self._named_table(q) or fact
        if not table:
            return self._final("Please upload a CSV file first, then ask me a question about it.")
        df = self.ws.datasets[table].df
        measure, dim, date = self._measure(df, q), self._dimension(df, q), self._date(df)
        self._ctx = {"q": q, "measure": measure, "dim": dim, "worst": False}

        if re.search(r"anomal|outlier|unusual|suspicious", q):
            return self._tool("detect_anomalies", {"table": table})
        if re.search(r"quality|missing|clean|duplicate", q):
            return self._tool("data_quality", {"table": table})
        if re.search(r"forecast|predict|next \w+ months", q) and date and measure:
            sql = (f"SELECT date_trunc('month', {date}) AS month, SUM({measure}) AS {measure} "
                   f"FROM {table} GROUP BY 1 ORDER BY 1")
            self._ctx["dim"] = None
            return self._tool("forecast", {"sql": sql, "date_column": "month", "value_column": measure})
        if re.search(r"\bsql\b|query", q) and self._last_sql:
            return self._final(f"Here is the SQL from my last analysis:\n\n```sql\n{self._last_sql}\n```\n\n"
                               "**How I got this**\n- Reused the most recent query in this session.")
        if re.search(r"describe|what data|columns|schema|overview|summar", q) and "revenue" not in q:
            return self._tool("describe_data", {"table": table})
        if re.search(r"trend|monthly|over time|by month|per month", q) and date and measure:
            sql = (f"SELECT date_trunc('month', {date}) AS month, SUM({measure}) AS {measure} "
                   f"FROM {table} GROUP BY 1 ORDER BY 1")
            self._last_sql = sql
            self._ctx["dim"] = None
            return self._tool("create_chart", {"sql": sql, "chart_type": "line", "x": "month", "y": measure,
                                               "title": f"Monthly {measure}"})
        if "customer" in q and fact and measure:
            worst = bool(re.search(r"under ?perform|lowest|worst|bottom|least", q))
            sql = self._customer_ranking(fact, self._measure(self.ws.datasets[fact].df, q), self._top_n(q), worst)
            if sql:
                self._last_sql = sql
                self._ctx["dim"] = "customer"
                return self._tool("run_sql", {"sql": sql, "purpose": "Rank customers by total revenue (join sales to customers)."})
        if dim and measure:
            worst = bool(re.search(r"under ?perform|lowest|worst|bottom|least", q))
            n = self._top_n(q) if ("top" in q or worst) else 10
            order = "ASC" if worst else "DESC"
            self._ctx["worst"] = worst
            sql = (f"SELECT {dim}, ROUND(SUM({measure}), 2) AS total_{measure} FROM {table} "
                   f"GROUP BY 1 ORDER BY 2 {order} LIMIT {n}")
            self._last_sql = sql
            if re.search(r"chart|plot|show|graph|pie|bar", q) or worst:
                kind = "pie" if "pie" in q else "bar"
                return self._tool("create_chart", {"sql": sql, "chart_type": kind, "x": dim,
                                                   "y": f"total_{measure}",
                                                   "title": f"{'Lowest' if worst else 'Top'} {dim} by {measure}"})
            return self._tool("run_sql", {"sql": sql, "purpose": f"Rank {dim} by total {measure}."})
        return self._tool("describe_data", {"table": table},
                          "I'm the offline demo model, so I can only handle simple questions.")

    def _summarise(self, result: dict) -> LLMResponse:
        content = str(result.get("content", ""))
        if result.get("is_error"):
            return self._final(f"The tool reported a problem:\n\n> {content}")
        ctx = getattr(self, "_ctx", {})
        lines = [ln for ln in content.splitlines() if ln and not ln.startswith("[")]
        head = "\n".join(f"    {ln}" for ln in lines[:8])
        if content.startswith(("Flagged", "No anomalies")):
            return self._final(f"{lines[0]}\n\nMost notable findings are listed in the table below.\n\n"
                               "**How I got this**\n- Robust modified z-scores (median/MAD) per numeric column, "
                               "plus duplicate-row and unexpected-negative checks; each row carries its own explanation.")
        if content.startswith("Data quality"):
            return self._final(f"{lines[0]}\n\n**How I got this**\n- Scanned every column for missing values, "
                               "duplicates, constants and inconsistent category spellings.")
        if content.startswith("Linear-trend"):
            return self._final("Forecast for the coming months (trend + seasonality, with a 95% band):\n\n"
                               + "\n".join(f"    {ln}" for ln in lines[1:5])
                               + "\n\n**How I got this**\n- Aggregated to monthly totals, fitted a linear trend, "
                                 "added month-of-year seasonality when 24+ months exist; the band is ±1.96 residual std.")
        data = lines[2:] if content.startswith("Chart") else lines[1:]
        top = data[0].split(",") if data else []
        lead = ""
        if ctx.get("dim") and top:
            word = "lowest" if ctx.get("worst") else "highest"
            lead = f"The {word} is **{top[0]}** with **{top[-1]}**.\n\n"
        return self._final(
            f"{lead}Preview of the result:\n\n{head}\n\n**How I got this**\n"
            f"- Ran an aggregation over the loaded table(s) using column `{ctx.get('measure')}`; "
            "open the SQL panel for the exact query.\n"
            "- *(Offline demo mode: add an Anthropic API key for richer insights and follow-up reasoning.)*")
