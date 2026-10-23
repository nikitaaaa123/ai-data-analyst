"""Export a conversation as a Markdown or self-contained HTML report."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from .models import AgentTurn


def _table_md(df: pd.DataFrame, n: int = 20) -> str:
    return df.head(n).to_markdown(index=False) if len(df) else "_(no rows)_"


def to_markdown(turns: list[AgentTurn], dataset_names: list[str]) -> str:
    out = ["# AI Data Analyst report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
           "**Datasets:** " + (", ".join(dataset_names) or "none"), ""]
    for i, t in enumerate(turns, 1):
        out += [f"## {i}. {t.question}", "", t.answer, ""]
        for a in t.artifacts:
            if a.kind == "code":
                out += [f"**{a.title}**", f"```{a.language or ''}", str(a.payload), "```", ""]
            elif a.kind in {"table", "anomalies"} and isinstance(a.payload, pd.DataFrame):
                out += [f"**{a.title}**", "", _table_md(a.payload), ""]
    return "\n".join(out)


def to_html(turns: list[AgentTurn], dataset_names: list[str]) -> str:
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>AI Data Analyst report</title>"
             "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2937}"
             "pre{background:#f3f4f6;padding:.75rem;border-radius:6px;overflow:auto}"
             "table{border-collapse:collapse;font-size:.85rem}td,th{border:1px solid #e5e7eb;padding:.25rem .5rem}"
             "h2{margin-top:2rem;border-top:1px solid #e5e7eb;padding-top:1rem}</style></head><body>",
             "<h1>AI Data Analyst report</h1>",
             f"<p><em>Generated {datetime.now():%Y-%m-%d %H:%M} · Datasets: "
             f"{html.escape(', '.join(dataset_names) or 'none')}</em></p>"]
    first_chart = True
    for i, t in enumerate(turns, 1):
        parts.append(f"<h2>{i}. {html.escape(t.question)}</h2>")
        parts.append("<div style='white-space:pre-wrap'>" + html.escape(t.answer) + "</div>")
        for a in t.artifacts:
            if a.kind == "code":
                parts.append(f"<h4>{html.escape(a.title)}</h4><pre>{html.escape(str(a.payload))}</pre>")
            elif a.kind in {"table", "anomalies"} and isinstance(a.payload, pd.DataFrame):
                parts.append(f"<h4>{html.escape(a.title)}</h4>" + a.payload.head(20).to_html(index=False, border=0))
            elif a.kind == "chart":
                parts.append(a.payload.to_html(full_html=False, include_plotlyjs="cdn" if first_chart else False))
                first_chart = False
    parts.append("</body></html>")
    return "".join(parts)
