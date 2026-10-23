"""Chart construction from a DataFrame + a small, validated spec."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .errors import ChartError

CHART_TYPES = ["bar", "line", "pie", "scatter", "histogram", "box", "area", "heatmap"]


def _check(df: pd.DataFrame, *cols: str | None) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ChartError(f"Column(s) {missing} not found in the query result. Available: {list(df.columns)}")


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly figure. Raises :class:`ChartError` on invalid specs."""
    if df.empty:
        raise ChartError("The query returned no rows, so there is nothing to chart.")
    chart_type = (chart_type or "").lower()
    if chart_type not in CHART_TYPES:
        raise ChartError(f"Unsupported chart type '{chart_type}'. Choose from {CHART_TYPES}.")
    # sensible defaults: first column = x, second = y
    cols = list(df.columns)
    if x is None and chart_type != "histogram":
        x = cols[0]
    if y is None and chart_type in {"bar", "line", "pie", "scatter", "area"} and len(cols) > 1:
        y = cols[1]
    _check(df, x, y, color)

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title, barmode="group")
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)
        elif chart_type == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title, hole=0.35)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x or y, color=color, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title)
        else:  # heatmap
            fig = px.density_heatmap(df, x=x, y=y, title=title)
    except Exception as exc:  # plotly raises ValueError/TypeError for bad dtypes
        raise ChartError(f"Could not draw the {chart_type} chart: {exc}") from exc

    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig


def build_forecast_chart(fc: pd.DataFrame, title: str = "Forecast") -> go.Figure:
    """Actuals + forecast line with a shaded 95% band."""
    fig = go.Figure()
    fut = fc[fc["forecast"].notna()]
    fig.add_trace(go.Scatter(x=fc["period"], y=fc["actual"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fut["period"], y=fut["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut["period"], y=fut["lower"], fill="tonexty", line=dict(width=0),
                             name="95% band", fillcolor="rgba(99,110,250,0.18)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut["period"], y=fut["forecast"], name="Forecast", mode="lines+markers",
                             line=dict(dash="dash")))
    fig.update_layout(title=title, template="plotly_white", height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig
