"""Streamlit rendering helpers for agent artifacts."""
from __future__ import annotations

import streamlit as st

from .charts import build_forecast_chart
from .models import AgentTurn, Artifact
from .quality import QualityReport


def render_turn(turn: AgentTurn, key: str) -> None:
    """Render one assistant answer with its charts, tables and reasoning trail."""
    st.markdown(turn.answer)

    charts = [a for a in turn.artifacts if a.kind == "chart"]
    forecasts = [a for a in turn.artifacts if a.kind == "forecast"]
    for i, a in enumerate(charts):
        st.plotly_chart(a.payload, width="stretch", key=f"{key}-chart-{i}")
    for i, a in enumerate(forecasts):
        st.plotly_chart(build_forecast_chart(a.payload, a.title), width="stretch", key=f"{key}-fc-{i}")

    for a in turn.artifacts:
        if a.kind == "anomalies":
            st.subheader("Flagged anomalies")
            if a.payload.empty:
                st.success("No anomalies found.")
            else:
                st.dataframe(a.payload, width="stretch", hide_index=True,
                             column_config={"reason": st.column_config.TextColumn("Why it was flagged", width="large")})
        elif a.kind == "quality":
            render_quality(a.payload)

    tables = [a for a in turn.artifacts if a.kind == "table"]
    if tables:
        with st.expander("Result table", expanded=not (charts or forecasts)):
            for a in tables:
                st.caption(a.title)
                st.dataframe(a.payload, width="stretch", hide_index=True)

    code = [a for a in turn.artifacts if a.kind == "code"]
    if code or turn.tool_calls:
        with st.expander("Reasoning trail: queries & agent steps"):
            for a in code:
                st.caption(a.title)
                st.code(str(a.payload), language=a.language or "text")
            if turn.tool_calls:
                st.caption("Agent steps")
                st.table([{"tool": c.name, "status": "ok" if c.ok else "error", "ms": round(c.elapsed_ms)}
                          for c in turn.tool_calls])
            if turn.input_tokens:
                st.caption(f"Tokens: {turn.input_tokens:,} in / {turn.output_tokens:,} out")


def render_quality(rep: QualityReport) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality score", f"{rep.score}/100")
    c2.metric("Rows", f"{rep.rows:,}")
    c3.metric("Columns", rep.columns)
    if rep.issues:
        for issue in rep.issues:
            st.warning(issue)
    else:
        st.success("No issues found.")
    with st.expander("Column profile"):
        st.dataframe(rep.column_profile, width="stretch", hide_index=True)


EXAMPLES = [
    "Which region generated the highest revenue?",
    "Show monthly sales trends.",
    "Which products are underperforming?",
    "What are the top five customers?",
    "Detect anomalies in the dataset.",
    "Generate SQL for this analysis.",
]
