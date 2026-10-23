"""Streamlit entry point:  streamlit run app.py"""
from __future__ import annotations

import hmac
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_analyst import AnalystSession, Settings  # noqa: E402
from data_analyst.anomaly import detect_anomalies  # noqa: E402
from data_analyst.config import setup_logging  # noqa: E402
from data_analyst.errors import AnalystError, LLMError  # noqa: E402
from data_analyst.models import AgentTurn, Artifact  # noqa: E402
from data_analyst.quality import assess_quality  # noqa: E402
from data_analyst.report import to_html, to_markdown  # noqa: E402
from data_analyst.ui import EXAMPLES, render_quality, render_turn  # noqa: E402

SAMPLE_DIR = Path(__file__).parent / "data"
log = logging.getLogger("app")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    s = Settings.from_env()
    setup_logging(s.log_level)
    return s


def _gate(settings: Settings) -> None:
    """Optional password protection (set APP_PASSWORD)."""
    if not settings.app_password or st.session_state.get("authed"):
        return
    st.title("📊 AI Data Analyst")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw, settings.app_password):
            st.session_state.authed = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def _session(settings: Settings) -> AnalystSession:
    if "session" not in st.session_state:
        st.session_state.session = AnalystSession(settings)
        st.session_state.seen_files = set()
    return st.session_state.session


def _sidebar(sess: AnalystSession) -> None:
    sb = st.sidebar
    sb.title("📊 AI Data Analyst")
    live = sess.mode.startswith("anthropic")
    sb.caption(("🟢 Live model: " if live else "🟠 Offline demo mode — ") + sess.mode)

    with sb.expander("🔑 Anthropic API key", expanded=not live and not sess.workspace.datasets):
        key = st.text_input("API key", type="password", placeholder="sk-ant-...",
                            help="Kept in this browser session only. Switching resets the conversation.")
        if st.button("Use this key", disabled=not key):
            sess.set_api_key(key)
            st.rerun()
        if live and st.button("Switch to offline demo"):
            sess.set_api_key(None)
            st.rerun()

    sb.subheader("Data")
    files = sb.file_uploader("Upload one or more CSV files", type=["csv", "tsv", "txt"], accept_multiple_files=True)
    for f in files or []:
        fid = (f.name, f.size)
        if fid in st.session_state.seen_files:
            continue
        st.session_state.seen_files.add(fid)
        try:
            ds = sess.workspace.add_csv(f.getvalue(), f.name)
            sb.success(f"Loaded **{ds.table}** ({ds.shape[0]:,} × {ds.shape[1]})")
            for w in ds.warnings:
                sb.warning(w)
        except AnalystError as exc:
            sb.error(str(exc))
    if sb.button("Load sample data (sales + customers)", width="stretch"):
        for name in ("sales.csv", "customers.csv"):
            if not any(d.name == name for d in sess.workspace.datasets.values()):
                sess.workspace.add_csv((SAMPLE_DIR / name).read_bytes(), name)
        st.rerun()

    for table, ds in list(sess.workspace.datasets.items()):
        c1, c2 = sb.columns([5, 1])
        c1.markdown(f"**{table}** · {ds.shape[0]:,} rows")
        if c2.button("✕", key=f"rm-{table}", help="Remove"):
            sess.workspace.remove(table)
            st.session_state.seen_files = set()
            st.rerun()

    sb.divider()
    if sb.button("🗑 New conversation", width="stretch"):
        sess.reset_conversation()
        st.rerun()
    if sess.turns:
        names = [d.name for d in sess.workspace.datasets.values()]
        sb.download_button("⬇ Export report (HTML)", to_html(sess.turns, names), "analysis_report.html", "text/html",
                           width="stretch")
        sb.download_button("⬇ Export report (Markdown)", to_markdown(sess.turns, names), "analysis_report.md",
                           "text/markdown", width="stretch")


def _chat_tab(sess: AnalystSession) -> None:
    if not sess.workspace.datasets:
        st.info("👈 Upload one or more CSV files — or click **Load sample data** — then ask a question in plain English.")
    for i, turn in enumerate(sess.turns):
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            render_turn(turn, key=f"t{i}")

    if sess.workspace.datasets and not sess.turns:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 3].button(q, key=f"ex{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

    typed = st.chat_input("Ask about your data…", disabled=not sess.workspace.datasets)
    question = st.session_state.pop("pending", None) or typed
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analysing…"):
                    turn = sess.ask(question)
                render_turn(turn, key=f"t{len(sess.turns) - 1}")
            except LLMError as exc:
                st.error(f"⚠️ {exc}")
            except AnalystError as exc:
                st.error(str(exc))
            except Exception:  # noqa: BLE001
                log.exception("unhandled error")
                st.error("Something went wrong while analysing your question. Please try rephrasing it.")


def _data_tab(sess: AnalystSession) -> None:
    if not sess.workspace.datasets:
        st.info("No data loaded yet.")
        return
    table = st.selectbox("Table", list(sess.workspace.datasets))
    ds = sess.workspace.datasets[table]
    st.caption(f"{ds.name} · {ds.shape[0]:,} rows × {ds.shape[1]} columns")
    st.dataframe(ds.df.head(200), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    if c1.button("🩺 Run data-quality check", width="stretch"):
        render_quality(assess_quality(ds.df, table))
    if c2.button("🚨 Detect anomalies (no LLM)", width="stretch"):
        rep = detect_anomalies(ds.df)
        st.info(rep.summary)
        if rep.count:
            st.dataframe(rep.table, width="stretch", hide_index=True,
                         column_config={"reason": st.column_config.TextColumn("Why it was flagged", width="large")})


def main() -> None:
    settings = _settings()
    _gate(settings)
    sess = _session(settings)
    _sidebar(sess)
    st.title("Ask your data anything")
    chat, data = st.tabs(["💬 Chat", "🗂 Data explorer"])
    with chat:
        _chat_tab(sess)
    with data:
        _data_tab(sess)


main()
