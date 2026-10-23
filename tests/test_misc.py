import pytest

from data_analyst.llm import AnthropicLLM
from data_analyst.errors import LLMError
from data_analyst.report import to_html, to_markdown
from data_analyst import AnalystSession, Settings


def test_report_exports(sample_ws):
    s = AnalystSession(Settings())
    s.workspace.add_csv(sample_ws.datasets["sales"].df.to_csv(index=False).encode(), "sales.csv")
    s.ask("Show monthly sales trends.")
    md, html = to_markdown(s.turns, ["sales.csv"]), to_html(s.turns, ["sales.csv"])
    assert "```sql" in md and "plotly" in html.lower()


def test_anthropic_errors_are_mapped(monkeypatch):
    import anthropic
    import httpx

    llm = AnthropicLLM("sk-test")
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad", response=httpx.Response(401, request=req), body=None)

    def boom(**kw):
        raise err
    monkeypatch.setattr(llm._client.messages, "create", boom)
    with pytest.raises(LLMError, match="API key"):
        llm.create("s", [{"role": "user", "content": "hi"}], [])


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ANALYST_MODEL", "m1")
    monkeypatch.setenv("ANALYST_MAX_UPLOAD_MB", "oops")
    s = Settings.from_env()
    assert s.model == "m1" and s.max_upload_mb == 50
