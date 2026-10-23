"""Agent loop tests with a scripted fake LLM (no network)."""
import pytest

from data_analyst import AnalystSession, Settings
from data_analyst.agent import DataAnalystAgent
from data_analyst.errors import LLMError
from data_analyst.llm import LLMResponse


class Scripted:
    name = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def create(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        self.system = system
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def tool(name, **inp):
    return LLMResponse([{"type": "tool_use", "id": "t1", "name": name, "input": inp}], "tool_use", 10, 5)


def text(t):
    return LLMResponse([{"type": "text", "text": t}], "end_turn", 10, 5)


def test_tool_loop_and_artifacts(ws):
    llm = Scripted(tool("run_sql", sql="SELECT region, SUM(revenue) r FROM orders GROUP BY 1", purpose="p"),
                   text("North leads. **How I got this** ..."))
    turn = DataAnalystAgent(llm, ws).ask("which region?")
    assert turn.answer.startswith("North")
    assert [c.name for c in turn.tool_calls] == ["run_sql"] and turn.tool_calls[0].ok
    assert {a.kind for a in turn.artifacts} == {"code", "table"}
    assert turn.input_tokens == 20
    assert "orders" in llm.system                       # schema injected


def test_tool_error_is_fed_back_for_self_correction(ws):
    llm = Scripted(tool("run_sql", sql="SELECT bad FROM orders", purpose="x"),
                   tool("run_sql", sql="SELECT COUNT(*) c FROM orders", purpose="x"), text("done"))
    agent = DataAnalystAgent(llm, ws)
    turn = agent.ask("count")
    assert [c.ok for c in turn.tool_calls] == [False, True]
    err_msg = llm.seen[1][-1]["content"][0]
    assert err_msg["is_error"] and "SQL error" in err_msg["content"]


def test_conversation_context_is_kept(ws):
    llm = Scripted(text("first"), text("second"))
    agent = DataAnalystAgent(llm, ws)
    agent.ask("q1")
    agent.ask("q2")
    assert [m["content"] for m in llm.seen[1] if m["role"] == "user"][0] == "q1"
    assert len(llm.seen[1]) == 3


def test_llm_failure_rolls_back_history(ws):
    llm = Scripted(text("ok"), LLMError("boom"), text("again"))
    agent = DataAnalystAgent(llm, ws)
    agent.ask("q1")
    n = len(agent.messages)
    with pytest.raises(LLMError):
        agent.ask("q2")
    assert len(agent.messages) == n
    agent.ask("q3")  # history still valid


def test_unknown_tool_and_bad_args(ws):
    llm = Scripted(tool("nope"), tool("run_sql"), text("fine"))
    turn = DataAnalystAgent(llm, ws).ask("x")
    assert [c.ok for c in turn.tool_calls] == [False, False]


def test_step_limit(ws):
    s = Settings(max_tool_iterations=2)
    w = type(ws)(s)
    w.add_csv(b"a,b\n1,2\n3,4\n", "t.csv")
    llm = Scripted(*[tool("describe_data") for _ in range(5)])
    turn = DataAnalystAgent(llm, w, s).ask("loop")
    assert "step limit" in turn.answer


def test_history_window_starts_on_user_text(ws):
    s = Settings(max_history_messages=4)
    llm = Scripted(*[text(f"a{i}") for i in range(6)])
    agent = DataAnalystAgent(llm, ws, s)
    for i in range(6):
        agent.ask(f"q{i}")
    first = llm.seen[-1][0]
    assert first["role"] == "user" and isinstance(first["content"], str)


def test_empty_question(ws):
    with pytest.raises(ValueError):
        DataAnalystAgent(Scripted(), ws).ask("  ")


def test_demo_llm_end_to_end(sample_ws):
    sess = AnalystSession(Settings())
    for ds in sample_ws.datasets.values():
        sess.workspace.add_csv(ds.df.to_csv(index=False).encode(), ds.name)
    t = sess.ask("Which region generated the highest revenue?")
    assert "North America" in t.answer
    t = sess.ask("Show monthly sales trends.")
    assert any(a.kind == "chart" for a in t.artifacts)
    t = sess.ask("What are the top five customers?")
    assert len([a for a in t.artifacts if a.kind == "table"][0].payload) == 5
    t = sess.ask("Detect anomalies in the dataset.")
    assert any(a.kind == "anomalies" for a in t.artifacts)
    t = sess.ask("Generate SQL for this analysis.")
    assert "```sql" in t.answer
