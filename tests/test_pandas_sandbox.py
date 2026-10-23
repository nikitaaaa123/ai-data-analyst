import pandas as pd
import pytest

from data_analyst.errors import QueryExecutionError, UnsafeQueryError
from data_analyst.pandas_sandbox import run_pandas

T = {"orders": pd.DataFrame({"a": [1, 2, 3], "g": ["x", "x", "y"]})}


def test_runs_and_returns_frame():
    r = run_pandas("result = orders.groupby('g')['a'].sum()", T)
    assert r.df.shape == (2, 2)


def test_original_not_mutated():
    run_pandas("orders['a'] = 0\nresult = orders", T)
    assert T["orders"]["a"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("code", [
    "import os\nresult = 1", "result = open('/etc/passwd').read()", "result = __import__('os')",
    "result = orders.__class__", "result = pd.read_csv('x.csv')", "orders.to_csv('x.csv')\nresult=1",
    "result = eval('1+1')", "result = getattr(orders, 'a')",
])
def test_unsafe_code_blocked(code):
    with pytest.raises(UnsafeQueryError):
        run_pandas(code, T)


def test_missing_result_and_runtime_errors():
    with pytest.raises(QueryExecutionError, match="result"):
        run_pandas("x = 1", T)
    with pytest.raises(QueryExecutionError, match="KeyError"):
        run_pandas("result = orders['zzz']", T)
    with pytest.raises(QueryExecutionError, match="syntax"):
        run_pandas("result = (", T)


def test_timeout():
    with pytest.raises(QueryExecutionError, match="time limit"):
        run_pandas("while True:\n    pass", T, timeout_s=0.3)
