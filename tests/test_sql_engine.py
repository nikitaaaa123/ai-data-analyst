import pytest

from data_analyst.errors import QueryExecutionError, UnsafeQueryError


def test_select_works(ws):
    r = ws.engine.execute("SELECT region, SUM(revenue) AS r FROM orders GROUP BY 1 ORDER BY 2 DESC")
    assert list(r.df.columns) == ["region", "r"] and len(r.df) == 3


def test_cache_hit(ws):
    ws.engine.execute("SELECT COUNT(*) FROM orders")
    assert ws.engine.execute("SELECT COUNT(*) FROM orders;").cached


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "DELETE FROM orders", "INSERT INTO orders VALUES (1)", "CREATE TABLE x AS SELECT 1",
    "SELECT 1; SELECT 2", "COPY orders TO '/tmp/x.csv'", "ATTACH '/tmp/x.db'", "", "PRAGMA database_list",
])
def test_unsafe_sql_blocked(sql):
    from data_analyst.sql_engine import SQLEngine
    with pytest.raises((UnsafeQueryError, QueryExecutionError)):
        SQLEngine().execute(sql)


def test_external_file_reads_blocked(ws):
    with pytest.raises(QueryExecutionError):
        ws.engine.execute("SELECT * FROM read_csv('/etc/passwd')")


def test_sql_error_is_friendly(ws):
    with pytest.raises(QueryExecutionError, match="SQL error"):
        ws.engine.execute("SELECT nope FROM orders")


def test_row_cap():
    from data_analyst.sql_engine import SQLEngine
    r = SQLEngine(max_rows=10).execute("SELECT * FROM range(100)")
    assert len(r.df) == 10 and r.truncated
