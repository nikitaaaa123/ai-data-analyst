"""Read-only DuckDB engine over the uploaded DataFrames."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import duckdb
import pandas as pd

from .errors import QueryExecutionError, UnsafeQueryError
from .loader import Dataset

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql: str
    truncated: bool
    elapsed_ms: float
    cached: bool = False


class SQLEngine:
    """Executes validated SELECT statements only.

    Safety layers: (1) statement parsed by DuckDB and must be a single SELECT,
    (2) external file/network access disabled, (3) timeout via interrupt,
    (4) row cap.
    """

    def __init__(self, max_rows: int = 1000, timeout_s: float = 20.0, cache_size: int = 64):
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._con = duckdb.connect(":memory:")
        self._locked = False
        self._lock = threading.Lock()
        self._tables: dict[str, pd.DataFrame] = {}
        self._cache: OrderedDict[str, QueryResult] = OrderedDict()
        self._cache_size = cache_size

    # -- table management -------------------------------------------------
    def register(self, ds: Dataset) -> None:
        with self._lock:
            # Materialise as a real table: cursors (used per query for timeouts) do not
            # see connection-local registered DataFrames.
            self._con.register("_incoming", ds.df)
            self._con.execute(f'CREATE OR REPLACE TABLE "{ds.table}" AS SELECT * FROM _incoming')
            self._con.unregister("_incoming")
            self._tables[ds.table] = ds.df
            self._cache.clear()
            self._lock_down()

    def _lock_down(self) -> None:
        """Block file/network access from SQL (read_csv('/etc/passwd'), httpfs...) and freeze settings."""
        if not self._locked:
            self._con.execute("SET enable_external_access=false")
            self._con.execute("SET lock_configuration=true")
            self._locked = True

    def unregister(self, table: str) -> None:
        with self._lock:
            self._con.execute(f'DROP TABLE IF EXISTS "{table}"')
            self._tables.pop(table, None)
            self._cache.clear()

    @property
    def tables(self) -> list[str]:
        return list(self._tables)

    # -- validation -------------------------------------------------------
    def validate(self, sql: str) -> str:
        sql = (sql or "").strip().rstrip(";").strip()
        if not sql:
            raise UnsafeQueryError("The SQL query is empty.")
        try:
            stmts = self._con.extract_statements(sql)
        except duckdb.Error as exc:
            raise QueryExecutionError(f"SQL syntax error: {exc}") from exc
        if len(stmts) != 1:
            raise UnsafeQueryError("Only a single SQL statement is allowed.")
        if stmts[0].type != duckdb.StatementType.SELECT:
            raise UnsafeQueryError("Only read-only SELECT queries are allowed.")
        return sql

    # -- execution --------------------------------------------------------
    def execute(self, sql: str) -> QueryResult:
        sql = self.validate(sql)
        if sql in self._cache:
            self._cache.move_to_end(sql)
            hit = self._cache[sql]
            return QueryResult(hit.df.copy(), sql, hit.truncated, 0.0, cached=True)

        wrapped = f"SELECT * FROM ({sql}) AS _q LIMIT {self.max_rows + 1}"
        start = time.perf_counter()
        # Cursor per call: thread-safe and lets us interrupt on timeout.
        with self._lock:
            cur = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        timer.start()
        try:
            df = cur.execute(wrapped).fetchdf()
        except duckdb.InterruptException as exc:
            raise QueryExecutionError(f"Query exceeded the {self.timeout_s:.0f}s time limit.") from exc
        except duckdb.Error as exc:
            raise QueryExecutionError(f"SQL error: {exc}") from exc
        finally:
            timer.cancel()
            cur.close()

        truncated = len(df) > self.max_rows
        if truncated:
            df = df.head(self.max_rows)
        elapsed = (time.perf_counter() - start) * 1000
        result = QueryResult(df, sql, truncated, elapsed)
        self._cache[sql] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        log.info("sql ok rows=%d truncated=%s %.1fms", len(df), truncated, elapsed)
        return QueryResult(df.copy(), sql, truncated, elapsed)
