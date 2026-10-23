"""Workspace: the set of datasets loaded in one user session."""
from __future__ import annotations

from typing import BinaryIO

from .config import Settings
from .loader import Dataset, load_csv
from .sql_engine import SQLEngine


class Workspace:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.datasets: dict[str, Dataset] = {}
        self.engine = SQLEngine(self.settings.max_result_rows, self.settings.query_timeout_s)

    def add_csv(self, source: bytes | BinaryIO | str, filename: str) -> Dataset:
        ds = load_csv(source, filename, set(self.datasets), self.settings.max_upload_mb)
        self.datasets[ds.table] = ds
        self.engine.register(ds)
        return ds

    def remove(self, table: str) -> None:
        if table in self.datasets:
            del self.datasets[table]
            self.engine.unregister(table)

    def clear(self) -> None:
        for t in list(self.datasets):
            self.remove(t)

    def frames(self) -> dict:
        return {t: d.df for t, d in self.datasets.items()}

    def schema_text(self) -> str:
        if not self.datasets:
            return "No datasets are loaded."
        return "\n\n".join(d.schema_text() for d in self.datasets.values())
