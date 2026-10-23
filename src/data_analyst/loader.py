"""CSV upload, validation and normalisation."""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .errors import DataValidationError

log = logging.getLogger(__name__)

_NUMERIC_CLEAN = re.compile(r"[,$€£%\s]")


@dataclass
class Dataset:
    """A loaded CSV, with SQL-friendly names."""

    name: str                     # original file name
    table: str                    # SQL table name
    df: pd.DataFrame
    column_map: dict[str, str] = field(default_factory=dict)  # original -> sanitized
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def schema_text(self, sample_rows: int = 3) -> str:
        lines = [f'Table "{self.table}" (from {self.name}): {len(self.df):,} rows']
        for col in self.df.columns:
            s = self.df[col]
            extra = ""
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                extra = f" min={s.min():g} max={s.max():g}"
            elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
                extra = f" {s.min().date()}..{s.max().date()}"
            else:
                n = s.nunique(dropna=True)
                extra = f" {n} distinct"
                if 0 < n <= 8:
                    extra += ": " + ", ".join(map(str, s.dropna().unique()[:8]))
            lines.append(f"  - {col} ({_sql_type(s)}){extra}")
        sample = self.df.head(sample_rows).to_csv(index=False)
        lines.append("  sample rows:\n    " + sample.strip().replace("\n", "\n    "))
        return "\n".join(lines)


def _sql_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "TIMESTAMP"
    return "VARCHAR"


def sanitize_identifier(raw: str, fallback: str = "col") -> str:
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip()).strip("_").lower()
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"{fallback}_{ident}"
    return ident


def _unique(name: str, taken: set[str]) -> str:
    candidate, i = name, 2
    while candidate in taken:
        candidate = f"{name}_{i}"
        i += 1
    return candidate


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise DataValidationError("Could not decode the file. Please save it as UTF-8 CSV.")


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _infer_types(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Convert numeric-looking and date-looking text columns."""
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna().astype(str)
        if non_null.empty:
            continue
        cleaned = non_null.str.replace(_NUMERIC_CLEAN, "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            df[col] = pd.to_numeric(s.astype("string").str.replace(_NUMERIC_CLEAN, "", regex=True), errors="coerce")
            bad = int(df[col].isna().sum() - s.isna().sum())
            if bad:
                warnings.append(f"Column '{col}': {bad} non-numeric value(s) were set to null.")
            continue
        looks_dated = bool(re.search(r"date|time|month|day|_at$|year", col, re.I))
        sample = non_null.head(200)
        if looks_dated or sample.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True).mean() > 0.8:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if parsed.notna().sum() >= 0.9 * s.notna().sum() and s.notna().sum() > 0:
                df[col] = parsed
    return df


def load_csv(
    source: bytes | BinaryIO | str | Path,
    filename: str,
    existing_tables: set[str] | None = None,
    max_mb: int = 50,
) -> Dataset:
    """Read, validate and normalise a CSV into a :class:`Dataset`."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()

    if not filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise DataValidationError(f"'{filename}' is not a CSV file.")
    if not raw or not raw.strip():
        raise DataValidationError(f"'{filename}' is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DataValidationError(f"'{filename}' exceeds the {max_mb} MB upload limit.")

    text = _decode(raw)
    sep = _sniff_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"'{filename}' has no data.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"'{filename}' could not be parsed as CSV: {exc}") from exc

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataValidationError(f"'{filename}' has no rows or no columns.")

    warnings: list[str] = []
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise DataValidationError(f"'{filename}' contains only empty cells.")

    taken: set[str] = set()
    column_map: dict[str, str] = {}
    for i, col in enumerate(df.columns):
        clean = _unique(sanitize_identifier(col, f"column{i}"), taken)
        taken.add(clean)
        column_map[str(col)] = clean
    if any(k != v for k, v in column_map.items()):
        warnings.append("Column names were normalised to snake_case for SQL.")
    df.columns = list(column_map.values())

    df = _infer_types(df.copy(), warnings)
    df = df.reset_index(drop=True)

    base = sanitize_identifier(Path(filename).stem, "table")
    table = _unique(base, set(existing_tables or ()))
    log.info("loaded %s as %s: %d rows x %d cols", filename, table, *df.shape)
    return Dataset(name=filename, table=table, df=df, column_map=column_map, warnings=warnings)
