"""Data-quality profiling."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QualityReport:
    table: str
    rows: int
    columns: int
    score: int
    issues: list[str] = field(default_factory=list)
    column_profile: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_text(self) -> str:
        head = f"Data quality for '{self.table}': score {self.score}/100 ({self.rows:,} rows, {self.columns} columns)."
        if not self.issues:
            return head + " No issues found."
        return head + "\nIssues:\n" + "\n".join(f"- {i}" for i in self.issues)


def assess_quality(df: pd.DataFrame, table: str = "data") -> QualityReport:
    n = len(df)
    issues: list[str] = []
    penalty = 0.0
    profile = []

    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        pct = 100 * miss / n if n else 0
        profile.append({
            "column": col, "dtype": str(s.dtype), "missing": miss, "missing_pct": round(pct, 1),
            "distinct": int(s.nunique(dropna=True)),
        })
        if pct >= 1:
            issues.append(f"'{col}' has {miss:,} missing values ({pct:.1f}%).")
            penalty += min(pct, 40) / max(len(df.columns), 1) * 2
        if s.nunique(dropna=True) <= 1 and n > 1:
            issues.append(f"'{col}' is constant (carries no information).")
            penalty += 2
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            neg = int((s < 0).sum())
            if 0 < neg < 0.01 * n + 1:
                issues.append(f"'{col}' has {neg} negative value(s) in an otherwise non-negative column.")
                penalty += 3
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            vals = s.dropna().astype(str)
            if len(vals):
                trimmed = vals.str.strip().str.lower()
                if trimmed.nunique() < vals.nunique():
                    issues.append(f"'{col}' has inconsistent casing/whitespace variants of the same value.")
                    penalty += 3

    dups = int(df.duplicated().sum())
    if dups:
        issues.append(f"{dups:,} fully duplicated row(s) ({100 * dups / n:.1f}%).")
        penalty += min(100 * dups / n, 20)

    score = max(0, round(100 - penalty))
    return QualityReport(table, n, df.shape[1], score, issues, pd.DataFrame(profile))
