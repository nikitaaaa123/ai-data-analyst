"""Explainable anomaly detection (robust statistics, no black boxes)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_CONSTANT = 0.6745  # makes MAD comparable to a standard deviation for normal data


@dataclass
class AnomalyReport:
    table: pd.DataFrame            # row, id, column, value, method, score, reason
    summary: str
    columns_checked: list[str]

    @property
    def count(self) -> int:
        return len(self.table)


def _id_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_id") or c == "id"]


def _fmt(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _numeric_candidates(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                       and not pd.api.types.is_bool_dtype(df[c])]
    ids = set(_id_columns(df))
    out = []
    for c in cols:
        if c in ids or c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) < 5:   # flags / tiny enums are not measurements
            continue
        out.append(c)
    return out


def _robust_z(s: pd.Series) -> tuple[pd.Series, float, float, bool]:
    """Modified z-score. Returns (z, median, mad, used_std_fallback)."""
    median = float(s.median())
    mad = float((s - median).abs().median())
    if mad > 0:
        return MAD_CONSTANT * (s - median) / mad, median, mad, False
    std = float(s.std())
    if not std or np.isnan(std):
        return pd.Series(0.0, index=s.index), median, mad, True
    return (s - float(s.mean())) / std, median, std, True


def _best_group(df: pd.DataFrame, col: str, work: pd.Series) -> str | None:
    """Pick a categorical column that explains most of `col`'s variation (contextual baseline)."""
    global_mad = float((work - work.median()).abs().median())
    if global_mad == 0:
        return None
    best, best_ratio = None, 0.5          # grouping must at least halve the typical spread
    for g in df.columns:
        if g == col or pd.api.types.is_numeric_dtype(df[g]) or pd.api.types.is_datetime64_any_dtype(df[g]):
            continue
        n = df[g].nunique(dropna=True)
        if not 2 <= n <= 40 or n > len(df) / 8:
            continue
        grp = work.groupby(df[g], observed=True)
        mads = grp.apply(lambda x: float((x - x.median()).abs().median()))
        mad = float((mads * grp.size()).sum() / grp.size().sum())   # size-weighted spread within groups
        if mad / global_mad < best_ratio:
            best, best_ratio = g, mad / global_mad
    return best


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    z_threshold: float = 3.5,
    max_results: int = 50,
) -> AnomalyReport:
    """Flag outliers per numeric column plus duplicate rows.

    Methods
    -------
    * ``modified_z``: |0.6745 * (x - median) / MAD| > z_threshold (Iglewicz-Hoaglin).
      Skewed positive columns are log-transformed first; if another categorical column
      explains the spread (e.g. unit_price by product), the score is computed *within* that
      group so normal differences between groups are not flagged. Falls back to mean/std
      when MAD is 0.
    * ``negative_value``: negatives in a column that is almost always >= 0.
    * ``duplicate_row``: fully identical rows.
    """
    ids = _id_columns(df)
    rows: list[dict] = []
    cols = _numeric_candidates(df, columns)

    def _row_id(idx: int) -> str:
        return ", ".join(f"{c}={df.at[idx, c]}" for c in ids[:1]) if ids else ""

    for col in cols:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        use_log = float((s > 0).mean()) >= 0.99 and abs(float(s.skew())) > 1
        work = np.log1p(s.clip(lower=0)) if use_log else s
        group = _best_group(df.loc[s.index], col, work)
        if group:
            z = pd.Series(0.0, index=s.index)
            ctx: dict = {}
            for key, idxs in df.loc[s.index].groupby(group, observed=True).groups.items():
                if len(idxs) < 8:
                    continue
                gz, med, mad, _ = _robust_z(work.loc[idxs])
                z.loc[idxs] = gz
                ctx[key] = float(s.loc[idxs].median())
        else:
            z, med, mad, _ = _robust_z(work)
            ctx = {}
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        for idx in z[z.abs() > z_threshold].index:
            v = float(df.at[idx, col])
            direction = "above" if z[idx] > 0 else "below"
            if group:
                typical = ctx.get(df.at[idx, group], float(s.median()))
                where = f"for {group}='{df.at[idx, group]}' (typical {_fmt(typical)})"
            else:
                where = f"overall (median {_fmt(float(s.median()))}, IQR {_fmt(q1)}-{_fmt(q3)})"
            how = "log-scale " if use_log else ""
            rows.append({
                "row": int(idx), "id": _row_id(idx), "column": col, "value": v,
                "method": "modified_z", "score": round(float(abs(z[idx])), 2),
                "reason": (f"{col}={_fmt(v)} is far {direction} normal {where}; "
                           f"{how}robust z-score {z[idx]:.1f} exceeds ±{z_threshold}."),
            })
        s_all = df[col].dropna()
        neg_share = float((s_all < 0).mean())
        if 0 < neg_share <= 0.01:
            for idx in s_all[s_all < 0].index:
                rows.append({
                    "row": int(idx), "id": _row_id(idx), "column": col, "value": float(s_all[idx]),
                    "method": "negative_value", "score": 10.0,
                    "reason": f"{col}={_fmt(float(s_all[idx]))} is negative while {100 * (1 - neg_share):.1f}% "
                              f"of values are non-negative — likely a data-entry error or a refund.",
                })

    dup_mask = df.duplicated(keep="first")
    for idx in df.index[dup_mask]:
        rows.append({
            "row": int(idx), "id": _row_id(idx), "column": "(all columns)", "value": np.nan,
            "method": "duplicate_row", "score": 5.0,
            "reason": "Row is an exact duplicate of an earlier row.",
        })

    table = pd.DataFrame(rows, columns=["row", "id", "column", "value", "method", "score", "reason"])
    if not table.empty:
        table = table.sort_values("score", ascending=False).head(max_results).reset_index(drop=True)

    if table.empty:
        summary = f"No anomalies found in {len(cols)} numeric column(s) (threshold ±{z_threshold})."
    else:
        by_method = table["method"].value_counts().to_dict()
        summary = (f"Flagged {len(table)} anomalous value(s) across {len(cols)} numeric column(s): "
                   + ", ".join(f"{n} {m}" for m, n in by_method.items()) + ".")
    return AnomalyReport(table=table, summary=summary, columns_checked=cols)
