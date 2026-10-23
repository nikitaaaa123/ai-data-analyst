"""Lightweight, transparent time-series forecasting (trend + seasonality)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import QueryExecutionError


def forecast_series(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 6) -> pd.DataFrame:
    """Monthly linear-trend forecast with optional month-of-year seasonality.

    Returns columns: period, actual, forecast, lower, upper (95% band from residual std).
    """
    for c in (date_col, value_col):
        if c not in df.columns:
            raise QueryExecutionError(f"Column '{c}' not found. Available: {list(df.columns)}")
    periods = max(1, min(int(periods), 36))
    s = df[[date_col, value_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna()
    monthly = s.set_index(date_col)[value_col].astype(float).resample("MS").sum()
    if len(monthly) < 6:
        raise QueryExecutionError("Need at least 6 monthly data points to forecast.")

    t = np.arange(len(monthly))
    slope, intercept = np.polyfit(t, monthly.values, 1)
    trend = slope * t + intercept
    resid = monthly.values - trend

    seasonal = np.zeros(12)
    if len(monthly) >= 24:
        months = monthly.index.month.values
        for m in range(1, 13):
            seasonal[m - 1] = resid[months == m].mean()
        resid = resid - seasonal[months - 1]
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

    fut_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    ft = np.arange(len(monthly), len(monthly) + periods)
    fc = slope * ft + intercept + seasonal[fut_idx.month.values - 1]
    hist = pd.DataFrame({"period": monthly.index, "actual": monthly.values,
                         "forecast": np.nan, "lower": np.nan, "upper": np.nan})
    fut = pd.DataFrame({"period": fut_idx, "actual": np.nan, "forecast": fc,
                        "lower": fc - 1.96 * sigma, "upper": fc + 1.96 * sigma})
    return pd.concat([hist, fut], ignore_index=True)
