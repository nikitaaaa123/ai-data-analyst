import numpy as np
import pandas as pd
import pytest

from data_analyst.anomaly import detect_anomalies
from data_analyst.charts import CHART_TYPES, build_chart
from data_analyst.errors import ChartError, QueryExecutionError
from data_analyst.forecast import forecast_series
from data_analyst.quality import assess_quality


def test_anomaly_detects_planted_outliers(sample_ws):
    rep = detect_anomalies(sample_ws.datasets["sales"].df)
    flagged = set(rep.table["row"])
    assert 905 in flagged            # 48,500 revenue typo
    assert 310 in flagged            # 250-unit bulk order
    assert 1422 in flagged           # negative revenue
    assert "duplicate_row" in set(rep.table["method"])
    assert rep.count < 20            # not noisy
    assert rep.table["reason"].str.len().min() > 20


def test_anomaly_clean_data_has_none():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(100, 5, 300)})
    assert detect_anomalies(df).count == 0


def test_quality_report(sample_ws):
    rep = assess_quality(sample_ws.datasets["sales"].df, "sales")
    text = " ".join(rep.issues)
    assert "region" in text and "duplicated" in text and "negative" in text
    assert 0 <= rep.score < 100


@pytest.mark.parametrize("kind", CHART_TYPES)
def test_every_chart_type(kind):
    df = pd.DataFrame({"k": list("abcd"), "v": [4, 3, 2, 1], "w": [1.5, 2, 3, 4]})
    assert build_chart(df, kind, x="k" if kind != "scatter" else "w", y="v", title="t") is not None


def test_chart_errors():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ChartError):
        build_chart(df, "bar", x="zzz")
    with pytest.raises(ChartError):
        build_chart(df, "donut")
    with pytest.raises(ChartError):
        build_chart(df.iloc[0:0], "bar")


def test_forecast_shape_and_errors():
    idx = pd.date_range("2023-01-01", periods=30, freq="MS")
    df = pd.DataFrame({"d": idx, "v": np.arange(30) * 10.0 + 100})
    out = forecast_series(df, "d", "v", 6)
    assert out["forecast"].notna().sum() == 6
    assert out.loc[out.forecast.notna(), "forecast"].iloc[0] == pytest.approx(400, rel=0.05)
    with pytest.raises(QueryExecutionError):
        forecast_series(df.head(3), "d", "v")
    with pytest.raises(QueryExecutionError):
        forecast_series(df, "nope", "v")
