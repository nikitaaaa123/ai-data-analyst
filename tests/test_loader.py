import pytest

from data_analyst.errors import DataValidationError
from data_analyst.loader import load_csv, sanitize_identifier


def test_basic_load_and_types():
    ds = load_csv(b"Order Date,Total $,Name\n2024-01-01,\"$1,200\",a\n2024-01-02,300,b\n", "My Sales.csv")
    assert ds.table == "my_sales"
    assert list(ds.df.columns) == ["order_date", "total", "name"]
    assert str(ds.df["order_date"].dtype).startswith("datetime")
    assert ds.df["total"].tolist() == [1200, 300]
    assert ds.warnings  # names normalised


def test_semicolon_delimiter_and_latin1():
    ds = load_csv("a;b\n1;café\n2;té\n".encode("latin-1"), "x.csv")
    assert ds.df.shape == (2, 2)
    assert ds.df["b"].iloc[0] == "café"


@pytest.mark.parametrize("raw,name", [(b"", "a.csv"), (b"   \n", "a.csv"), (b"a,b\n", "a.csv"), (b"a,b\n1,2\n", "a.pdf")])
def test_invalid_files_rejected(raw, name):
    with pytest.raises(DataValidationError):
        load_csv(raw, name)


def test_size_limit():
    with pytest.raises(DataValidationError, match="limit"):
        load_csv(b"a\n" + b"1\n" * 2_000_000, "big.csv", max_mb=1)


def test_unique_table_names_and_duplicate_columns():
    ds = load_csv(b"a,a,A\n1,2,3\n", "t.csv", existing_tables={"t"})
    assert ds.table == "t_2"
    assert len(set(ds.df.columns)) == 3


def test_sanitize():
    assert sanitize_identifier("2024 Sales!") == "col_2024_sales"
