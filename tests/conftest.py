import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_analyst import Settings  # noqa: E402
from data_analyst.workspace import Workspace  # noqa: E402

CSV = b"order_id,order_date,region,revenue,quantity\n" + b"\n".join(
    f"O{i},2024-0{1 + i % 9}-1{i % 9},{['N','S','E'][i % 3]},{100 + i % 7},{1 + i % 4}".encode() for i in range(60)
) + b"\nO999,2024-05-05,N,99999,2\n"


@pytest.fixture
def ws():
    w = Workspace(Settings())
    w.add_csv(CSV, "orders.csv")
    return w


@pytest.fixture
def sample_ws():
    w = Workspace(Settings())
    for n in ("sales", "customers"):
        w.add_csv(ROOT / "data" / f"{n}.csv", f"{n}.csv")
    return w
