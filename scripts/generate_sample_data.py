"""Generate reproducible sample datasets with deliberately planted anomalies.

    python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data"

REGIONS = {"North America": 1.4, "Europe": 1.2, "Asia Pacific": 1.0, "Latin America": 0.6, "Middle East & Africa": 0.4}
PRODUCTS = {  # name: (category, price, popularity)
    "Aurora Laptop": ("Electronics", 1200, 1.0), "Pulse Headphones": ("Electronics", 180, 1.6),
    "Nimbus Tablet": ("Electronics", 550, 0.9), "Orbit Smartwatch": ("Electronics", 260, 0.7),
    "ErgoChair Pro": ("Furniture", 340, 0.8), "Standing Desk": ("Furniture", 620, 0.6),
    "Lumen Desk Lamp": ("Furniture", 65, 1.1), "Notebook Pack": ("Office Supplies", 14, 2.0),
    "Gel Pen Set": ("Office Supplies", 9, 2.2), "Label Printer": ("Office Supplies", 120, 0.15),  # underperformer
    "Cable Organizer": ("Accessories", 22, 0.12),                                                 # underperformer
    "Webcam HD": ("Accessories", 85, 1.0),
}

# customers -------------------------------------------------------------
n_cust = 120
customers = pd.DataFrame({
    "customer_id": [f"C{1000 + i}" for i in range(n_cust)],
    "customer_name": [f"{a} {b}" for a, b in zip(
        rng.choice(["Alex", "Priya", "Chen", "Maria", "Omar", "Sofia", "Liam", "Aisha", "Noah", "Yuki"], n_cust),
        rng.choice(["Patel", "Garcia", "Smith", "Kim", "Nguyen", "Brown", "Khan", "Silva", "Muller", "Sato"], n_cust))],
    "segment": rng.choice(["Enterprise", "SMB", "Consumer"], n_cust, p=[0.2, 0.35, 0.45]),
    "country": rng.choice(["USA", "UK", "Germany", "India", "Japan", "Brazil", "UAE", "Canada"], n_cust),
    "signup_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n_cust), unit="D"),
})
customers["signup_date"] = customers["signup_date"].dt.date
whale = customers.index[:6]  # a few big customers -> meaningful "top customers"
cust_w = np.ones(n_cust)
cust_w[whale] = 8

# sales -----------------------------------------------------------------
n = 2400
dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
season = 1 + 0.35 * np.sin((dates.month - 3) / 12 * 2 * np.pi) + 0.45 * (dates.month >= 11) \
    + 0.25 * (dates - pd.Timestamp("2024-01-01")).days / 730          # seasonality + growth
keep = rng.random(n) < season / season.max()
dates = dates[keep]
n = len(dates)
region = rng.choice(list(REGIONS), n, p=np.array(list(REGIONS.values())) / sum(REGIONS.values()))
prod_names = list(PRODUCTS)
pop = np.array([PRODUCTS[p][2] for p in prod_names])
product = rng.choice(prod_names, n, p=pop / pop.sum())
qty = rng.integers(1, 6, n)
base_price = np.array([PRODUCTS[p][1] for p in product], dtype=float)
discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], n)
unit_price = np.round(base_price * rng.normal(1, 0.03, n), 2)
revenue = np.round(qty * unit_price * (1 - discount), 2)

sales = pd.DataFrame({
    "order_id": [f"ORD-{20000 + i}" for i in range(n)],
    "order_date": dates.date,
    "region": region,
    "product": product,
    "category": [PRODUCTS[p][0] for p in product],
    "customer_id": rng.choice(customers["customer_id"], n, p=cust_w / cust_w.sum()),
    "quantity": qty,
    "unit_price": unit_price,
    "discount": discount,
    "revenue": revenue,
}).sort_values("order_date").reset_index(drop=True)

# planted problems (so anomaly detection + data quality have something to find)
sales.loc[310, ["quantity", "revenue"]] = [250, 250 * sales.loc[310, "unit_price"]]            # bulk-order spike
sales.loc[905, "revenue"] = 48500.00                                                           # typo / extra zeros
sales.loc[1422, "revenue"] = -899.00                                                           # refund booked as sale
sales.loc[1700, "unit_price"] = 9.99                                                           # price far below list
sales.loc[[50, 51, 700], "region"] = np.nan                                                    # missing region
sales.loc[[600, 601], "customer_id"] = np.nan
sales = pd.concat([sales, sales.iloc[[200]]], ignore_index=True)                               # exact duplicate row
sales.loc[820, "region"] = "europe "                                                           # inconsistent spelling

sales.to_csv(OUT / "sales.csv", index=False)
customers.to_csv(OUT / "customers.csv", index=False)
print(f"sales.csv: {len(sales)} rows | customers.csv: {len(customers)} rows -> {OUT}")
