"""
Phase 2 — Feature Engineering
MSME Shop Creditworthiness Scoring Engine
------------------------------------------
Builds one row per shop with 6 ML-ready features from raw
invoices.csv, customers.csv, and gst_filings.csv.

Features:
  1. revenue_cv              — coefficient of variation of monthly revenue
                               (lower = more stable)
  2. payment_collection_ratio— amount collected / amount invoiced
  3. avg_dso                 — average days sales outstanding
  4. repeat_customer_rate    — % of customers who are repeat buyers
  5. gst_compliance_rate     — % of GST filings submitted on time
  6. revenue_growth_slope    — trend of monthly revenue (polyfit slope)

Output: /data/features.csv (one row per shop)
"""

import os
import pandas as pd
import numpy as np

# Use path relative to this script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

print(f"Loading data from: {DATA_DIR}")
invoices = pd.read_csv(
    os.path.join(DATA_DIR, "invoices.csv"),
    parse_dates=["invoice_date"]
)
customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
gst_filings = pd.read_csv(os.path.join(DATA_DIR, "gst_filings.csv"))

# ---------- 1 & 6: Monthly revenue -> CV + growth slope ----------
invoices["month"] = invoices["invoice_date"].dt.to_period("M")
monthly_revenue = (
    invoices.groupby(["shop_id", "month"])["amount"].sum().reset_index()
)
monthly_revenue.columns = ["shop_id", "month", "monthly_revenue"]

revenue_stats = monthly_revenue.groupby("shop_id")["monthly_revenue"].agg(
    revenue_mean="mean", revenue_std="std"
).reset_index()
revenue_stats["revenue_cv"] = (
    revenue_stats["revenue_std"] / revenue_stats["revenue_mean"]
)


def compute_growth_slope(group):
    months = np.arange(len(group))
    revenue = group["monthly_revenue"].values
    if len(months) < 2:
        return 0.0
    slope, _ = np.polyfit(months, revenue, deg=1)
    return slope


# In pandas, grouping and applying a function returns a series indexed by shop
# We use include_groups=False to prevent warnings on newer pandas versions
growth_stats = (
    monthly_revenue.sort_values(["shop_id", "month"])
    .groupby("shop_id")
    .apply(compute_growth_slope, include_groups=False)
    .reset_index()
)
growth_stats.columns = ["shop_id", "revenue_growth_slope"]

# ---------- 2: Payment collection ratio ----------
collection_stats = invoices.groupby("shop_id").apply(
    lambda x: pd.Series({
        "total_invoiced": x["total_amount"].sum(),
        "total_collected": x.loc[
            x["payment_status"] == "paid", "total_amount"
        ].sum(),
    }),
    include_groups=False
).reset_index()
collection_stats["payment_collection_ratio"] = (
    collection_stats["total_collected"] / collection_stats["total_invoiced"]
)

# ---------- 3: Average DSO ----------
dso_stats = invoices.groupby("shop_id")["collection_days"].mean().reset_index()
dso_stats.columns = ["shop_id", "avg_dso"]

# ---------- 4: Repeat customer rate ----------
repeat_stats = (
    customers.groupby("shop_id")["is_repeat_customer"]
    .mean()
    .reset_index()
)
repeat_stats.columns = ["shop_id", "repeat_customer_rate"]

# ---------- 5: GST compliance rate ----------
gst_stats = (
    gst_filings.groupby("shop_id")["filed_on_time"]
    .mean()
    .reset_index()
)
gst_stats.columns = ["shop_id", "gst_compliance_rate"]

# ---------- Merge everything into one feature table ----------
features = (
    revenue_stats[["shop_id", "revenue_cv"]]
    .merge(growth_stats, on="shop_id")
    .merge(
        collection_stats[["shop_id", "payment_collection_ratio"]],
        on="shop_id"
    )
    .merge(dso_stats, on="shop_id")
    .merge(repeat_stats, on="shop_id")
    .merge(gst_stats, on="shop_id")
)

# Fill any leftover NaN (e.g. a shop with all outstanding invoices -> no DSO)
features["avg_dso"] = features["avg_dso"].fillna(features["avg_dso"].max())
features["revenue_cv"] = features["revenue_cv"].fillna(0)

output_file = os.path.join(DATA_DIR, "features.csv")
features.to_csv(output_file, index=False)

print("Feature table built:")
print(features.round(3))
print(f"\nSaved to {output_file}")
