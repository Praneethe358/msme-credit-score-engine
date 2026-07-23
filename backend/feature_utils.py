"""
Shared Feature Engineering Module
------------------------------------
Same logic as Phase 2's feature_engineering.py, refactored into a
reusable function that works on in-memory DataFrames (not just fixed
file paths). Used by both the batch training pipeline and the live
FastAPI /score endpoint, so there's exactly one source of truth for
how features are computed.
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    "revenue_cv", "revenue_growth_slope", "payment_collection_ratio",
    "avg_dso", "repeat_customer_rate", "gst_compliance_rate"
]


def compute_growth_slope(group):
    months = np.arange(len(group))
    revenue = group["monthly_revenue"].values
    if len(months) < 2:
        return 0.0
    slope, _ = np.polyfit(months, revenue, deg=1)
    return slope


def compute_features_for_shop(invoices_df, customers_df, gst_df, shop_id="UPLOADED_SHOP"):
    """
    Computes the 6 credit-scoring features for a single shop from raw
    invoice/customer/GST data. Expects the same column schema as
    TruBill's exported data:

      invoices_df: invoice_date, amount, total_amount, payment_status, collection_days
      customers_df: is_repeat_customer
      gst_df: filed_on_time

    Returns: dict of {feature_name: value}
    """
    invoices_df = invoices_df.copy()
    invoices_df["invoice_date"] = pd.to_datetime(invoices_df["invoice_date"])
    invoices_df["month"] = pd.DatetimeIndex(invoices_df["invoice_date"]).to_period("M")

    # --- revenue_cv + revenue_growth_slope ---
    monthly_revenue = (
        invoices_df.groupby("month")["amount"].sum().reset_index()
    )
    monthly_revenue.columns = ["month", "monthly_revenue"]
    monthly_revenue = monthly_revenue.sort_values("month")

    revenue_mean = monthly_revenue["monthly_revenue"].mean()
    revenue_std = monthly_revenue["monthly_revenue"].std()
    revenue_cv = (revenue_std / revenue_mean) if revenue_mean else 0.0
    revenue_cv = 0.0 if pd.isna(revenue_cv) else revenue_cv

    revenue_growth_slope = compute_growth_slope(monthly_revenue)

    # --- payment_collection_ratio ---
    total_invoiced = invoices_df["total_amount"].sum()
    total_collected = invoices_df.loc[
        invoices_df["payment_status"] == "paid", "total_amount"
    ].sum()
    payment_collection_ratio = (
        total_collected / total_invoiced if total_invoiced else 0.0
    )

    # --- avg_dso ---
    avg_dso = invoices_df["collection_days"].mean()
    avg_dso = 30.0 if pd.isna(avg_dso) else avg_dso  # neutral default if no paid invoices

    # --- repeat_customer_rate ---
    if customers_df is not None and len(customers_df) > 0:
        repeat_customer_rate = customers_df["is_repeat_customer"].mean()
    else:
        repeat_customer_rate = 0.0

    # --- gst_compliance_rate ---
    if gst_df is not None and len(gst_df) > 0:
        gst_compliance_rate = gst_df["filed_on_time"].mean()
    else:
        gst_compliance_rate = 0.0

    return {
        "shop_id": shop_id,
        "revenue_cv": round(float(revenue_cv), 4),
        "revenue_growth_slope": round(float(revenue_growth_slope), 4),
        "payment_collection_ratio": round(float(payment_collection_ratio), 4),
        "avg_dso": round(float(avg_dso), 4),
        "repeat_customer_rate": round(float(repeat_customer_rate), 4),
        "gst_compliance_rate": round(float(gst_compliance_rate), 4),
    }
