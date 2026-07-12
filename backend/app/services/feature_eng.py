import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_growth_slope(group):
    months = np.arange(len(group))
    revenue = group["monthly_revenue"].values
    if len(months) < 2:
        return 0.0
    slope, _ = np.polyfit(months, revenue, deg=1)
    return slope


class FeatureEngineeringService:
    @staticmethod
    def calculate_features(
        invoices_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        gst_filings_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes raw transactional invoices, customers, and gst_filings DataFrames,
        and aggregates them to compute features per shop.
        """
        # Copy dataframes to avoid mutating original inputs
        invoices = invoices_df.copy()
        customers = customers_df.copy()
        gst_filings = gst_filings_df.copy()

        # Parse date columns
        invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"])

        # ---------- 1 & 6: Monthly revenue -> CV + growth slope ----------
        invoices["month"] = invoices["invoice_date"].dt.to_period("M")
        monthly_revenue = (
            invoices.groupby(["shop_id", "month"])["amount"].sum().reset_index()
        )
        monthly_revenue.columns = ["shop_id", "month", "monthly_revenue"]

        revenue_stats = (
            monthly_revenue.groupby("shop_id")["monthly_revenue"]
            .agg(revenue_mean="mean", revenue_std="std")
            .reset_index()
        )
        # Handle division by zero or NaN std
        revenue_stats["revenue_cv"] = (
            revenue_stats["revenue_std"] / revenue_stats["revenue_mean"]
        ).fillna(0.0)

        growth_stats = (
            monthly_revenue.sort_values(["shop_id", "month"])
            .groupby("shop_id")
            .apply(compute_growth_slope, include_groups=False)
            .reset_index()
        )
        growth_stats.columns = ["shop_id", "revenue_growth_slope"]

        # ---------- 2: Payment collection ratio ----------
        collection_stats = (
            invoices.groupby("shop_id")
            .apply(
                lambda x: pd.Series(
                    {
                        "total_invoiced": x["total_amount"].sum(),
                        "total_collected": x.loc[
                            x["payment_status"].str.lower() == "paid",
                            "total_amount",
                        ].sum(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        
        # Avoid division by zero
        collection_stats["payment_collection_ratio"] = (
            collection_stats["total_collected"] / collection_stats["total_invoiced"]
        ).fillna(0.0)

        # ---------- 3: Average DSO ----------
        dso_stats = (
            invoices.groupby("shop_id")["collection_days"].mean().reset_index()
        )
        dso_stats.columns = ["shop_id", "avg_dso"]

        # ---------- 4: Repeat customer rate ----------
        # Ensure is_repeat_customer is boolean/numeric
        customers["is_repeat_customer"] = customers["is_repeat_customer"].astype(
            float
        )
        repeat_stats = (
            customers.groupby("shop_id")["is_repeat_customer"]
            .mean()
            .reset_index()
        )
        repeat_stats.columns = ["shop_id", "repeat_customer_rate"]

        # ---------- 5: GST compliance rate ----------
        # Ensure filed_on_time is boolean/numeric
        gst_filings["filed_on_time"] = gst_filings["filed_on_time"].astype(float)
        gst_stats = (
            gst_filings.groupby("shop_id")["filed_on_time"].mean().reset_index()
        )
        gst_stats.columns = ["shop_id", "gst_compliance_rate"]

        # Merge everything into one feature table
        features = (
            revenue_stats[["shop_id", "revenue_cv"]]
            .merge(growth_stats, on="shop_id")
            .merge(
                collection_stats[["shop_id", "payment_collection_ratio"]],
                on="shop_id",
            )
            .merge(dso_stats, on="shop_id")
            .merge(repeat_stats, on="shop_id")
            .merge(gst_stats, on="shop_id")
        )

        # Fill any leftover NaN (e.g. no DSO because all outstanding)
        # Use a sensible fallback like global defaults or max DSO
        max_dso = features["avg_dso"].max()
        if pd.isna(max_dso) or np.isinf(max_dso):
            max_dso = 15.0  # reasonable fallback limit
        
        features["avg_dso"] = features["avg_dso"].fillna(max_dso)
        features["revenue_cv"] = features["revenue_cv"].fillna(0.0)

        return features
