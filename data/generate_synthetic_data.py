"""
Synthetic Data Generator — MSME Shop Creditworthiness Scoring Engine
----------------------------------------------------------------------
Generates realistic invoicing, payment, customer, and GST filing data
for 18-20 fictional MSME shops (textile/footwear retail), modeled on
TruBill's real production schema.

Each shop is assigned a hidden "health profile" (strong / average / weak)
that drives realistic behavior patterns — this profile is used later to
derive proxy creditworthiness labels for model training, and is NOT
exposed as a feature (it simulates the "ground truth" we don't have
real loan-default data for).

Output files (in /data):
  - shops.csv
  - customers.csv
  - invoices.csv
  - gst_filings.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta
import random
import uuid

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "/home/claude/msme-credit-score/data"
NUM_SHOPS = 18
HISTORY_MONTHS = 9          # each shop gets 6-12 months, avg 9
SIM_END_DATE = pd.Timestamp("2026-07-01")

SHOP_TYPES = ["Textile Retail", "Footwear Retail", "Textile Wholesale", "Footwear Wholesale"]
CITIES = ["Tiruppur", "Coimbatore", "Erode", "Karur"]

# Health profile controls behavior distributions.
# strong  -> consistent revenue, fast collections, high GST compliance
# average -> moderate variance, occasional late payments
# weak    -> volatile revenue, slow collections, poor GST compliance
HEALTH_PROFILES = {
    "strong":  {"weight": 0.30, "revenue_cv": 0.10, "late_payment_rate": 0.05,
                "avg_collection_days": 3,  "gst_ontime_rate": 0.95, "repeat_customer_rate": 0.65},
    "average": {"weight": 0.45, "revenue_cv": 0.25, "late_payment_rate": 0.20,
                "avg_collection_days": 10, "gst_ontime_rate": 0.75, "repeat_customer_rate": 0.40},
    "weak":    {"weight": 0.25, "revenue_cv": 0.45, "late_payment_rate": 0.45,
                "avg_collection_days": 25, "gst_ontime_rate": 0.45, "repeat_customer_rate": 0.20},
}


def pick_health_profile():
    profiles = list(HEALTH_PROFILES.keys())
    weights = [HEALTH_PROFILES[p]["weight"] for p in profiles]
    return random.choices(profiles, weights=weights, k=1)[0]


def generate_shops():
    shops = []
    for i in range(1, NUM_SHOPS + 1):
        shop_id = f"SHOP{i:03d}"
        health = pick_health_profile()
        months_active = random.randint(6, 12)
        shops.append({
            "shop_id": shop_id,
            "shop_name": f"{fake.last_name()} {random.choice(['Textiles', 'Footwear', 'Garments', 'Shoe Mart', 'Fashions'])}",
            "shop_type": random.choice(SHOP_TYPES),
            "city": random.choice(CITIES),
            "gst_registered": True,
            "onboarded_date": (SIM_END_DATE - timedelta(days=months_active * 30)).date(),
            "months_active": months_active,
            "_health_profile": health,  # hidden — used only for label generation, not a feature
        })
    return pd.DataFrame(shops)


def generate_customers(shops_df):
    customers = []
    for _, shop in shops_df.iterrows():
        profile = HEALTH_PROFILES[shop["_health_profile"]]
        num_customers = random.randint(20, 80)
        for j in range(num_customers):
            is_repeat = random.random() < profile["repeat_customer_rate"]
            customers.append({
                "customer_id": f"{shop['shop_id']}-CUST{j+1:03d}",
                "shop_id": shop["shop_id"],
                "customer_name": fake.name(),
                "is_repeat_customer": is_repeat,
            })
    return pd.DataFrame(customers)


def generate_invoices(shops_df, customers_df):
    invoices = []
    invoice_counter = 1

    for _, shop in shops_df.iterrows():
        profile = HEALTH_PROFILES[shop["_health_profile"]]
        shop_customers = customers_df[customers_df["shop_id"] == shop["shop_id"]]
        months_active = shop["months_active"]
        start_date = SIM_END_DATE - timedelta(days=months_active * 30)

        # base monthly revenue varies by shop, with seasonal + noise pattern
        base_monthly_revenue = np.random.uniform(80000, 400000)

        for month_offset in range(months_active):
            month_start = start_date + timedelta(days=month_offset * 30)

            # seasonal bump: Oct-Dec (festival season) gets a lift
            month_num = (month_start.month)
            seasonal_factor = 1.35 if month_num in [10, 11, 12] else 1.0

            # revenue for the month with configured variance (coefficient of variation)
            month_revenue = max(
                10000,
                np.random.normal(base_monthly_revenue * seasonal_factor,
                                  base_monthly_revenue * profile["revenue_cv"])
            )

            num_invoices_this_month = random.randint(15, 45)
            avg_invoice_value = month_revenue / num_invoices_this_month

            for _ in range(num_invoices_this_month):
                invoice_date = month_start + timedelta(days=random.randint(0, 29))
                if invoice_date > SIM_END_DATE:
                    continue

                amount = max(200, np.random.normal(avg_invoice_value, avg_invoice_value * 0.4))
                gst_rate = 0.05 if amount <= 1000 else 0.12
                gst_amount = round(amount * gst_rate, 2)

                customer = shop_customers.sample(1).iloc[0]

                is_late = random.random() < profile["late_payment_rate"]
                if is_late:
                    collection_days = int(np.random.normal(
                        profile["avg_collection_days"] * 3, profile["avg_collection_days"]))
                    collection_days = max(collection_days, profile["avg_collection_days"] + 5)
                else:
                    collection_days = max(0, int(np.random.normal(
                        profile["avg_collection_days"], 2)))

                payment_date = invoice_date + timedelta(days=collection_days)
                is_paid = payment_date <= SIM_END_DATE and random.random() > 0.03  # 3% bad debt

                invoices.append({
                    "invoice_id": f"INV{invoice_counter:06d}",
                    "shop_id": shop["shop_id"],
                    "customer_id": customer["customer_id"],
                    "invoice_date": invoice_date.date(),
                    "amount": round(amount, 2),
                    "gst_amount": gst_amount,
                    "total_amount": round(amount + gst_amount, 2),
                    "due_date": (invoice_date + timedelta(days=15)).date(),
                    "payment_status": "paid" if is_paid else "outstanding",
                    "payment_date": payment_date.date() if is_paid else None,
                    "collection_days": collection_days if is_paid else None,
                })
                invoice_counter += 1

    return pd.DataFrame(invoices)


def generate_gst_filings(shops_df):
    filings = []
    for _, shop in shops_df.iterrows():
        profile = HEALTH_PROFILES[shop["_health_profile"]]
        months_active = shop["months_active"]
        start_date = SIM_END_DATE - timedelta(days=months_active * 30)

        for month_offset in range(months_active):
            month_start = start_date + timedelta(days=month_offset * 30)
            filed_on_time = random.random() < profile["gst_ontime_rate"]
            filings.append({
                "shop_id": shop["shop_id"],
                "period": month_start.strftime("%Y-%m"),
                "filed_on_time": filed_on_time,
                "filing_date": (month_start + timedelta(days=random.randint(1, 25) if filed_on_time
                                                          else random.randint(26, 45))).date(),
            })
    return pd.DataFrame(filings)


def main():
    print("Generating shops...")
    shops_df = generate_shops()

    print("Generating customers...")
    customers_df = generate_customers(shops_df)

    print("Generating invoices...")
    invoices_df = generate_invoices(shops_df, customers_df)

    print("Generating GST filings...")
    gst_df = generate_gst_filings(shops_df)

    # Save hidden health profile separately (for label generation in Phase 3 only —
    # NOT to be used as a model feature, and not shipped with the "public" dataset)
    shops_public = shops_df.drop(columns=["_health_profile"])
    shops_with_health = shops_df.copy()

    shops_public.to_csv(f"{OUTPUT_DIR}/shops.csv", index=False)
    shops_with_health.to_csv(f"{OUTPUT_DIR}/_shops_with_health_profile.csv", index=False)
    customers_df.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    invoices_df.to_csv(f"{OUTPUT_DIR}/invoices.csv", index=False)
    gst_df.to_csv(f"{OUTPUT_DIR}/gst_filings.csv", index=False)

    print(f"\nDone.")
    print(f"  Shops:     {len(shops_df)}")
    print(f"  Customers: {len(customers_df)}")
    print(f"  Invoices:  {len(invoices_df)}")
    print(f"  GST rows:  {len(gst_df)}")


if __name__ == "__main__":
    main()
