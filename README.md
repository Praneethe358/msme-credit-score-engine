# MSME Shop Creditworthiness Scoring Engine

An ML-based alternative credit scoring platform for micro, small, and medium enterprise (MSME) shop owners who lack formal credit history.

## Overview
The system analyzes GST invoicing and payment behavior data (sourced from TruBill, a WhatsApp-native invoicing SaaS) to generate a credit score (300–900 scale), enabling fairer access to formal lending for small retail businesses in Tamil Nadu.

## Problem Statement
Small textile and footwear shop owners in Tiruppur/Coimbatore operate with consistent daily cash flow but have no formal credit history — no ITR filings, no bank statement trail, no collateral. This causes loan rejections from banks/NBFCs despite genuine repayment capacity, pushing them toward predatory informal lending. Meanwhile, GST billing platforms like TruBill already capture rich transactional data that goes unused for credit assessment.

## Solution
A standalone ML platform that:
1. Accepts exported shop invoicing/payment data (CSV)
2. Engineers financial behavior features (revenue consistency, payment collection ratio, customer retention, DSO, GST compliance regularity)
3. Runs a trained ML model to generate a creditworthiness score
4. Provides SHAP-based explainability ("why this score")
5. Suggests a loan eligibility tier as decision support for lenders

## Sector
FinTech / Financial Inclusion — Alternative Lending & MSME Credit Scoring

## Tech Stack
| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 (App Router) |
| **Backend** | FastAPI (Python) |
| **ML/Modeling** | scikit-learn, XGBoost |
| **Explainability** | SHAP |
| **Data Processing** | Pandas, NumPy |
| **Database** | Supabase (PostgreSQL) |
| **Deployment** | Vercel (frontend), Render (backend) |

## Directory Structure
```text
.
├── backend/            # FastAPI Python service
├── data/               # Datasets directory (raw, processed, sample exports)
├── docs/               # System reports, technical docs, and diagrams
├── frontend/           # Next.js web application
├── notebooks/          # Jupyter notebooks for EDA and model training
└── .gitignore          # Git exclusion rules
```

## Dataset
Due to TruBill being a WhatsApp-native invoicing SaaS, the platform processes exported transactional customer and billing logs to evaluate alternative financial metrics. For details on the expected CSV schema and the feature engineering process, please refer to the [Data README](./data/README.md).
