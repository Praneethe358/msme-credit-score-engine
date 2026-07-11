# TruBill MSME Dataset Specification

This directory holds the raw, processed, and synthetic datasets derived from TruBill (a WhatsApp-native invoicing SaaS) exports. These datasets are used for alternative credit scoring and loan eligibility evaluation.

## Directory Structure
- `raw/`: Unprocessed CSV files exported directly from the TruBill database.
- `processed/`: Datasets after cleaning, imputation, and feature engineering.
- `synthetic/`: Generated synthetic datasets for testing or supplementing minor classes.

---

## Dataset Schema

The scoring engine expects exported CSV files from TruBill containing the following primary transactional invoice columns:

| Column Name | Type | Description |
|---|---|---|
| `business_id` | String | Unique identifier of the MSME shop owner / business. |
| `invoice_id` | String | Unique identifier of the customer invoice. |
| `customer_id` | String | Unique identifier of the client buying from the business. |
| `invoice_date` | Date (YYYY-MM-DD) | Date the invoice was generated and sent. |
| `invoice_amount` | Float | The total billing amount (including tax) in INR. |
| `gst_amount` | Float | The GST portion of the invoice amount in INR. |
| `payment_status` | String | Status of the payment (`PAID`, `UNPAID`, `OVERDUE`). |
| `payment_date` | Date (YYYY-MM-DD) | Date when the payment was settled (null if unpaid). |

---

## Engineered Financial Behavior Features

To assess creditworthiness without formal credit histories, the raw invoicing logs are aggregated at the `business_id` level to construct the following alternative credit features:

1. **Revenue Consistency (MoM)**
   - *Calculation*: Coefficient of variation (Standard Deviation / Mean) of month-over-month sales.
   - *Significance*: Low variance indicates stable cash flow and reliable income.

2. **Payment Collection Ratio**
   - *Calculation*: `Sum of PAID invoice amounts / Total generated invoice amounts`.
   - *Significance*: Measures the business's ability to successfully collect revenue from their customers.

3. **Days Sales Outstanding (DSO)**
   - *Calculation*: Average of `(payment_date - invoice_date)` for all paid invoices.
   - *Significance*: Highlights working capital health. Shorter collection times suggest higher efficiency and liquidity.

4. **Customer Retention Rate**
   - *Calculation*: `Count of active returning customers / Total unique customers` over a 6-month period.
   - *Significance*: High retention indicates robust customer loyalty and lower business risk.

5. **GST Compliance Regularity**
   - *Calculation*: Percentage of invoice uploads and GST calculations occurring on or before monthly filing deadlines.
   - *Significance*: Acts as a proxy for regulatory compliance and business transparency.
