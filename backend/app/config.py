import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths to data, model, and docs directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Model configurations
MODEL_PATH = os.path.join(MODEL_DIR, "credit_model.pkl")

# Data files configurations
LABELED_FEATURES_PATH = os.path.join(DATA_DIR, "labeled_features.csv")
SHOPS_PATH = os.path.join(DATA_DIR, "shops.csv")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
INVOICES_PATH = os.path.join(DATA_DIR, "invoices.csv")
GST_FILINGS_PATH = os.path.join(DATA_DIR, "gst_filings.csv")
