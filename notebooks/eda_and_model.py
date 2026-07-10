#!/usr/bin/env python3
"""
MSME Credit Risk & Loan Eligibility Model Pipeline
This script simulates a Jupyter Notebook workflow for:
1. Exploratory Data Analysis (EDA)
2. Model Training & Validation
3. Exporting the final trained model for FastAPI backend serving
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_msme_data.csv")
MODEL_EXPORT_PATH = os.path.join(BASE_DIR, "backend", "app", "models", "msme_risk_model.joblib")

def load_and_inspect_data():
    print("=== 1. Loading & Inspecting Data ===")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please make sure it exists.")
        
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 records:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    return df

def perform_eda(df):
    print("\n=== 2. Exploratory Data Analysis ===")
    # Calculate correlation with target
    correlations = df.drop(columns=["business_id"]).corr()["approved"].sort_values(ascending=False)
    print("Correlation of features with Loan Approval ('approved'):")
    print(correlations)
    
    # Check class distribution
    class_dist = df["approved"].value_counts(normalize=True)
    print(f"\nClass distribution (Approved: 1, Rejected: 0):")
    for cls, pct in class_dist.items():
        print(f"Class {cls}: {pct:.2%}")

def train_and_evaluate(df):
    print("\n=== 3. Model Training & Validation ===")
    # Split features and target
    X = df.drop(columns=["business_id", "approved"])
    y = df["approved"]
    
    # Train/test split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    
    # Initialize Random Forest Classifier
    # Using small hyperparameters due to small dataset size
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Feature Importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importances:")
    for feature, weight in importances.items():
        print(f" - {feature}: {weight:.4f}")
        
    return model

def export_model(model):
    print("\n=== 4. Exporting Model for Backend ===")
    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_EXPORT_PATH)
    print(f"Successfully exported model to: {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    df = load_and_inspect_data()
    perform_eda(df)
    model = train_and_evaluate(df)
    export_model(model)
