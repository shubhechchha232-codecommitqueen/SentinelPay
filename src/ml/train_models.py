from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# SENTINELPAY
# MACHINE LEARNING TRAINING PIPELINE
# ============================================================

print("=" * 70)
print("          SENTINELPAY - ML TRAINING PIPELINE")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV_FILE = BASE_DIR / ".env"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

print("\n[1/9] Loading environment configuration...")

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)


DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


if not DB_NAME:
    raise ValueError("DB_NAME missing from .env")

if not DB_USER:
    raise ValueError("DB_USER missing from .env")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD missing from .env")


# ============================================================
# 3. CONNECT TO POSTGRESQL
# ============================================================

print("\n[2/9] Connecting to PostgreSQL...")


DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host="localhost",
    port=5432,
    database=DB_NAME
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


print("PostgreSQL connection ready.")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n[3/9] Loading transactions from PostgreSQL...")


query = """
SELECT *
FROM transactions
"""


df = pd.read_sql(
    query,
    engine
)


print(
    f"Loaded {len(df):,} transactions."
)


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

print("\n[4/9] Validating dataset...")


if "is_fraud" not in df.columns:

    raise ValueError(
        "Target column 'is_fraud' not found."
    )


print(
    f"Columns available: {len(df.columns)}"
)

print(
    f"Fraud cases: {df['is_fraud'].sum():,}"
)

print(
    f"Normal cases: {(df['is_fraud'] == 0).sum():,}"
)


# ============================================================
# 6. SELECT FEATURES
# ============================================================

print("\n[5/9] Preparing machine-learning features...")


FEATURE_COLUMNS = [

    "transaction_time",

    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",

    "amount",

    "amount_log",

    "amount_zscore",

    "transaction_frequency_24h",

    "avg_transaction_amount",

    "amount_deviation",

    "pca_mean",

    "pca_std",

    "pca_abs_mean"
]


missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]


if missing_features:

    print("\nMissing features:")
    print(missing_features)

    raise ValueError(
        "Required ML features are missing."
    )


X = df[
    FEATURE_COLUMNS
].copy()


y = df[
    "is_fraud"
].astype(int)


# ============================================================
# 7. HANDLE INFINITE / MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(
    X.median()
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n[6/9] Creating train/test split...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training transactions: {len(X_train):,}"
)

print(
    f"Testing transactions : {len(X_test):,}"
)

print(
    f"Training fraud cases  : {y_train.sum():,}"
)

print(
    f"Testing fraud cases   : {y_test.sum():,}"
)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

print("\nScaling numerical features...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n[7/9] Training Random Forest...")


rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=18,

    min_samples_leaf=2,

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


rf_model.fit(
    X_train_scaled,
    y_train
)


rf_probabilities = rf_model.predict_proba(
    X_test_scaled
)[:, 1]


rf_predictions = (
    rf_probabilities >= 0.50
).astype(int)


rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


rf_pr_auc = average_precision_score(
    y_test,
    rf_probabilities
)


print("\nRandom Forest Results")
print("-" * 50)

print(
    f"ROC-AUC : {rf_roc_auc:.4f}"
)

print(
    f"PR-AUC  : {rf_pr_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 11. XGBOOST
# ============================================================

print("\n[8/9] Training XGBoost...")


negative_cases = (
    y_train == 0
).sum()


positive_cases = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_cases /
    positive_cases
)


print(
    f"XGBoost scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


xgb_model = XGBClassifier(

    n_estimators=400,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


xgb_predictions = (
    xgb_probabilities >= 0.50
).astype(int)


xgb_roc_auc = roc_auc_score(
    y_test,
    xgb_probabilities
)


xgb_pr_auc = average_precision_score(
    y_test,
    xgb_probabilities
)


print("\nXGBoost Results")
print("-" * 50)

print(
    f"ROC-AUC : {xgb_roc_auc:.4f}"
)

print(
    f"PR-AUC  : {xgb_pr_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nXGBoost Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        xgb_predictions
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\nTop XGBoost Features")
print("-" * 50)


feature_importance = pd.DataFrame({

    "feature":
    FEATURE_COLUMNS,

    "importance":
    xgb_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE MODELS
# ============================================================

print("\n[9/9] Saving trained models...")


joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.pkl"
)


joblib.dump(
    xgb_model,
    MODEL_DIR / "xgboost_fraud_model.pkl"
)


joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.pkl"
)


# ============================================================
# 15. SAVE FEATURE LIST
# ============================================================

with open(
    MODEL_DIR / "feature_columns.json",
    "w"
) as file:

    json.dump(
        FEATURE_COLUMNS,
        file,
        indent=4
    )


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics = {

    "random_forest": {

        "roc_auc":
        float(rf_roc_auc),

        "pr_auc":
        float(rf_pr_auc)

    },

    "xgboost": {

        "roc_auc":
        float(xgb_roc_auc),

        "pr_auc":
        float(xgb_pr_auc)

    },

    "dataset": {

        "total_transactions":
        int(len(df)),

        "fraud_transactions":
        int(y.sum()),

        "fraud_percentage":
        float(y.mean() * 100)

    }

}


with open(
    MODEL_DIR / "model_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("             ML TRAINING COMPLETE")
print("=" * 70)

print("\nModels saved:")

print(
    MODEL_DIR / "random_forest.pkl"
)

print(
    MODEL_DIR / "xgboost_fraud_model.pkl"
)

print(
    MODEL_DIR / "feature_scaler.pkl"
)

print(
    MODEL_DIR / "feature_columns.json"
)

print(
    MODEL_DIR / "model_metrics.json"
)

print("\nSentinelPay supervised ML layer is ready.")

print("=" * 70)