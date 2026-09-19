import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# SENTINELPAY - DATA PREPROCESSING PIPELINE
# ============================================================

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_PATH = Path("data/processed/transactions_clean.csv")


print("=" * 70)
print("        SENTINELPAY - DATA PREPROCESSING PIPELINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1/7] Loading raw dataset...")

df = pd.read_csv(RAW_PATH)

print(f"Loaded {len(df):,} transactions.")


# ------------------------------------------------------------
# 2. BASIC VALIDATION
# ------------------------------------------------------------

print("\n[2/7] Validating dataset...")

required_columns = (
    ["Time", "Amount", "Class"]
    + [f"V{i}" for i in range(1, 29)]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")

print(f"Missing values: {df.isnull().sum().sum()}")


# ------------------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------------------

print("\n[3/7] Removing duplicate transactions...")

before_duplicates = len(df)

df = df.drop_duplicates().reset_index(drop=True)

after_duplicates = len(df)

duplicates_removed = (
    before_duplicates - after_duplicates
)

print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Remaining transactions: {after_duplicates:,}")


# ------------------------------------------------------------
# 4. CREATE TRANSACTION IDENTIFIER
# ------------------------------------------------------------

print("\n[4/7] Creating transaction identifiers...")

df.insert(
    0,
    "transaction_id",
    [
        f"TXN_{i:07d}"
        for i in range(1, len(df) + 1)
    ]
)


# ------------------------------------------------------------
# 5. ADVANCED FEATURE ENGINEERING
# ------------------------------------------------------------

print("\n[5/7] Creating behavioral features...")


# Time converted to seconds from the beginning
df["time_seconds"] = df["Time"].astype(float)


# Log-transformed transaction amount
df["amount_log"] = np.log1p(df["Amount"])


# Amount z-score
amount_mean = df["Amount"].mean()
amount_std = df["Amount"].std()

df["amount_zscore"] = (
    (df["Amount"] - amount_mean)
    / amount_std
)


# Transaction frequency proxy
# Number of transactions observed in the previous
# 24-hour window based on the dataset's Time field.

df = df.sort_values("time_seconds").reset_index(drop=True)

time_values = df["time_seconds"].values

frequency_24h = []

window = 24 * 60 * 60

left = 0

for right in range(len(time_values)):

    while (
        time_values[right]
        - time_values[left]
        > window
    ):
        left += 1

    frequency_24h.append(
        right - left
    )

df["transaction_frequency_24h"] = frequency_24h


# ------------------------------------------------------------
# Rolling average transaction amount
# ------------------------------------------------------------

df["avg_transaction_amount"] = (
    df["Amount"]
    .rolling(window=20, min_periods=1)
    .mean()
)


# ------------------------------------------------------------
# Amount deviation from recent behavior
# ------------------------------------------------------------

df["amount_deviation"] = (
    df["Amount"]
    - df["avg_transaction_amount"]
)


# ------------------------------------------------------------
# PCA feature aggregation
# ------------------------------------------------------------

v_columns = [
    f"V{i}"
    for i in range(1, 29)
]

df["pca_mean"] = df[v_columns].mean(axis=1)

df["pca_std"] = df[v_columns].std(axis=1)

df["pca_abs_mean"] = (
    df[v_columns]
    .abs()
    .mean(axis=1)
)


# ------------------------------------------------------------
# 6. FINAL COLUMN SELECTION
# ------------------------------------------------------------

print("\n[6/7] Preparing final dataset...")

final_columns = [
    "transaction_id",
    "Time",
    "time_seconds",
]

final_columns += v_columns

final_columns += [
    "Amount",
    "amount_log",
    "amount_zscore",
    "transaction_frequency_24h",
    "avg_transaction_amount",
    "amount_deviation",
    "pca_mean",
    "pca_std",
    "pca_abs_mean",
    "Class",
]

df_final = df[final_columns].copy()


# ------------------------------------------------------------
# 7. SAVE PROCESSED DATA
# ------------------------------------------------------------

print("\n[7/7] Saving processed dataset...")

PROCESSED_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_final.to_csv(
    PROCESSED_PATH,
    index=False
)


# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("             PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nOriginal rows       : {before_duplicates:,}")
print(f"Duplicates removed  : {duplicates_removed:,}")
print(f"Final rows          : {len(df_final):,}")
print(f"Final columns       : {len(df_final.columns)}")

print("\nFraud distribution:")

print(
    df_final["Class"].value_counts()
)

print("\nFraud percentage:")

print(
    f"{df_final['Class'].mean() * 100:.4f}%"
)

print("\nSaved file:")

print(PROCESSED_PATH)

print("\n" + "=" * 70)